"""
Nifty Options Payoff Analyzer  (EDUCATIONAL USE ONLY - not investment advice)

Run:
    pip install streamlit yfinance numpy pandas plotly kiteconnect
    streamlit run nifty_options_analyzer.py
"""
import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from datetime import date

try:
    from kiteconnect import KiteConnect
except ImportError:
    KiteConnect = None

st.set_page_config(page_title="Nifty Options Analyzer", layout="wide")
st.title("📈 Nifty Options Payoff Analyzer")
st.caption("Educational tool only. Not SEBI-registered advice. Margin figures are rough estimates.")


# ---------- data ----------
@st.cache_data(ttl=300)
def get_spot(symbol):
    try:
        h = yf.Ticker(symbol).history(period="5d")
        return float(h["Close"].iloc[-1])
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def nfo_options(api_key, token):
    k = KiteConnect(api_key=api_key)
    k.set_access_token(token)
    df = pd.DataFrame(k.instruments("NFO"))
    return df[df.instrument_type.isin(["CE", "PE"])]


# ---------- pricing ----------
_erf = np.vectorize(math.erf)


def ncdf(x):
    return 0.5 * (1 + _erf(x / math.sqrt(2)))


def bs_price(S, K, T, r, iv, kind):
    S = np.asarray(S, dtype=float)
    if T <= 0 or iv <= 0:
        return np.maximum(S - K, 0) if kind == "CE" else np.maximum(K - S, 0)
    S = np.maximum(S, 1e-9)
    d1 = (np.log(S / K) + (r + iv ** 2 / 2) * T) / (iv * math.sqrt(T))
    d2 = d1 - iv * math.sqrt(T)
    if kind == "CE":
        return S * ncdf(d1) - K * math.exp(-r * T) * ncdf(d2)
    return K * math.exp(-r * T) * ncdf(-d2) - S * ncdf(-d1)


# ---------- sidebar ----------
with st.sidebar:
    st.header("Market inputs")
    symbol = st.selectbox("Index", ["^NSEI", "^NSEBANK"], format_func=lambda x: "NIFTY 50" if x == "^NSEI" else "BANK NIFTY")
    # ---- Zerodha Kite (optional, read-only: quotes + margin; no order placement) ----
    with st.expander("🔌 Zerodha Kite (optional)"):
        if KiteConnect is None:
            st.warning("Run: pip install kiteconnect")
        else:
            api_key = st.text_input("API key", key="k_api")
            if api_key:
                api_secret = st.text_input("API secret", type="password", key="k_sec")
                st.markdown(f"1. [Login to Kite]({KiteConnect(api_key=api_key).login_url()}) → copy `request_token` from the redirect URL")
                req = st.text_input("2. request_token", key="k_req")
                if st.button("Generate session") and req and api_secret:
                    try:
                        k = KiteConnect(api_key=api_key)
                        tok = k.generate_session(req, api_secret=api_secret)["access_token"]
                        k.set_access_token(tok)
                        st.session_state.update(kite=k, kite_key=api_key, kite_tok=tok)
                        st.success("Connected (token valid for today only)")
                    except Exception as e:
                        st.error(f"Login failed: {e}")
                tok_in = st.text_input("...or paste today's access_token", type="password", key="k_tok")
                if tok_in and st.button("Connect with token"):
                    k = KiteConnect(api_key=api_key)
                    k.set_access_token(tok_in)
                    st.session_state.update(kite=k, kite_key=api_key, kite_tok=tok_in)
                    st.success("Connected")
        st.caption("Keys stay in this session only. Never hard-code or share them.")

    kite = st.session_state.get("kite")
    kname = "NIFTY" if symbol == "^NSEI" else "BANKNIFTY"
    inst, expiry = None, None
    live = None
    if kite:
        try:
            inst = nfo_options(st.session_state.kite_key, st.session_state.kite_tok)
            inst = inst[inst["name"] == kname]
            expiry = st.selectbox("Expiry", sorted(inst.expiry.unique()))
            kq = "NSE:NIFTY 50" if kname == "NIFTY" else "NSE:NIFTY BANK"
            live = kite.ltp([kq])[kq]["last_price"]
        except Exception as e:
            st.error(f"Kite error (token expired?): {e}")
            kite, inst, expiry = None, None, None
    if live is None:
        live = get_spot(symbol)
    spot = st.number_input("Spot price", value=float(round(live, 2)) if live else 24500.0, step=50.0)
    if live is None:
        st.warning("Spot fetch failed - enter manually.")
    default_lot = int(inst.lot_size.iloc[0]) if inst is not None and len(inst) else (65 if symbol == "^NSEI" else 30)
    lot = st.number_input("Lot size" + (" (from Kite)" if inst is not None else " (verify on NSE)"), value=default_lot, step=1)
    if expiry is not None:
        dte = max((expiry - date.today()).days, 0)
        st.caption(f"Days to expiry: {dte}")
    else:
        dte = st.number_input("Days to expiry", value=7, min_value=0, step=1)
    iv = st.number_input("Implied volatility %", value=14.0, step=0.5) / 100
    rate = st.number_input("Risk-free rate %", value=6.5, step=0.25) / 100
    margin_pct = st.number_input("Naked-short margin % of notional (approx)", value=12.0, step=1.0) / 100

# ---------- strategy presets ----------
atm = round(spot / 50) * 50
step = 50 if symbol == "^NSEI" else 100


def kite_symbol(kind, strike):
    if inst is None or expiry is None:
        return None
    m = inst[(inst.expiry == expiry) & (inst.instrument_type == kind) & (inst.strike == float(strike))]
    return m.tradingsymbol.iloc[0] if len(m) else None


def kite_ltps(pairs):
    """pairs: list of (type, strike) -> {(type, strike): ltp}"""
    syms = {p: kite_symbol(*p) for p in pairs}
    keys = [f"NFO:{v}" for v in syms.values() if v]
    if not (kite and keys):
        return {}
    q = kite.ltp(keys)
    return {p: q[f"NFO:{v}"]["last_price"] for p, v in syms.items() if v and f"NFO:{v}" in q}


def preset(name):
    a = atm
    P = {
        "Custom": [],
        "Long Straddle": [("CE", "Buy", a, 1), ("PE", "Buy", a, 1)],
        "Short Straddle": [("CE", "Sell", a, 1), ("PE", "Sell", a, 1)],
        "Long Strangle": [("CE", "Buy", a + 2 * step, 1), ("PE", "Buy", a - 2 * step, 1)],
        "Short Strangle": [("CE", "Sell", a + 2 * step, 1), ("PE", "Sell", a - 2 * step, 1)],
        "Bull Call Spread": [("CE", "Buy", a, 1), ("CE", "Sell", a + 2 * step, 1)],
        "Bear Put Spread": [("PE", "Buy", a, 1), ("PE", "Sell", a - 2 * step, 1)],
        "Iron Condor": [("PE", "Buy", a - 4 * step, 1), ("PE", "Sell", a - 2 * step, 1),
                        ("CE", "Sell", a + 2 * step, 1), ("CE", "Buy", a + 4 * step, 1)],
        "Iron Butterfly": [("PE", "Buy", a - 3 * step, 1), ("PE", "Sell", a, 1),
                           ("CE", "Sell", a, 1), ("CE", "Buy", a + 3 * step, 1)],
    }
    rows = []
    try:
        live_p = kite_ltps([(t, k) for t, _, k, _ in P[name]]) if kite and P[name] else {}
    except Exception:
        live_p = {}
    for t, act, k, q in P[name]:
        prem = live_p.get((t, k)) or float(bs_price(spot, k, dte / 365, rate, iv, t))
        rows.append({"Type": t, "Action": act, "Strike": float(k), "Premium": round(prem, 2), "Lots": q})
    return pd.DataFrame(rows, columns=["Type", "Action", "Strike", "Premium", "Lots"])


name = st.selectbox("Strategy template (premiums auto-filled with Black-Scholes at your IV)",
                    ["Iron Condor", "Custom", "Long Straddle", "Short Straddle", "Long Strangle",
                     "Short Strangle", "Bull Call Spread", "Bear Put Spread", "Iron Butterfly"])
st.write("**Legs** - edit freely. Premiums are live Kite LTPs when connected, otherwise Black-Scholes estimates:")
st.session_state.setdefault("ver", 0)
if st.session_state.get("ov_name") != name:
    st.session_state.pop("override", None)
base = st.session_state["override"] if "override" in st.session_state else preset(name)
legs = st.data_editor(
    base, num_rows="dynamic", use_container_width=True,
    key=f"legs_{name}_{atm}_{dte}_{iv}_{st.session_state.ver}",
    column_config={
        "Type": st.column_config.SelectboxColumn(options=["CE", "PE"], required=True),
        "Action": st.column_config.SelectboxColumn(options=["Buy", "Sell"], required=True),
        "Strike": st.column_config.NumberColumn(step=step, required=True),
        "Premium": st.column_config.NumberColumn(step=0.05, required=True),
        "Lots": st.column_config.NumberColumn(min_value=1, step=1, default=1, required=True),
    })
legs = legs.dropna()
if kite and expiry is not None and not legs.empty:
    if st.button("🔄 Refresh LTPs from Kite for these legs"):
        try:
            got = kite_ltps([(r.Type, r.Strike) for r in legs.itertuples()])
            new = legs.copy()
            new["Premium"] = [got.get((r.Type, r.Strike), r.Premium) for r in legs.itertuples()]
            st.session_state.update(override=new, ov_name=name, ver=st.session_state.ver + 1)
            st.rerun()
        except Exception as e:
            st.error(f"Could not fetch quotes: {e}")
if legs.empty:
    st.info("Add at least one leg.")
    st.stop()

# ---------- payoff maths ----------
grid = np.linspace(0, spot * 3, 60001)


def payoff(S, T=0.0):
    total = np.zeros_like(S, dtype=float)
    for _, l in legs.iterrows():
        sign = 1 if l.Action == "Buy" else -1
        qty = l.Lots * lot
        val = bs_price(S, l.Strike, T, rate, iv, l.Type) if T > 0 else (
            np.maximum(S - l.Strike, 0) if l.Type == "CE" else np.maximum(l.Strike - S, 0))
        total += sign * qty * (val - l.Premium)
    return total


pnl = payoff(grid)
net_call = sum((1 if l.Action == "Buy" else -1) * l.Lots for _, l in legs.iterrows() if l.Type == "CE")

max_p, max_l = pnl.max(), pnl.min()
if net_call > 0:
    max_profit_txt = "Unlimited"
else:
    max_profit_txt = f"₹{max_p:,.0f}"
if net_call < 0:
    max_loss_txt = "Unlimited"
else:
    max_loss_txt = f"₹{max_l:,.0f}"

# breakevens
s = np.sign(pnl)
idx = np.where(s[:-1] * s[1:] < 0)[0]
bes = [grid[i] - pnl[i] * (grid[i + 1] - grid[i]) / (pnl[i + 1] - pnl[i]) for i in idx]
zero = grid[pnl == 0]
bes = sorted(set([round(b, 1) for b in bes] + [round(z, 1) for z in zero[:0]]))

# margin estimate
bounded = net_call >= 0
puts_short_net = -sum((1 if l.Action == "Buy" else -1) * l.Lots for _, l in legs.iterrows() if l.Type == "PE")
net_credit = -sum((1 if l.Action == "Buy" else -1) * l.Premium * l.Lots * lot for _, l in legs.iterrows())
if net_call >= 0 and puts_short_net <= 0 and max_l >= -1e12:
    # defined risk (long calls cover short calls; puts bounded by strike anyway) -> margin ~ max loss
    naked_lots = 0
else:
    naked_lots = max(0, -net_call) + max(0, puts_short_net)
if naked_lots:
    margin = naked_lots * lot * spot * margin_pct + max(0, -net_credit) * 0
    margin_note = "Naked short exposure - estimate = % of notional. Actual SPAN+exposure varies."
else:
    margin = abs(min(max_l, 0)) if net_credit >= 0 else abs(net_credit)
    margin = max(margin, abs(net_credit) if net_credit < 0 else 0)
    margin_note = "Defined-risk - estimate ≈ max loss (spread margin) / premium paid. Brokers often charge more."
    if net_credit >= 0 and max_l >= 0:
        margin = 0

# ---------- metrics ----------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Max profit", max_profit_txt)
c2.metric("Max loss", max_loss_txt)
c3.metric("Break-even(s)", ", ".join(f"{b:,.0f}" for b in bes) if bes else "None")
c4.metric("Est. margin required", f"₹{margin:,.0f}" if np.isfinite(margin) else "n/a")
st.caption(f"Net {'credit' if net_credit >= 0 else 'debit'}: ₹{abs(net_credit):,.0f}  |  {margin_note}")

# ---------- real margin from Zerodha ----------
if kite and expiry is not None:
    if st.button("🏦 Get actual margin from Zerodha"):
        orders = []
        for l in legs.itertuples():
            sym = kite_symbol(l.Type, l.Strike)
            if not sym:
                st.error(f"No contract found for {l.Type} {l.Strike:.0f} on {expiry}.")
                break
            orders.append(dict(exchange="NFO", tradingsymbol=sym, variety="regular", product="NRML",
                               order_type="MARKET", transaction_type="BUY" if l.Action == "Buy" else "SELL",
                               quantity=int(l.Lots * lot)))
        else:
            try:
                r = kite.basket_order_margins(orders, consider_positions=False)
                m1, m2 = st.columns(2)
                m1.metric("Margin without hedge benefit", f"₹{r['initial']['total']:,.0f}")
                m2.metric("Required margin (with hedge benefit)", f"₹{r['final']['total']:,.0f}")
                st.caption("From Zerodha's margin API. Treat as indicative - verify on Kite before trading, especially in high-VIX periods.")
            except Exception as e:
                st.error(f"Margin API error: {e}")

# ---------- chart ----------
lo, hi = spot * 0.85, spot * 1.15
x = np.linspace(lo, hi, 800)
fig = go.Figure()
y = payoff(x)
fig.add_trace(go.Scatter(x=x, y=np.where(y >= 0, y, np.nan), fill="tozeroy", mode="lines",
                         line=dict(color="green"), name="Profit at expiry", fillcolor="rgba(0,160,0,0.2)"))
fig.add_trace(go.Scatter(x=x, y=np.where(y < 0, y, np.nan), fill="tozeroy", mode="lines",
                         line=dict(color="red"), name="Loss at expiry", fillcolor="rgba(220,0,0,0.2)"))
if dte > 1:
    t_mid = max(dte / 365 / 2, 1 / 365)
    fig.add_trace(go.Scatter(x=x, y=payoff(x, t_mid), mode="lines",
                             line=dict(dash="dot", color="orange"), name="Theoretical, half-time"))
fig.add_vline(x=spot, line_dash="dash", annotation_text=f"Spot {spot:,.0f}")
for b in bes:
    if lo <= b <= hi:
        fig.add_vline(x=b, line_color="gray", line_dash="dot", annotation_text=f"BE {b:,.0f}")
fig.add_hline(y=0, line_color="black")
fig.update_layout(height=520, xaxis_title="Index at expiry", yaxis_title="P&L (₹)", hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

# probability-of-profit (lognormal, under IV)
T = max(dte, 0.5) / 365
z = (np.log(grid[1:] / spot) + (rate - iv ** 2 / 2) * T) / (iv * math.sqrt(T))
cdf = ncdf(z)
pdf = np.diff(np.concatenate([[0], cdf]))
pop = pdf[pnl[1:] > 0].sum() * 100
st.metric("Approx. probability of profit (lognormal model)", f"{pop:.1f}%")

with st.expander("Limitations - please read"):
    st.markdown("""
- **Without Kite:** yfinance gives index spot only, so premiums are Black-Scholes estimates. With Kite connected, premiums, lot size, expiries and margin come from Zerodha.
- **This tool never places orders.** It is read-only analysis.
- **Margin is a rough estimate.** Real margin uses NSE SPAN + exposure; use your broker's margin calculator for actual numbers.
- **No tool can guarantee "maximising profit".** Probability of profit is model-based (assumes lognormal returns, constant IV) and ignores fees, STT, slippage and gaps.
- Verify the current lot size and expiry rules on nseindia.com.
- Educational only; consult a SEBI-registered advisor before trading real money.
""")
