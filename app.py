import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import requests

st.set_page_config(
    page_title="台美股市多空儀表板",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
body { background-color: #0e1117; }
.metric-card {
    background: #1e2130;
    border-radius: 12px;
    padding: 16px 20px;
    margin: 6px 0;
    border-left: 4px solid #444;
}
.bull { border-left-color: #00c853 !important; }
.bear { border-left-color: #ff1744 !important; }
.neutral { border-left-color: #ffd600 !important; }
.big-label { font-size: 13px; color: #aaa; margin-bottom: 4px; }
.big-value { font-size: 26px; font-weight: 700; }
.bull-text { color: #00c853; }
.bear-text { color: #ff1744; }
.neutral-text { color: #ffd600; }
.signal-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 700;
    margin: 2px 4px;
}
.badge-bull { background: #00c85322; color: #00c853; border: 1px solid #00c853; }
.badge-bear { background: #ff174422; color: #ff1744; border: 1px solid #ff1744; }
.badge-neutral { background: #ffd60022; color: #ffd600; border: 1px solid #ffd600; }
.section-title {
    font-size: 16px;
    font-weight: 700;
    color: #e0e0e0;
    margin: 18px 0 8px 0;
    padding-bottom: 6px;
    border-bottom: 1px solid #333;
}
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=900)
def fetch_price_data(ticker, period="1y"):
    try:
        df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception:
        return None


@st.cache_data(ttl=900)
def fetch_fear_greed():
    try:
        url = "https://api.alternative.me/fng/?limit=30&format=json"
        r = requests.get(url, timeout=8)
        data = r.json()["data"]
        latest = data[0]
        return {
            "value": int(latest["value"]),
            "label": latest["value_classification"],
            "history": [int(d["value"]) for d in reversed(data)]
        }
    except Exception:
        return None


def calc_ma(df, windows):
    result = {}
    close = df["Close"].squeeze()
    for w in windows:
        if len(close) >= w:
            result[w] = float(close.rolling(w).mean().iloc[-1])
    return result


def calc_rsi(df, period=14):
    close = df["Close"].squeeze()
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1])


def calc_macd(df):
    close = df["Close"].squeeze()
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    return float(macd.iloc[-1]), float(signal.iloc[-1]), float(hist.iloc[-1])


def calc_bollinger(df, period=20):
    close = df["Close"].squeeze()
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = mid + 2 * std
    lower = mid - 2 * std
    price = float(close.iloc[-1])
    denom = upper.iloc[-1] - lower.iloc[-1]
    pct_b = float((price - lower.iloc[-1]) / denom) if denom != 0 else 0.5
    return float(upper.iloc[-1]), float(mid.iloc[-1]), float(lower.iloc[-1]), pct_b


def pct_change(df):
    close = df["Close"].squeeze()
    c = float(close.iloc[-1])
    p1d = float(close.iloc[-2]) if len(close) >= 2 else c
    p1m = float(close.iloc[-22]) if len(close) >= 22 else float(close.iloc[0])
    p3m = float(close.iloc[-66]) if len(close) >= 66 else float(close.iloc[0])
    return c, (c - p1d) / p1d * 100, (c - p1m) / p1m * 100, (c - p3m) / p3m * 100


def trend_signal(price, ma21, ma55, ma144):
    signals = []
    score = 0
    if price > ma21:
        signals.append(("多頭站上21MA", "bull"))
        score += 2
    else:
        signals.append(("空頭跌破21MA", "bear"))
        score -= 2

    if price > ma55:
        signals.append(("站上55MA", "bull"))
        score += 1
    else:
        signals.append(("跌破55MA", "bear"))
        score -= 1

    if price > ma144:
        signals.append(("站上144MA", "bull"))
        score += 2
    else:
        signals.append(("空頭跌破144MA", "bear"))
        score -= 2

    if ma21 > ma55 > ma144:
        signals.append(("均線多頭排列", "bull"))
        score += 2
    elif ma21 < ma55 < ma144:
        signals.append(("均線空頭排列", "bear"))
        score -= 2
    else:
        signals.append(("均線糾結整理", "neutral"))

    return score, signals


def score_to_label(score, max_score=7):
    pct = score / max_score
    if pct >= 0.5:
        return "多頭", "bull"
    elif pct <= -0.5:
        return "空頭", "bear"
    else:
        return "中性", "neutral"


def badge(text, kind):
    return f'<span class="signal-badge badge-{kind}">{text}</span>'


def metric_card(label, value, change=None, kind="neutral"):
    chg_html = ""
    if change is not None:
        color = "bull-text" if change >= 0 else "bear-text"
        arrow = "▲" if change >= 0 else "▼"
        chg_html = f'<div class="{color}" style="font-size:14px">{arrow} {abs(change):.2f}%</div>'
    return f"""
    <div class="metric-card {kind}">
        <div class="big-label">{label}</div>
        <div class="big-value {kind}-text">{value}</div>
        {chg_html}
    </div>
    """


def plot_candlestick(df, title, ma_dict):
    df2 = df.tail(180).copy()
    close_full = df["Close"].squeeze()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.75, 0.25], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(
        x=df2.index,
        open=df2["Open"].squeeze(), high=df2["High"].squeeze(),
        low=df2["Low"].squeeze(), close=df2["Close"].squeeze(),
        name="K線",
        increasing_line_color="#00c853", decreasing_line_color="#ff1744",
        increasing_fillcolor="#00c853", decreasing_fillcolor="#ff1744",
    ), row=1, col=1)
    for w, color, lbl in [(21,"#FFD600","MA21"),(55,"#FF9800","MA55"),(144,"#E91E63","MA144")]:
        ma_series = close_full.rolling(w).mean().tail(180)
        fig.add_trace(go.Scatter(x=df2.index, y=ma_series, mode="lines",
                                 line=dict(color=color, width=1.5), name=lbl), row=1, col=1)
    vol_colors = ["#00c853" if c >= o else "#ff1744"
                  for c, o in zip(df2["Close"].squeeze(), df2["Open"].squeeze())]
    fig.add_trace(go.Bar(x=df2.index, y=df2["Volume"].squeeze(),
                         marker_color=vol_colors, name="成交量", opacity=0.7), row=2, col=1)
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#e0e0e0")),
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", font=dict(color="#aaa"),
        xaxis_rangeslider_visible=False, margin=dict(l=10,r=10,t=40,b=10),
        legend=dict(orientation="h", y=1.05, font=dict(size=11)), height=450,
    )
    fig.update_xaxes(gridcolor="#1e2130", showgrid=True)
    fig.update_yaxes(gridcolor="#1e2130", showgrid=True)
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

st.markdown("## 📊 台美股市多空儀表板")
st.markdown(
    f"<small style='color:#666'>資料每15分鐘自動更新 · 最後更新：{datetime.now().strftime('%Y-%m-%d %H:%M')}</small>",
    unsafe_allow_html=True
)

col_refresh, _ = st.columns([1, 9])
with col_refresh:
    if st.button("🔄 立即更新"):
        st.cache_data.clear()
        st.rerun()

TICKERS = {
    "台灣加權指數": "^TWII",
    "S&P 500": "^GSPC",
    "那斯達克": "^IXIC",
    "費半 (SOX)": "^SOX",
    "道瓊": "^DJI",
    "VIX 恐慌": "^VIX",
    "美元指數": "DX-Y.NYB",
    "10Y 美債殖利率": "^TNX",
}

with st.spinner("載入市場數據中…"):
    data = {name: fetch_price_data(ticker) for name, ticker in TICKERS.items()}
    fg = fetch_fear_greed()

# ── 總覽 ──
st.markdown('<div class="section-title">🌐 全球市場總覽</div>', unsafe_allow_html=True)
overview_items = ["台灣加權指數","S&P 500","那斯達克","費半 (SOX)","道瓊","VIX 恐慌","美元指數","10Y 美債殖利率"]
cols = st.columns(4)
for i, name in enumerate(overview_items):
    df = data.get(name)
    with cols[i % 4]:
        if df is not None and not df.empty:
            price, chg1d, chg1m, chg3m = pct_change(df)
            kind = ("bear" if chg1d >= 0 else "bull") if name == "VIX 恐慌" else ("bull" if chg1d >= 0 else "bear")
            fmt = f"{price:,.2f}" if price > 100 else f"{price:.4f}"
            st.markdown(metric_card(name, fmt, chg1d, kind), unsafe_allow_html=True)
        else:
            st.markdown(metric_card(name, "N/A", kind="neutral"), unsafe_allow_html=True)

# ── 台股多空 ──
st.markdown('<div class="section-title">🇹🇼 台股加權指數多空判斷</div>', unsafe_allow_html=True)
tw_df = data.get("台灣加權指數")
if tw_df is not None and not tw_df.empty:
    tw_price = float(tw_df["Close"].squeeze().iloc[-1])
    tw_mas = calc_ma(tw_df, [5, 10, 21, 55, 144, 233])
    tw_rsi = calc_rsi(tw_df)
    _, _, tw_macd_hist = calc_macd(tw_df)
    _, _, _, boll_pct = calc_bollinger(tw_df)
    tw_score, tw_signals = trend_signal(tw_price, tw_mas.get(21,tw_price), tw_mas.get(55,tw_price), tw_mas.get(144,tw_price))
    if tw_rsi < 30: tw_signals.append(("RSI超賣(<30)","bull")); tw_score += 1
    elif tw_rsi > 70: tw_signals.append(("RSI超買(>70)","bear")); tw_score -= 1
    else: tw_signals.append((f"RSI中性({tw_rsi:.0f})","neutral"))
    if tw_macd_hist > 0: tw_signals.append(("MACD柱轉正","bull")); tw_score += 1
    else: tw_signals.append(("MACD柱轉負","bear")); tw_score -= 1
    tw_label, tw_kind = score_to_label(tw_score, 9)

    c1, c2, c3 = st.columns([2, 3, 3])
    with c1:
        st.markdown(f"""<div class="metric-card {tw_kind}" style="text-align:center;padding:24px 16px;">
            <div style="font-size:14px;color:#aaa;margin-bottom:8px">台股整體判斷</div>
            <div style="font-size:48px;font-weight:900;" class="{tw_kind}-text">{tw_label}</div>
            <div style="font-size:20px;color:#ccc;margin-top:6px">{tw_price:,.2f}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("**均線位置**")
        for w, lbl in [(5,"MA5"),(10,"MA10"),(21,"MA21"),(55,"MA55"),(144,"MA144"),(233,"MA233")]:
            v = tw_mas.get(w)
            if v:
                d = (tw_price-v)/v*100
                st.markdown(f'<span style="color:#aaa;font-size:13px">{lbl}: </span><span style="color:#ccc;font-size:13px">{v:,.0f}</span> <span class="{"bull-text" if d>=0 else "bear-text"}" style="font-size:12px">{"▲" if d>=0 else "▼"}{abs(d):.1f}%</span>', unsafe_allow_html=True)
    with c3:
        st.markdown("**技術訊號**")
        st.markdown("".join(badge(t,k) for t,k in tw_signals), unsafe_allow_html=True)
        st.markdown(f'<div style="margin-top:12px;font-size:13px;color:#aaa">RSI(14): <span style="color:#e0e0e0;font-weight:700">{tw_rsi:.1f}</span>&nbsp; MACD柱: <span style="color:{"#00c853" if tw_macd_hist>0 else "#ff1744"};font-weight:700">{tw_macd_hist:.1f}</span>&nbsp; 布林%B: <span style="color:#e0e0e0;font-weight:700">{boll_pct:.2f}</span></div>', unsafe_allow_html=True)
    st.plotly_chart(plot_candlestick(tw_df, "台灣加權指數 K線圖 (近180日)", tw_mas), use_container_width=True)

# ── 美股多空 ──
st.markdown('<div class="section-title">🇺🇸 美股多空判斷</div>', unsafe_allow_html=True)
cols_us = st.columns(3)
for col, (name, _) in zip(cols_us, [("S&P 500","^GSPC"),("那斯達克","^IXIC"),("費半 (SOX)","^SOX")]):
    df = data.get(name)
    with col:
        if df is not None and not df.empty:
            price, chg1d, chg1m, chg3m = pct_change(df)
            mas = calc_ma(df, [21, 55, 144])
            rsi_v = calc_rsi(df)
            _, _, macd_h = calc_macd(df)
            score, signals = trend_signal(price, mas.get(21,price), mas.get(55,price), mas.get(144,price))
            if rsi_v > 70: score -= 1; signals.append((f"RSI超買{rsi_v:.0f}","bear"))
            elif rsi_v < 30: score += 1; signals.append((f"RSI超賣{rsi_v:.0f}","bull"))
            score += 1 if macd_h > 0 else -1
            signals.append(("MACD轉正","bull") if macd_h > 0 else ("MACD轉負","bear"))
            label, kind = score_to_label(score, 9)
            arrow = "▲" if chg1d >= 0 else "▼"
            chg_color = "bull-text" if chg1d >= 0 else "bear-text"
            st.markdown(f"""<div class="metric-card {kind}" style="padding:16px">
                <div style="font-size:15px;color:#aaa">{name}</div>
                <div style="font-size:28px;font-weight:700;color:#e0e0e0">{price:,.2f}</div>
                <div class="{chg_color}">{arrow} {abs(chg1d):.2f}% (日)</div>
                <div style="font-size:22px;font-weight:700;margin:8px 0" class="{kind}-text">● {label}</div>
                <div style="font-size:12px;color:#888">1M: <span class="{"bull-text" if chg1m>=0 else "bear-text"}">{chg1m:+.1f}%</span>&nbsp; 3M: <span class="{"bull-text" if chg3m>=0 else "bear-text"}">{chg3m:+.1f}%</span></div>
                <div style="margin-top:8px">{"".join(badge(t,k) for t,k in signals[:4])}</div>
            </div>""", unsafe_allow_html=True)
            st.plotly_chart(plot_candlestick(df, f"{name} K線圖", mas), use_container_width=True)

# ── 市場情緒 ──
st.markdown('<div class="section-title">🌡️ 市場情緒指標</div>', unsafe_allow_html=True)
col_vix, col_fg, col_bond = st.columns(3)

with col_vix:
    vix_df = data.get("VIX 恐慌")
    if vix_df is not None:
        vix_price, vix_chg, _, _ = pct_change(vix_df)
        vix_kind = "bear" if vix_price > 25 else ("neutral" if vix_price > 18 else "bull")
        vix_label = "極度恐慌" if vix_price > 30 else ("恐慌" if vix_price > 25 else ("警戒" if vix_price > 18 else "低波動"))
        st.markdown(metric_card(f"VIX ({vix_label})", f"{vix_price:.2f}", vix_chg, vix_kind), unsafe_allow_html=True)
        vix_close = vix_df["Close"].squeeze().tail(90)
        fig_vix = go.Figure()
        fig_vix.add_trace(go.Scatter(x=vix_close.index, y=vix_close, mode="lines", fill="tozeroy",
                                     line=dict(color="#FF9800",width=2), fillcolor="rgba(255,152,0,0.15)"))
        for lvl, color, lbl in [(20,"#ffd600","20"),(30,"#ff1744","30")]:
            fig_vix.add_hline(y=lvl, line_dash="dash", line_color=color, annotation_text=lbl, annotation_font_color=color)
        fig_vix.update_layout(title="VIX 近90日", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                               font=dict(color="#aaa"), margin=dict(l=10,r=10,t=40,b=10), height=200, showlegend=False)
        fig_vix.update_xaxes(gridcolor="#1e2130"); fig_vix.update_yaxes(gridcolor="#1e2130")
        st.plotly_chart(fig_vix, use_container_width=True)

with col_fg:
    if fg:
        val = fg["value"]
        kind_fg = "bull" if val >= 60 else ("bear" if val <= 40 else "neutral")
        st.markdown(metric_card(f"Fear & Greed ({fg['label']})", str(val), kind=kind_fg), unsafe_allow_html=True)
        fig_fg = go.Figure(go.Indicator(
            mode="gauge+number", value=val, domain={"x":[0,1],"y":[0,1]},
            gauge={"axis":{"range":[0,100],"tickcolor":"#aaa"},"bar":{"color":"#ffd600"},
                   "steps":[{"range":[0,25],"color":"#ff1744"},{"range":[25,45],"color":"#ff9800"},
                             {"range":[45,55],"color":"#ffd600"},{"range":[55,75],"color":"#8bc34a"},
                             {"range":[75,100],"color":"#00c853"}],
                   "threshold":{"line":{"color":"white","width":3},"thickness":0.8,"value":val}},
            number={"font":{"color":"#e0e0e0","size":36}}
        ))
        fig_fg.update_layout(paper_bgcolor="#0e1117", font=dict(color="#aaa"), margin=dict(l=20,r=20,t=20,b=20), height=220)
        st.plotly_chart(fig_fg, use_container_width=True)

with col_bond:
    tnx_df = data.get("10Y 美債殖利率")
    if tnx_df is not None:
        tnx_price, tnx_chg, _, _ = pct_change(tnx_df)
        tnx_kind = "bear" if tnx_price > 4.5 else ("neutral" if tnx_price > 3.5 else "bull")
        st.markdown(metric_card("10Y美債殖利率", f"{tnx_price:.3f}%", tnx_chg, tnx_kind), unsafe_allow_html=True)
    usd_df = data.get("美元指數")
    if usd_df is not None:
        usd_price, usd_chg, _, _ = pct_change(usd_df)
        usd_kind = "neutral" if 100 <= usd_price <= 106 else ("bear" if usd_price > 106 else "bull")
        st.markdown(metric_card("美元指數 DXY", f"{usd_price:.2f}", usd_chg, usd_kind), unsafe_allow_html=True)

# ── 摘要表 ──
st.markdown('<div class="section-title">📋 多空分析摘要</div>', unsafe_allow_html=True)
summary_rows = []
for name in ["台灣加權指數","S&P 500","那斯達克","費半 (SOX)","道瓊"]:
    df = data.get(name)
    if df is not None and not df.empty:
        price, chg1d, chg1m, chg3m = pct_change(df)
        mas = calc_ma(df, [21, 55, 144])
        rsi_v = calc_rsi(df)
        _, _, macd_h = calc_macd(df)
        score, _ = trend_signal(price, mas.get(21,price), mas.get(55,price), mas.get(144,price))
        if rsi_v > 70: score -= 1
        elif rsi_v < 30: score += 1
        score += 1 if macd_h > 0 else -1
        label, _ = score_to_label(score, 9)
        summary_rows.append({
            "指數": name, "收盤價": f"{price:,.2f}",
            "日漲跌%": f"{chg1d:+.2f}%", "月漲跌%": f"{chg1m:+.2f}%", "季漲跌%": f"{chg3m:+.2f}%",
            "站上21MA": "✅" if price > mas.get(21,price) else "❌",
            "站上55MA": "✅" if price > mas.get(55,price) else "❌",
            "站上144MA": "✅" if price > mas.get(144,price) else "❌",
            "RSI": f"{rsi_v:.1f}", "判斷": label,
        })
if summary_rows:
    df_s = pd.DataFrame(summary_rows)
    def color_判斷(v):
        if v=="多頭": return "color:#00c853;font-weight:bold"
        if v=="空頭": return "color:#ff1744;font-weight:bold"
        return "color:#ffd600;font-weight:bold"
    st.dataframe(df_s.style.map(color_判斷, subset=["判斷"]), use_container_width=True, hide_index=True)

# ── 策略提示 ──
st.markdown('<div class="section-title">💡 系統策略提示</div>', unsafe_allow_html=True)
if tw_df is not None and not tw_df.empty:
    tw_p = float(tw_df["Close"].squeeze().iloc[-1])
    tw_m21 = calc_ma(tw_df,[21]).get(21, tw_p)
    tw_m144 = calc_ma(tw_df,[144]).get(144, tw_p)
    if tw_p > tw_m21:
        st.success(f"🟢 **台股多頭**：加權指數 {tw_p:,.0f} 站上21日均線 {tw_m21:,.0f}，趨勢偏多。可留意回測21MA支撐後的買點。")
    elif tw_p < tw_m144:
        st.error(f"🔴 **台股空頭**：加權指數 {tw_p:,.0f} 跌破144日均線 {tw_m144:,.0f}，趨勢偏空。建議降低持股比重。")
    else:
        st.warning(f"🟡 **台股中性整理**：指數介於21MA ({tw_m21:,.0f}) 與144MA ({tw_m144:,.0f}) 之間，建議觀望。")

vix_df2 = data.get("VIX 恐慌")
if vix_df2 is not None:
    vix_v = float(vix_df2["Close"].squeeze().iloc[-1])
    if vix_v > 30:
        st.error(f"⚠️ **市場極度恐慌 (VIX={vix_v:.1f})**：適合逢低分批佈局或等待恐慌消退。")
    elif vix_v < 15:
        st.info(f"😴 **市場自滿 (VIX={vix_v:.1f})**：低波動期，注意尾部風險，可適度減碼或買保護。")

st.markdown("---")
st.markdown("<div style='text-align:center;color:#555;font-size:12px'>資料來源：Yahoo Finance · alternative.me | 僅供參考，不構成投資建議</div>", unsafe_allow_html=True)
