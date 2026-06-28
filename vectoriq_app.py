import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone, date
import os
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 1. CONFIGURATION & BRANDING ---
st.set_page_config(
    page_title="Vector IQ",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS FOR ROUNDED CORNERS & CLEAN UI (Fixed Markdown Leaks & Font Clashes) ---
css = """<style>
.main-title { font-size: 3em; font-weight: bold; color: #FF4B4B; margin-bottom: -10px; }
.sub-title { font-size: 1.2em; color: var(--text-color); opacity: 0.7; margin-bottom: 20px; }
.date-banner { background-color: rgba(255, 75, 75, 0.1); color: var(--text-color); padding: 10px 15px; border-radius: 5px; border-left: 5px solid #FF4B4B; font-weight: bold; margin-bottom: 20px; }
.metric-box { padding: 12px; background-color: rgba(0, 255, 136, 0.1); color: #00FF88; border-radius: 8px; margin-bottom: 15px; font-weight: bold; text-align: center; border: 1px solid rgba(0, 255, 136, 0.3); font-size: 1.1em; }
[data-testid="stPlotlyChart"] { border-radius: 15px; overflow: hidden; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3); }
</style>"""
st.markdown(css, unsafe_allow_html=True)

# --- 2. DATA LOADING LAYER ---
@st.cache_data(show_spinner=False, ttl=1800)
def download_market_data(tickers):
    if not tickers:
        return pd.DataFrame()
    
    fixed_tickers = [t if t.endswith('.NS') or t == "^NSEI" else f"{t}.NS" for t in tickers]
    download_list = list(set(fixed_tickers + ["^NSEI"]))
    
    try:
        data = yf.download(
            download_list,
            period="2y", 
            group_by='ticker',
            threads=True,
            progress=False
        )
        if data.index.tz is not None:
            data.index = data.index.tz_localize(None)
            
        return data
    except Exception as e:
        st.error(f"Download API failed: {e}")
        return pd.DataFrame()

# --- 3. HELPER: REGIME TILE RENDERER ---
def render_regime_tile(title, value, series, threshold, positive=True, suffix=""):
    if pd.isna(value): return
    
    regime_color = "#00FF88" if (value >= threshold if positive else value <= threshold) else "#FF4B4B"

    series_5d = series.dropna().tail(5)
    if series_5d.empty: return
    
    dates_5d = series_5d.index.strftime('%Y-%m-%d').tolist() if hasattr(series_5d.index, 'strftime') else list(range(len(series_5d)))
    prev_val = series_5d.iloc[-2] if len(series_5d) >= 2 else value

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.65, 0.35], 
        vertical_spacing=0.15,    
        specs=[[{"type": "indicator"}], [{"type": "xy"}]]
    )
    
    fig.add_trace(go.Indicator(
        mode="number+delta",
        value=value,
        number={"suffix": suffix, "font": {"size": 36, "color": regime_color, "family": "Arial Black"}},
        delta={'reference': prev_val, 'relative': False, 'position': "right", 'valueformat': '.2f', 'font': {'size': 16}},
        title={"text": title.upper(), "font": {"size": 13, "color": "gray", "family": "Arial"}},
        align="left"
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=dates_5d, y=series_5d, mode='lines+markers',
        line=dict(width=3, color=regime_color), marker=dict(size=5, color=regime_color),
        fill='tozeroy', fillcolor=f"rgba{tuple(int(regime_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (0.1,)}",
        hovertemplate='<b>%{x}</b><br>Val: %{y:.2f}<extra></extra>' 
    ), row=2, col=1)

    fig.update_layout(
        height=125, margin=dict(l=20, r=20, t=25, b=15),
        template="plotly_dark", paper_bgcolor='#111827', plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(visible=False, fixedrange=True), yaxis=dict(visible=False, fixedrange=True),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

# --- 4. METRIC ENGINE ---
def calculate_advanced_metrics(df, bench_series):
    if df.empty or len(df) < 260: return None

    close, high, low, volume = df['Close'], df['High'], df['Low'], df['Volume']
    c = close.iloc[-1]
    
    sma50, sma150, sma200 = close.rolling(50).mean(), close.rolling(150).mean(), close.rolling(200).mean()
    high_52w = high.rolling(252).max()
    h52 = high_52w.iloc[-1]
    dist_52w_high_pct = ((h52 - c) / h52) * 100 if h52 > 0 else 100

    s50, s150, s200 = sma50.iloc[-1], sma150.iloc[-1], sma200.iloc[-1]
    d50 = (c - s50) / s50 if s50 > 0 else 0
    d150 = (c - s150) / s150 if s150 > 0 else 0
    d200 = (c - s200) / s200 if s200 > 0 else 0
    ma_dist_score = min(max((d50 + d150 + d200) * 200, 0), 50)
    
    spread = (s50 - s200) / s200 if s200 > 0 else 0
    alignment_score = min(max(spread * 300, 0), 30)
    
    s200_prev_20 = sma200.iloc[-20]
    slope = (s200 - s200_prev_20) / s200_prev_20 if s200_prev_20 > 0 else 0
    slope_score = min(max(slope * 500, 0), 20)
    trend_score = ma_dist_score + alignment_score + slope_score

    bench_series = bench_series.ffill()
    rs_line = close / bench_series
    rs_curr, rs_prev_20 = rs_line.iloc[-1], rs_line.iloc[-20]
    rs_mom = (rs_curr - rs_prev_20) / rs_prev_20 if rs_prev_20 > 0 else 0
    rs_mom_norm = min(max(rs_mom * 10, 0), 1) * 100 
    rs_raw = rs_curr

    def get_range(window):
        h, l = high.tail(window).max(), low.tail(window).min()
        return (h - l) / h if h > 0 else 1

    r20, r60 = get_range(20), get_range(60)
    compression_ratio = r20 / r60 if r60 > 0 else 1
    tight_score = min(max((1 - compression_ratio) * 100, 0), 100)

    v, v_5d, v_50d = volume.iloc[-1], volume.tail(5).mean(), volume.rolling(50).mean().iloc[-1]
    
    if v_50d > 0:
        dry_score = min(max((1 - (v_5d / v_50d)) * 100, 0), 50)
        vol_expansion = v / v_50d 
    else:
        dry_score, vol_expansion = 0, 0
        
    vol_score = dry_score + min(vol_expansion * 10, 50)
    readiness_score = min(max((1 - ((h52 - c) / h52 / 0.10)) * 100, 0), 100) if h52 > 0 else 0

    tight_len, max_rng_pct = 10, 12
    rolling_high_10 = high.rolling(tight_len).max().iloc[-1]
    rolling_low_10 = low.rolling(tight_len).min().iloc[-1]
    
    vcp_range_pct = ((rolling_high_10 - rolling_low_10) / rolling_high_10) * 100 if rolling_high_10 > 0 else None
    is_tight = vcp_range_pct < max_rng_pct if vcp_range_pct is not None else False

    vol_dry_check = v_5d < v_50d if v_50d > 0 else False
    trend_ok = (c > s150) and (s50 > s200)
    basic_vcp = is_tight and vol_dry_check and trend_ok

    base_len = 45
    segment = max(1, base_len // 3)
    elite_vcp = False
    
    if len(df) >= base_len:
        try:
            h1, l1 = high.iloc[-segment*3:-segment*2].max(), low.iloc[-segment*3:-segment*2].min()
            h2, l2 = high.iloc[-segment*2:-segment].max(), low.iloc[-segment*2:-segment].min()
            h3, l3 = high.iloc[-segment:].max(), low.iloc[-segment:].min()
            
            def depth(h, l): return ((h - l) / h) * 100 if h > 0 else 0
            d1, d2, d3 = depth(h1, l1), depth(h2, l2), depth(h3, l3)
            elite_vcp = basic_vcp and (d1 > d2) and (d2 > d3) and (d3 < 12)            
        except:
            elite_vcp = False

    base_low, base_high = low.iloc[-base_len:].min(), high.iloc[-base_len:].max()
    dist_to_breakout = ((base_high - c) / base_high) * 100 if base_high > 0 else 100

    in_base = (close >= base_low) & (close <= base_high)
    base_duration = min(int(in_base.astype(int).iloc[::-1].cummin().sum()), base_len * 2)

    sma20 = close.rolling(20).mean().iloc[-1]
    failure_score = 0
    if vol_expansion < 1.3: failure_score += 30
    h_day, l_day = high.iloc[-1], low.iloc[-1]
    if c < sma20: failure_score += 20
    if (h_day - l_day) > 0 and ((c - l_day)/(h_day - l_day)) < 0.5: failure_score += 20
    if sma20 > 0 and ((c - sma20)/sma20) > 0.15: failure_score += 20
    failure_risk = min(failure_score, 100)

    persist_score = ((close.diff() > 0).tail(20).sum() / 20) * 40
    if rs_curr > rs_prev_20: persist_score += 20
    persist_score += min(((high.diff() > 0).tail(20).sum() / 20) * 30, 30)
    if r20 < 0.10: persist_score += 10

    breakout_20d = c > high.rolling(20).max().shift(1).iloc[-1]
    breakout_50d = c > high.rolling(50).max().shift(1).iloc[-1]
    extension = (c - s200) / s200 if s200 > 0 else 0
    
    stage2_candidate = (c > s200) and (s50 > s200) and (s200 > sma200.iloc[-20]) and \
                       (breakout_20d or breakout_50d) and (0.02 < extension < 0.20) and (vol_expansion >= 1.3)

    return {
        "Ticker": "", "Price": c, "Trend Score": int(trend_score),
        "RS Raw": rs_raw, "RS Mom Score": rs_mom_norm, "Tightness %": int(tight_score), 
        "Vol Dry Score": int(vol_score), "Near Breakout": int(readiness_score),
        "Failure Risk": int(failure_risk), "Persistence": int(min(persist_score, 100)),
        "Vol Expansion": round(vol_expansion, 2), "Breakout 20D": breakout_20d,
        "Stage2_Candidate": stage2_candidate, "Basic VCP": basic_vcp, "Elite VCP": elite_vcp,
        "VCP Range %": round(vcp_range_pct, 2) if vcp_range_pct is not None else 0,
        "Dist to Breakout %": round(dist_to_breakout, 2), "Within 25% 52W High": dist_52w_high_pct <= 25,
        "Base Duration": base_duration 
    }

# --- 5. BREADTH ENGINE ---
def calculate_market_breadth(raw_data, start_date, end_date):
    if isinstance(raw_data.columns, pd.MultiIndex):
        stock_data = raw_data.drop(columns=["^NSEI"], level=0, errors='ignore')
    else: return []

    try:
        closes, highs, lows = stock_data.xs('Close', level=1, axis=1), stock_data.xs('High', level=1, axis=1), stock_data.xs('Low', level=1, axis=1)
        
        sma20, sma200 = closes.rolling(20).mean(), closes.rolling(200).mean()
        above_20dma, above_200dma = (closes > sma20), (closes > sma200)
        
        roll_high_252, roll_low_252 = highs.rolling(252).max(), lows.rolling(252).min() 
        is_new_high, is_new_low = (highs >= roll_high_252), (lows <= roll_low_252)
        
        daily_diff = closes.diff()
        ad_line = (daily_diff > 0).sum(axis=1) - (daily_diff < 0).sum(axis=1)
        ad_line = ad_line.cumsum() 
        
        breakout_10d_ago = (closes > highs.rolling(20).max().shift(1)).shift(10)
        successful_breakout = breakout_10d_ago & (closes.pct_change(10) > 0)
        
        rolling_attempts = breakout_10d_ago.sum(axis=1).rolling(10, min_periods=3).sum()
        rolling_successes = successful_breakout.sum(axis=1).rolling(10, min_periods=3).sum()
        
        rolling_bo_success_series = pd.Series(np.where(rolling_attempts > 0, (rolling_successes / rolling_attempts) * 100, np.nan), index=closes.index)
        break_below_20dma = (closes < sma20) & (closes.shift(1) > sma20.shift(1))
        
        valid_dates = closes.index[(closes.index.date >= start_date) & (closes.index.date <= end_date)]
        breadth_records = []
        
        for d in valid_dates:
            total_valid_stocks = closes.loc[d].count()
            if total_valid_stocks == 0: continue
            
            idx_loc = closes.index.get_loc(d)
            slope_val = np.nan
            if idx_loc >= 20:
                y = ad_line.iloc[idx_loc-19 : idx_loc+1].values
                if len(y) == 20:
                    denom = abs(ad_line.iloc[idx_loc-20])
                    slope_val = (np.polyfit(np.arange(len(y)), y, 1)[0] / denom) * 100 if denom > 0 else 0

            nh, nl = is_new_high.loc[d].sum(), is_new_low.loc[d].sum()
            bo_val = rolling_bo_success_series.loc[d]

            breadth_records.append({
                "Date": d.date(),
                "% Above 20 DMA": round((above_20dma.loc[d].sum() / total_valid_stocks) * 100, 2),
                "% Above 200 DMA": round((above_200dma.loc[d].sum() / total_valid_stocks) * 100, 2),
                "New Highs": int(nh), "New Lows": int(nl), "Net New Highs": int(nh - nl),
                "AD Line": int(ad_line.loc[d]), "AD Slope 20D": round(slope_val, 2) if not np.isnan(slope_val) else np.nan,
                "AD Change 20D": round(ad_line.iloc[idx_loc] - ad_line.iloc[idx_loc - 20], 2) if idx_loc >= 20 else np.nan,
                "Rolling BO Success 10D": round(bo_val, 2) if not np.isnan(bo_val) else np.nan,
                "% Breaking < 20 DMA": round((break_below_20dma.loc[d].sum() / total_valid_stocks) * 100, 1)
            })
            
        return breadth_records
    except Exception:
        return []

# --- 6. SCANNER ORCHESTRATOR ---
def scan_stocks(tickers, start_date, end_date, progress_bar, status_text):
    status_text.text("🔌 Downloading Data...")
    raw_data = download_market_data(tickers)
    if raw_data.empty: return [], [], [], [], [], [] 
    
    status_text.text("📊 Calculating Breadth...")
    breadth_data = calculate_market_breadth(raw_data, start_date, end_date)
    bench_data = raw_data["^NSEI"]['Close'] if "^NSEI" in raw_data.columns.levels[0] else pd.Series()

    ath_results, pop_results, stage2_results, current_state_results, cmbf_results = [], [], [], [], []
    downloaded_tickers = [col[0] for col in raw_data.columns] if isinstance(raw_data.columns, pd.MultiIndex) else []
    downloaded_tickers = list(set([t for t in downloaded_tickers if t != "^NSEI"]))
    
    for idx, ticker in enumerate(downloaded_tickers):
        progress_bar.progress((idx + 1) / len(downloaded_tickers))
        status_text.text(f"Analyzing {ticker}...")
        
        try:
            df = raw_data[ticker].copy()
            df['Close'] = df['Close'].ffill()
            valid_closes = df['Close'].dropna()
            if valid_closes.empty or len(df) < 260: continue
            
            current_ltp = float(valid_closes.iloc[-1])
            df['Vol_MA50'] = df['Volume'].rolling(50).mean()
            if (df['Close'] * df['Volume']).rolling(50).mean().iloc[-1] < 1_00_000_000: continue

            df['SMA20'], df['SMA50'], df['SMA200'] = df['Close'].rolling(20).mean(), df['Close'].rolling(50).mean(), df['Close'].rolling(200).mean()
            df['High_20'], df['High_50'] = df['High'].rolling(20).max().shift(1), df['High'].rolling(50).max().shift(1)
            df['SMA200_Rising'] = df['SMA200'] > df['SMA200'].shift(20)

            bench_aligned = bench_data.reindex(df.index).ffill()
            df['RS_Line'] = df['Close'] / bench_aligned
            df['RS_Mom_Ratio'] = df['RS_Line'] / df['RS_Line'].shift(20)
            df['RS_Score'] = (((df['RS_Line'] - df['RS_Line'].shift(20)) / df['RS_Line'].shift(20)) * 10).clip(0, 1) * 100

            df['Vol_Exp'] = df['Volume'] / df['Vol_MA50']
            close_loc = (df['Close'] - df['Low']) / (df['High'] - df['Low']).replace(0, np.nan)
            
            df['Failure_Risk'] = (np.where(df['Vol_Exp'] < 1.2, 30, 0) + np.where(((df['High'] - df['Low']) > 0) & (close_loc < 0.5), 20, 0) + np.where((df['SMA20'] > 0) & ((df['Close'] - df['SMA20'])/df['SMA20'] > 0.15), 20, 0)).clip(max=100)
            df['Persist'] = (((df['Close'].diff() > 0).rolling(20).sum() / 20) * 40 + ((df['High'].diff() > 0).rolling(20).sum() / 20 * 30).clip(upper=30) + np.where(df['Vol_Exp'] > 1.2, 10, 0)).clip(max=100)

            df['EMA20'], df['EMA50'], df['EMA200'] = df['Close'].ewm(span=20, adjust=False).mean(), df['Close'].ewm(span=50, adjust=False).mean(), df['Close'].ewm(span=200, adjust=False).mean()
            df['Near_High'] = df['Close'] >= 0.95 * df['High'].rolling(252).max()
            df['RS_Pass'] = (df['RS_Line'].pct_change(20) > 0) & (df['RS_Line'].pct_change(50) > 0)
            df['EMA_Stack'] = (df['Close'] > df['EMA20']) & (df['EMA20'] > df['EMA50']) & (df['EMA50'] > df['EMA200'])
            
            gain = df['Close'].diff().clip(lower=0).rolling(14).mean()
            loss = -df['Close'].diff().clip(upper=0).rolling(14).mean()
            df['Daily_RSI'] = 100 - (100 / (1 + (gain / loss)))

            weekly = df['Close'].resample("W").last()
            w_gain, w_loss = weekly.diff().clip(lower=0).rolling(14).mean(), -weekly.diff().clip(upper=0).rolling(14).mean()
            df['Weekly_RSI'] = (100 - (100 / (1 + (w_gain / w_loss)))).reindex(df.index, method='ffill').fillna(50)

            df['Box_Range_Pct'] = (df['High'].rolling(20).max() - df['Low'].rolling(20).min()) / df['Low'].rolling(20).min().replace(0, np.nan)
            df['Box_Pass'] = df['Box_Range_Pct'] < 0.08
            df['Vol_Pass'] = (df['Volume'] * (df['Close'] > df['Close'].shift(1))).rolling(20).sum() > (df['Volume'] * (df['Close'] < df['Close'].shift(1))).rolling(20).sum()

            tr = np.maximum(df['High'] - df['Low'], np.maximum(abs(df['High'] - df['Close'].shift()), abs(df['Low'] - df['Close'].shift())))
            df['ATR_Pct'] = tr.rolling(14).mean() / df['Close'].replace(0, np.nan)

            range_df = df.loc[(df.index.date >= start_date) & (df.index.date <= end_date)]
            if range_df.empty: continue

            # --- CHECKS ---
            for event_date, row in range_df[range_df['Close'] > df['High'].expanding().max().shift(1).loc[range_df.index]].iterrows():
                ath_results.append({
                    "Ticker": ticker.replace(".NS", ""), "ATH Date": event_date.date(), "ATH_Event_Price": float(row["Close"]), "LTP": current_ltp,
                    "ATH_Return": float(((current_ltp - float(row["Close"])) / float(row["Close"])) * 100) if float(row["Close"]) > 0 else 0.0,
                    "ATH_Vol_Expansion": round(row["Vol_Exp"], 2), "ATH_Failure_Risk": int(row["Failure_Risk"]), "ATH_Persistence": int(row["Persist"]), "RS Rating": int(row["RS_Score"])
                })

            for event_date, row in range_df[(range_df['Close'] > range_df['High_20']) & (range_df['Volume'] > 1.2 * range_df['Vol_MA50'])].iterrows():
                pop_results.append({
                    "Ticker": ticker.replace(".NS", ""), "Pop Date": event_date.date(), "Pop_Event_Price": float(row["Close"]), "LTP": current_ltp,
                    "Pop_Return": float(((current_ltp - float(row["Close"])) / float(row["Close"])) * 100) if float(row["Close"]) > 0 else 0.0,
                    "Pop_Vol_Expansion": round(row["Vol_Exp"], 2), "Pop_Failure_Risk": int(row["Failure_Risk"]), "Pop_Persistence": int(row["Persist"]), "RS Rating": int(row["RS_Score"])
                })

            s2_dist = (range_df['Close'] - range_df['SMA200']) / range_df['SMA200']
            for event_date, row in range_df[(range_df['Close'] > range_df['SMA200']) & (range_df['SMA50'] > range_df['SMA200']) & (range_df['SMA200_Rising']) & ((range_df['Close'] > range_df['High_50']) | (range_df['Close'] > range_df['High_20'])) & (range_df['Volume'] > 1.3 * range_df['Vol_MA50']) & (s2_dist > 0.02) & (s2_dist < 0.20)].iterrows():
                if (row["RS_Score"] >= 60) and (row["Persist"] >= 50):
                    quality = "High" if row["RS_Score"] >= 75 and row["Persist"] >= 65 and row["Failure_Risk"] < 30 else ("Medium" if row["RS_Score"] >= 60 else "Low")
                    stage2_results.append({
                        "Ticker": ticker.replace(".NS", ""), "Stage2 Date": event_date.date(), "S2_Event_Price": float(row["Close"]), "LTP": current_ltp,
                        "S2_Return": float(((current_ltp - float(row["Close"])) / float(row["Close"])) * 100) if float(row["Close"]) > 0 else 0.0,
                        "S2_Vol_Expansion": round(row["Vol_Exp"], 2), "S2_Failure_Risk": int(row["Failure_Risk"]), "S2_Persistence": int(row["Persist"]), "Event RS Mom": round(row["RS_Mom_Ratio"], 4), "RS Rating": int(row["RS_Score"]), "Signal Quality": quality
                    })

            if curr_metrics := calculate_advanced_metrics(df.loc[:end_date], bench_data.reindex(df.loc[:end_date].index).ffill()):
                curr_metrics["Ticker"] = ticker.replace(".NS", "")
                current_state_results.append(curr_metrics)

            for event_date, row in range_df[range_df['Near_High'] & range_df['RS_Pass'] & range_df['EMA_Stack'] & range_df['Box_Pass']].iterrows():
                score = int((row['Daily_RSI'] > 60) and (row['Weekly_RSI'] > 60)) + int(row['Vol_Pass'])
                if score > 0:
                    cmbf_results.append({
                        "Ticker": ticker.replace(".NS", ""), "CMBF Date": event_date.date(), "Price": float(row['Close']),
                        "Grade": "A" if score == 2 else "B", "Protocol": "Kinetic" if row['ATR_Pct'] < 0.04 else "Affordable",
                        "Daily RSI": round(row['Daily_RSI'], 1), "Weekly RSI": round(row['Weekly_RSI'], 1), "Box Range %": round(row['Box_Range_Pct'] * 100, 2), "ATR %": round(row['ATR_Pct'] * 100, 2)
                    })

        except Exception: 
            continue

    df_current = pd.DataFrame(current_state_results)
    if not df_current.empty:
        df_current['RS %'] = ((df_current['RS Raw'].rank(pct=True) * 100 * 0.7) + (df_current['RS Mom Score'] * 0.3)).fillna(0).astype(int)
        df_current['Rocket Score'] = (0.30 * df_current['Trend Score'] + 0.25 * df_current['RS %'] + 0.20 * df_current['Tightness %'] + 0.15 * df_current['Vol Dry Score'] + 0.10 * df_current['Near Breakout']).round(2)
        
    return ath_results, df_current.to_dict('records') if not df_current.empty else [], pop_results, breadth_data, pd.DataFrame(stage2_results).to_dict('records') if stage2_results else [], cmbf_results

# --- 7. UTILS: STYLING ---
def apply_text_styling(val, mode='standard'):
    if pd.isna(val) or val is None: return ''
    if isinstance(val, bool): return 'color: #00FF88; font-weight: bold;' if val else 'color: #888;'

    if not isinstance(val, (int, float)): 
        if mode == 'quality': return 'color: #00FF88; font-weight: bold;' if val == 'High' else ('color: #DAA520; font-weight: bold;' if val == 'Medium' else 'color: #FF0000; font-weight: bold;')
        return ''
    
    if mode == 'standard': return 'color: #008000; font-weight: bold;' if val >= 70 else ('color: #DAA520; font-weight: bold;' if val >= 40 else 'color: #FF0000; font-weight: bold;')
    elif mode == 'inverse': return 'color: #008000; font-weight: bold;' if val <= 30 else ('color: #DAA520; font-weight: bold;' if val <= 60 else 'color: #FF0000; font-weight: bold;')
    elif mode == 'return': return 'color: #008000; font-weight: bold;' if val > 0 else 'color: #FF0000; font-weight: bold;'
    elif mode == 'vol_expansion': return 'color: #008000; font-weight: bold;' if val >= 2.0 else ('color: #DAA520; font-weight: bold;' if val >= 1.2 else 'color: #FF0000; font-weight: bold;')
    elif mode == 'bo_success': return 'color: #008000; font-weight: bold;' if val >= 60 else ('color: #DAA520; font-weight: bold;' if val >= 40 else 'color: #FF0000; font-weight: bold;')
    return ''

# --- 8. MAIN UI ---
with st.sidebar:
    st.title("⚙️ Configuration")
    file_path = "NiftyTM.txt" 
    
    NIFTY50_TICKERS = ["RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "BHARTIARTL", "ITC", "SBIN", "LICI", "HINDUNILVR", "LT", "BAJFINANCE", "HCLTECH", "MARUTI", "SUNPHARMA", "ADANIENT", "KOTAKBANK", "TITAN", "ONGC", "TATAMOTORS", "NTPC", "AXISBANK", "ADANIPORTS", "ULTRACEMCO", "POWERGRID", "BAJAJFINSV", "M&M", "WIPRO", "COALINDIA", "TATASTEEL", "ASIANPAINT", "JSWSTEEL", "HDFCLIFE", "SBILIFE", "LTIM", "GRASIM", "TECHM", "BRITANNIA", "INDUSINDBK", "HINDALCO", "DIVISLAB", "EICHERMOT", "APOLLOHOSP", "TATACONSUM", "NESTLEIND", "DRREDDY", "BAJAJ-AUTO", "CIPLA", "HEROMOTOCO", "BPCL"]
    tickers = [line.strip() for line in open(file_path, "r").readlines() if line.strip()] if os.path.exists(file_path) else NIFTY50_TICKERS
    
    if not os.path.exists(file_path):
        st.sidebar.warning(f"'{file_path}' not found. Fallback to NIFTY50.")
        if uploaded := st.sidebar.file_uploader("Upload Ticker List", type=["txt"]):
            tickers = [line.strip() for line in uploaded.read().decode("utf-8").splitlines() if line.strip()]

    st.divider()
    preset = st.radio("Analysis Date:", ["Today", "Yesterday", "This Week", "Last Week", "Last Week till Date", "This Month", "This Year", "Custom"], index=4)
    today = date.today()
    if preset == "Today": s_d, e_d = today, today
    elif preset == "Yesterday": s_d, e_d = today-timedelta(1), today-timedelta(1)
    elif preset == "This Week": s_d, e_d = today-timedelta(today.weekday()), today
    elif preset == "Last Week": s_d, e_d = today-timedelta(today.weekday()+7), today-timedelta(today.weekday()+1)
    elif preset == "Last Week till Date": s_d, e_d = today-timedelta(today.weekday()+7), today 
    elif preset == "This Month": s_d, e_d = date(today.year, today.month, 1), today
    elif preset == "This Year": s_d, e_d = date(today.year, 1, 1), today
    else: 
        c1, c2 = st.columns(2)
        with c1: s_d = st.date_input("Start", value=today-timedelta(7))
        with c2: e_d = st.date_input("End", value=today)
    
    st.divider()
    run_btn = st.button("🧮 Run Vector Engine", type="primary", use_container_width=True)

st.markdown('<div class="main-title">Vector IQ</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Breadth. Breakouts. Regime.</div>', unsafe_allow_html=True)
st.markdown(f'<div class="date-banner">📅 Period: {s_d} — {e_d}</div>', unsafe_allow_html=True)

if 'results' not in st.session_state: st.session_state.results = None

if tickers and (run_btn or st.session_state.results is None):
    with st.spinner("Downloading data... this might take 30 seconds..."):
        bar, status = st.progress(0), st.empty()
        st.session_state.results = scan_stocks(tickers, s_d, e_d, bar, status)
        bar.empty(), status.empty()

if st.session_state.results:
    ath, full_scan, breakouts, breadth, stage2, cmbf = st.session_state.results
    
    if breadth:
        df_breadth = pd.DataFrame(breadth)
        last10 = df_breadth.tail(10)
        
        if not last10.get("% Above 200 DMA", pd.Series(dtype=float)).empty:
            c1, c2, c3 = st.columns(3)
            with c1: render_regime_tile("% Above 20 DMA", last10["% Above 20 DMA"].iloc[-1], last10["% Above 20 DMA"], 50, True, "%")
            with c2: render_regime_tile("% Above 200 DMA", last10["% Above 200 DMA"].iloc[-1], last10["% Above 200 DMA"], 50, True, "%")
            with c3: render_regime_tile("% Breaking < 20 DMA", last10["% Breaking < 20 DMA"].iloc[-1], last10["% Breaking < 20 DMA"], 4, False, "%")
            c4, c5, c6 = st.columns(3)
            with c4: render_regime_tile("AD Slope 20D", last10["AD Slope 20D"].iloc[-1], last10["AD Slope 20D"].fillna(0), 0, True, "%")
            with c5: render_regime_tile("10D BO Success", last10["Rolling BO Success 10D"].iloc[-1], last10["Rolling BO Success 10D"].fillna(0), 50, True, "%")
            with c6: render_regime_tile("Net New Highs", (last10["New Highs"] - last10["New Lows"]).iloc[-1], (last10["New Highs"] - last10["New Lows"]).fillna(0), 0, True, "")

        if not df_breadth.empty:
            latest = df_breadth.iloc[-1]
            if latest["% Above 20 DMA"] > 60 and latest["AD Slope 20D"] > 0 and latest["Rolling BO Success 10D"] > 50: label = "🟢 EXPANSION REGIME"
            elif latest["% Above 20 DMA"] < 45 and latest["AD Slope 20D"] < 0 and latest["Rolling BO Success 10D"] < 40: label = "🔴 CHOP / RISK-OFF"
            else: label = "🟡 TRANSITIONAL"
            st.markdown(f"<div class='metric-box'>{label}</div>", unsafe_allow_html=True)

    tab_stage2, tab_ath, tab_pop, tab_trend, tab_base, tab_cmbf, tab_breadth = st.tabs(["📈 Stage 2 Breakouts", "⚡ ATH Breakouts", "💥 Volume Poppers", "🔭 Trend Watch", "🧱 Base Builder", "📦 CMBF", "📊 Market Breadth"])

    # FIX: Batch copy blocks fixed, passing `language=None` prevents Streamlit Javascript crash.
    def copy_tv(data):
        if not data: return
        unique_tickers = list(dict.fromkeys([f"NSE:{x['Ticker']}" for x in data]))
        batches = [", ".join(unique_tickers[i:i+30]) for i in range(0, len(unique_tickers), 30)]
        st.markdown("### 📋 TradingView Watchlist")
        for i, b in enumerate(batches):
            st.markdown(f"**Batch {i+1} ({len(unique_tickers[i*30:(i*30)+30])} items)**")
            st.code(b, language=None)

    with tab_stage2:
        df_s2 = pd.DataFrame(stage2)
        if df_s2.empty: st.info("No Stage 2 breakouts found.")
        else:
            if len(df_s2) >= 10: st.warning("⚠️ Signal Clustering Detected — Check Market Regime")
            if sel := st.multiselect("Filter by Ticker", sorted(df_s2["Ticker"].unique()), key="s2_filter"): df_s2 = df_s2[df_s2["Ticker"].isin(sel)]
            st.markdown(f"<div class='metric-box'>{len(df_s2)} Entries | {df_s2['Ticker'].nunique()} Unique Stocks</div>", unsafe_allow_html=True)
            
            styled = df_s2[["Ticker", "Stage2 Date", "Signal Quality", "S2_Event_Price", "LTP", "S2_Return", "RS Rating", "S2_Persistence", "S2_Failure_Risk"]].style.format({"S2_Event_Price": "₹ {:.2f}", "LTP": "₹ {:.2f}", "S2_Return": "{:.2f}%"}).map(lambda v: apply_text_styling(v, 'return'), subset=["S2_Return"]).map(lambda v: apply_text_styling(v, 'quality'), subset=["Signal Quality"]).map(lambda v: apply_text_styling(v, 'standard'), subset=["S2_Persistence", "RS Rating"]).map(lambda v: apply_text_styling(v, 'inverse'), subset=["S2_Failure_Risk"])
            st.dataframe(styled, use_container_width=True, hide_index=True)
            copy_tv(df_s2.to_dict('records'))

    with tab_ath:
        df_ath = pd.DataFrame(ath)
        if df_ath.empty: st.info("No ATH breakouts found.")
        else:
            if sel := st.multiselect("Filter by Ticker", sorted(df_ath["Ticker"].unique()), key="ath_filter"): df_ath = df_ath[df_ath["Ticker"].isin(sel)]
            st.markdown(f"<div class='metric-box'>{len(df_ath)} Entries | {df_ath['Ticker'].nunique()} Unique Stocks</div>", unsafe_allow_html=True)
            
            styled = df_ath[["Ticker", "ATH Date", "ATH_Event_Price", "LTP", "ATH_Return", "RS Rating", "ATH_Persistence", "ATH_Failure_Risk"]].style.format({"ATH_Event_Price": "₹ {:.2f}", "LTP": "₹ {:.2f}", "ATH_Return": "{:.2f}%"}).map(lambda v: apply_text_styling(v, 'return'), subset=["ATH_Return"]).map(lambda v: apply_text_styling(v, 'inverse'), subset=["ATH_Failure_Risk"]).map(lambda v: apply_text_styling(v, 'standard'), subset=["ATH_Persistence", "RS Rating"])
            st.dataframe(styled, use_container_width=True, hide_index=True)
            copy_tv(df_ath.to_dict('records'))

    with tab_pop:
        df_pop = pd.DataFrame(breakouts)
        if df_pop.empty: st.info("No Volume Poppers found.")
        else:
            if sel := st.multiselect("Filter by Ticker", sorted(df_pop["Ticker"].unique()), key="pop_filter"): df_pop = df_pop[df_pop["Ticker"].isin(sel)]
            st.markdown(f"<div class='metric-box'>{len(df_pop)} Entries | {df_pop['Ticker'].nunique()} Unique Stocks</div>", unsafe_allow_html=True)
            
            styled = df_pop[["Ticker", "Pop Date", "Pop_Event_Price", "LTP", "Pop_Return", "RS Rating", "Pop_Vol_Expansion", "Pop_Failure_Risk", "Pop_Persistence"]].style.format({"Pop_Event_Price": "₹ {:.2f}", "LTP": "₹ {:.2f}", "Pop_Return": "{:.2f}%", "Pop_Vol_Expansion": "{:.2f}x"}).map(lambda v: apply_text_styling(v, 'return'), subset=["Pop_Return"]).map(lambda v: apply_text_styling(v, 'vol_expansion'), subset=["Pop_Vol_Expansion"]).map(lambda v: apply_text_styling(v, 'inverse'), subset=["Pop_Failure_Risk"]).map(lambda v: apply_text_styling(v, 'standard'), subset=["Pop_Persistence", "RS Rating"])
            st.dataframe(styled, use_container_width=True, hide_index=True)
            copy_tv(df_pop.to_dict('records'))

    with tab_trend:
        if full_scan:
            df = pd.DataFrame(full_scan)[pd.DataFrame(full_scan)['Rocket Score'] >= 50].sort_values(by="Rocket Score", ascending=False)
            st.markdown(f"<div class='metric-box'>Found {len(df)} Trend Watch Setups</div>", unsafe_allow_html=True)
            styled = df[["Ticker", "Price", "Rocket Score", "Trend Score", "RS %", "Tightness %", "Vol Dry Score", "Near Breakout", "Failure Risk", "Persistence"]].style.format({"Price": "₹ {:.2f}", "Rocket Score": "{:.2f}"}).map(lambda v: apply_text_styling(v, 'standard'), subset=["Rocket Score", "Trend Score", "RS %", "Tightness %", "Vol Dry Score", "Near Breakout", "Persistence"]).map(lambda v: apply_text_styling(v, 'inverse'), subset=["Failure Risk"])
            st.dataframe(styled, use_container_width=True, hide_index=True)
            copy_tv(df.to_dict('records'))
        else: st.info("No Trend Watch Setups found.")

    with tab_base:
        if full_scan:
            df_base = pd.DataFrame(full_scan)
            df_base = df_base[(df_base['RS %'] >= 60) & (df_base["Basic VCP"] == True) & ((df_base["Within 25% 52W High"] == True) | (df_base["Elite VCP"] == True))]
            st.markdown(f"<div class='metric-box'>Found {len(df_base)} Base Structures (RS >= 60)</div>", unsafe_allow_html=True)
            
            if not df_base.empty:
                df_base = df_base.sort_values(by=["Elite VCP", "Base Duration", "Dist to Breakout %", "RS %"], ascending=[False, False, True, False])
                styled = df_base[["Ticker", "Price", "Base Duration", "Within 25% 52W High", "Elite VCP", "VCP Range %", "Dist to Breakout %", "RS %", "Vol Expansion"]].style.format({"Price": "₹ {:.2f}", "VCP Range %": "{:.2f}%", "Dist to Breakout %": "{:.2f}%", "Vol Expansion": "{:.2f}x"}).map(lambda v: apply_text_styling(v, 'standard'), subset=["Elite VCP", "Within 25% 52W High", "RS %"]).map(lambda v: apply_text_styling(v, 'vol_expansion'), subset=["Vol Expansion"])
                st.dataframe(styled, use_container_width=True, hide_index=True)
                copy_tv(df_base.to_dict('records'))
            else: st.warning("No stocks found matching the criteria (RS>=60, Basic VCP + [Elite VCP OR Near 52WH]).")
        else: st.info("No data available.")

    with tab_cmbf:
        df_cmbf = pd.DataFrame(cmbf)
        if not df_cmbf.empty:
            df_cmbf = df_cmbf[df_cmbf["Grade"] != "Reject"]
            if sel := st.multiselect("Filter by Ticker", sorted(df_cmbf["Ticker"].unique()), key="cmbf_filter"): df_cmbf = df_cmbf[df_cmbf["Ticker"].isin(sel)]
            st.markdown(f"<div class='metric-box'>{len(df_cmbf)} Entries | {df_cmbf['Ticker'].nunique()} Unique Stocks</div>", unsafe_allow_html=True)
            df_cmbf = df_cmbf.sort_values(by=["CMBF Date", "Grade", "ATR %"], ascending=[False, True, True])
            styled = df_cmbf[["Ticker", "CMBF Date", "Price", "Grade", "Protocol", "Daily RSI", "Weekly RSI", "Box Range %", "ATR %"]].style.format({"Price": "₹ {:.2f}", "ATR %": "{:.2f}%", "Box Range %": "{:.2f}%"})
            st.dataframe(styled, use_container_width=True, hide_index=True)
            copy_tv(df_cmbf.to_dict('records'))
        else: st.info("No CMBF setups found.")

    with tab_breadth:
        st.markdown("### 📊 Market Regime & Internals")
        if breadth:
            df = pd.DataFrame(breadth)
            def color_breadth(val):
                if pd.isna(val): return ''
                return 'color: #008000; font-weight: bold;' if val > 70 else ('color: #DAA520; font-weight: bold;' if val > 40 else 'color: #FF0000; font-weight: bold;')
            
            styled = df[["Date", "New Highs", "New Lows", "Net New Highs", "AD Slope 20D", "AD Change 20D", "Rolling BO Success 10D", "% Above 20 DMA", "% Breaking < 20 DMA", "% Above 200 DMA"]].style.format({"AD Slope 20D": "{:.2f}%", "AD Change 20D": "{:.2f}", "Rolling BO Success 10D": "{:.1f}%", "% Above 20 DMA": "{:.2f}%", "% Above 200 DMA": "{:.2f}%", "% Breaking < 20 DMA": "{:.2f}%"}).map(color_breadth, subset=["% Above 20 DMA", "% Above 200 DMA"]).map(lambda v: apply_text_styling(v, 'bo_success'), subset=["Rolling BO Success 10D"]).map(lambda val: 'color: #008000; font-weight: bold;' if pd.notna(val) and val > 0 else ('color: #FF0000; font-weight: bold;' if pd.notna(val) else ''), subset=["AD Slope 20D", "Net New Highs", "AD Change 20D"])
            st.dataframe(styled, use_container_width=True, hide_index=True)
        else: st.info("No Breadth Data.")
