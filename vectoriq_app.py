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

# --- CSS FOR ROUNDED CORNERS & CLEAN UI ---
st.markdown("""
    <style>
    .main-title {font-size: 3em; font-weight: bold; color: #FF4B4B;}
    .sub-title {font-size: 1.2em; color: #555;}
    .date-banner {
        background-color: #000000; 
        color: #ffffff;
        padding: 10px; 
        border-radius: 5px; 
        border-left: 5px solid #FF4B4B;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .metric-box {
        padding: 10px;
        background-color: #c3e6cb; 
        color: #0f5132; 
        border-radius: 5px;
        margin-bottom: 10px;
        font-weight: bold;
        text-align: center;
        border: 1px solid #b1dfbb;
    }
    /* ROUNDED CORNERS FOR PLOTLY TILES */
    .stPlotlyChart {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    </style>
""", unsafe_allow_html=True)

# --- 2. DATA LOADING LAYER (Cached & Optimized to 2y) ---
@st.cache_data(show_spinner=False, ttl=3600)
def download_data(tickers):
    if not tickers:
        return pd.DataFrame()
    
    fixed_tickers = [t if t.endswith('.NS') or t == "^NSEI" else f"{t}.NS" for t in tickers]
    download_list = list(set(fixed_tickers + ["^NSEI"]))
    
    try:
        data = yf.download(
            download_list,
            period="2y", # 🔥 OPTIMIZED: Reduced from 5y to 2y to speed up download and processing
            group_by='ticker',
            threads=True,
            progress=False
        )
        return data
    except Exception as e:
        st.error(f"Download API failed: {e}")
        return pd.DataFrame()

# --- 3. HELPER: REGIME TILE RENDERER (ROUNDED, 5D, NON-OBSTRUCTING) ---
def render_regime_tile(title, value, series, threshold, positive=True, suffix=""):
    if positive:
        regime_color = "#00FF88" if value >= threshold else "#FF4B4B"
    else:
        regime_color = "#00FF88" if value <= threshold else "#FF4B4B"

    series_5d = series.tail(5)
    dates_5d = series_5d.index.strftime('%Y-%m-%d').tolist() if hasattr(series_5d.index, 'strftime') else list(range(len(series_5d)))
    
    prev_val = series.iloc[-2] if len(series) >= 2 else value

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
        delta={
            'reference': prev_val, 
            'relative': False, 
            'position': "right",
            'valueformat': '.2f',
            'font': {'size': 16}
        },
        title={"text": title.upper(), "font": {"size": 13, "color": "gray", "family": "Arial"}},
        align="left"
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=dates_5d,
        y=series_5d,
        mode='lines+markers',
        line=dict(width=3, color=regime_color),
        marker=dict(size=5, color=regime_color),
        fill='tozeroy',
        fillcolor=f"rgba{tuple(int(regime_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (0.1,)}",
        hovertemplate='<b>%{x}</b><br>Val: %{y:.2f}<extra></extra>' 
    ), row=2, col=1)

    fig.update_layout(
        height=125, 
        margin=dict(l=20, r=20, t=25, b=15),
        template="plotly_dark",
        paper_bgcolor='#111827', 
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(visible=False, fixedrange=True),
        yaxis=dict(visible=False, fixedrange=True),
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

# --- 4. METRIC ENGINE (Updated with Vectorized Base Duration) ---
def calculate_advanced_metrics(df, bench_series):
    if df.empty or len(df) < 260: return None

    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume']
    
    sma50 = close.rolling(50).mean()
    sma150 = close.rolling(150).mean()
    sma200 = close.rolling(200).mean()
    c = close.iloc[-1]
    
    high_52w = high.rolling(252).max()
    low_52w = low.rolling(252).min()
    h52 = high_52w.iloc[-1]
    dist_52w_high_pct = ((h52 - c) / h52) * 100 if h52 > 0 else 100

    s50, s150, s200 = sma50.iloc[-1], sma150.iloc[-1], sma200.iloc[-1]
    d50 = (c - s50) / s50
    d150 = (c - s150) / s150
    d200 = (c - s200) / s200
    ma_dist_score = min(max((d50 + d150 + d200) * 200, 0), 50)
    
    spread = (s50 - s200) / s200 if s200 > 0 else 0
    alignment_score = min(max(spread * 300, 0), 30)
    
    s200_prev_20 = sma200.iloc[-20]
    slope = (s200 - s200_prev_20) / s200_prev_20 if s200_prev_20 > 0 else 0
    slope_score = min(max(slope * 500, 0), 20)
    
    trend_score = ma_dist_score + alignment_score + slope_score

    bench_series = bench_series.ffill()
    rs_line = close / bench_series
    rs_curr = rs_line.iloc[-1]
    rs_prev_20 = rs_line.iloc[-20]
    rs_mom = (rs_curr - rs_prev_20) / rs_prev_20 if rs_prev_20 > 0 else 0
    rs_mom_norm = min(max(rs_mom * 10, 0), 1) * 100 
    rs_raw = rs_curr

    def get_range(window):
        h = high.tail(window).max()
        l = low.tail(window).min()
        return (h - l) / h if h > 0 else 1

    r20, r60 = get_range(20), get_range(60)
    compression_ratio = r20 / r60 if r60 > 0 else 1
    tight_score = min(max((1 - compression_ratio) * 100, 0), 100)

    v = volume.iloc[-1]
    v_5d = volume.tail(5).mean()
    v_50d = volume.rolling(50).mean().iloc[-1]
    
    if v_50d > 0:
        dry_ratio = v_5d / v_50d
        dry_score = min(max((1 - dry_ratio) * 100, 0), 50)
        exp_ratio = v / v_50d
        exp_score = min(exp_ratio * 10, 50)
        vol_expansion = exp_ratio 
    else:
        dry_score, exp_score, vol_expansion = 0, 0, 0
        
    vol_score = dry_score + exp_score
    readiness_score = min(max((1 - ((h52 - c) / h52 / 0.10)) * 100, 0), 100) if h52 > 0 else 0

    tight_len = 10
    max_rng_pct = 12
    
    rolling_high_10 = high.rolling(tight_len).max().iloc[-1]
    rolling_low_10 = low.rolling(tight_len).min().iloc[-1]
    
    vcp_range_pct = None
    basic_vcp = False
    
    if rolling_high_10 is not None and rolling_high_10 > 0:
        vcp_range_pct = ((rolling_high_10 - rolling_low_10) / rolling_high_10) * 100
        is_tight = vcp_range_pct < max_rng_pct
    else:
        is_tight = False

    vol_dry_check = False
    if v_50d is not None and v_50d > 0:
        vol_dry_check = v_5d < v_50d
    
    trend_ok = (c > s150) and (s50 > s200)
    basic_vcp = is_tight and vol_dry_check and trend_ok

    base_len = 45
    num_contract = 3
    segment = max(1, base_len // num_contract)
    elite_vcp = False
    
    if len(df) >= base_len:
        try:
            h1, l1 = high.iloc[-segment*3:-segment*2].max(), low.iloc[-segment*3:-segment*2].min()
            h2, l2 = high.iloc[-segment*2:-segment].max(), low.iloc[-segment*2:-segment].min()
            h3, l3 = high.iloc[-segment:].max(), low.iloc[-segment:].min()
            
            def depth(h, l): return ((h - l) / h) * 100 if h > 0 else 0
                
            d1, d2, d3 = depth(h1, l1), depth(h2, l2), depth(h3, l3)
            
            elite_vcp = (
                basic_vcp and
                (d1 > d2) and       
                (d2 > d3) and       
                (d3 < 12)           
            )
        except:
            elite_vcp = False

    base_high = high.iloc[-base_len:].max()
    dist_to_breakout = ((base_high - c) / base_high) * 100 if base_high > 0 else 100

    # 🔥 OPTIMIZED: Vectorized Base Duration logic (Replaces slow for-loop)
    base_low = low.iloc[-base_len:].min()
    base_high = high.iloc[-base_len:].max()
    max_duration = base_len * 2 

    in_base = (close >= base_low) & (close <= base_high)
    base_duration = int(in_base.astype(int).iloc[::-1].cummin().sum())
    base_duration = min(base_duration, max_duration)

    sma20 = close.rolling(20).mean().iloc[-1]
    failure_score = 0
    if vol_expansion < 1.3: failure_score += 30
    h_day, l_day = high.iloc[-1], low.iloc[-1]
    if c < sma20: failure_score += 20
    if (h_day - l_day) > 0 and ((c - l_day)/(h_day - l_day)) < 0.5: failure_score += 20
    if sma20 > 0 and ((c - sma20)/sma20) > 0.15: failure_score += 20
    failure_risk = min(failure_score, 100)

    persist_score = 0
    up_days = (close.diff() > 0).tail(20).sum()
    persist_score += (up_days / 20) * 40
    if rs_curr > rs_prev_20: persist_score += 20
    higher_highs = (high.diff() > 0).tail(20).sum()
    persist_score += min((higher_highs / 20) * 30, 30)
    if r20 < 0.10: persist_score += 10
    persistence = min(persist_score, 100)

    breakout_20d = c > high.rolling(20).max().shift(1).iloc[-1]
    stage2_trend = (c > s200) and (s50 > s200) and (s200 > sma200.iloc[-20])
    breakout_50d = c > high.rolling(50).max().shift(1).iloc[-1]
    breakout_trigger = breakout_20d or breakout_50d
    extension = (c - s200) / s200 if s200 > 0 else 0
    
    not_extended = (extension > 0.02) and (extension < 0.20)
    vol_confirm = vol_expansion >= 1.3
    stage2_candidate = stage2_trend and breakout_trigger and not_extended and vol_confirm

    return {
        "Ticker": "",
        "Price": c,
        "Trend Score": int(trend_score),
        "RS Raw": rs_raw,
        "RS Mom Score": rs_mom_norm,
        "Tightness %": int(tight_score), 
        "Vol Dry Score": int(vol_score),
        "Near Breakout": int(readiness_score),
        "Failure Risk": int(failure_risk),
        "Persistence": int(persistence),
        "Vol Expansion": round(vol_expansion, 2),
        "Breakout 20D": breakout_20d,
        "Stage2_Candidate": stage2_candidate,
        "Basic VCP": basic_vcp,
        "Elite VCP": elite_vcp,
        "VCP Range %": round(vcp_range_pct, 2) if vcp_range_pct is not None else 0,
        "Dist to Breakout %": round(dist_to_breakout, 2),
        "Within 25% 52W High": dist_52w_high_pct <= 25,
        "Base Duration": base_duration 
    }

# --- 5. BREADTH ENGINE ---
def calculate_market_breadth(raw_data, start_date, end_date):
    if isinstance(raw_data.columns, pd.MultiIndex):
        stock_data = raw_data.drop(columns=["^NSEI"], level=0, errors='ignore')
    else: return []

    try:
        closes = stock_data.xs('Close', level=1, axis=1)
        highs = stock_data.xs('High', level=1, axis=1)
        lows = stock_data.xs('Low', level=1, axis=1)
        
        universe_size = closes.shape[1]

        sma20 = closes.rolling(20).mean()
        sma200 = closes.rolling(200).mean()
        above_20dma = (closes > sma20)
        above_200dma = (closes > sma200)
        
        roll_high_252 = highs.rolling(252).max()
        roll_low_252 = lows.rolling(252).min() 
        is_new_high = (highs >= roll_high_252)
        is_new_low = (lows <= roll_low_252)
        
        daily_diff = closes.diff()
        advances = (daily_diff > 0).sum(axis=1)
        declines = (daily_diff < 0).sum(axis=1)
        net_ad = advances - declines
        ad_line = net_ad.cumsum() 
        
        pivot_20 = highs.rolling(20).max().shift(1)
        is_breakout = (closes > pivot_20)
        
        breakout_10d_ago = is_breakout.shift(10)
        ret_10d = closes.pct_change(10)
        successful_breakout = breakout_10d_ago & (ret_10d > 0)
        
        daily_bo_attempts = breakout_10d_ago.sum(axis=1)
        daily_bo_successes = successful_breakout.sum(axis=1)
        
        rolling_attempts = daily_bo_attempts.rolling(10, min_periods=3).sum()
        rolling_successes = daily_bo_successes.rolling(10, min_periods=3).sum()
        
        rolling_bo_success_series = np.where(
            rolling_attempts > 0,
            (rolling_successes / rolling_attempts) * 100,
            np.nan
        )
        rolling_bo_success_series = pd.Series(
            rolling_bo_success_series,
            index=closes.index
        )

        break_below_20dma = (closes < sma20) & (closes.shift(1) > sma20.shift(1))
        
        mask = (closes.index.date >= start_date) & (closes.index.date <= end_date)
        valid_dates = closes.index[mask]
        
        breadth_records = []
        
        for d in valid_dates:
            total_valid_stocks = closes.loc[d].count()
            if total_valid_stocks == 0: continue
            
            idx_loc = closes.index.get_loc(d)

            slope_val = np.nan
            if idx_loc >= 20:
                y = ad_line.iloc[idx_loc-19 : idx_loc+1].values
                x = np.arange(len(y))
                if len(y) == 20:
                    slope = np.polyfit(x, y, 1)[0]
                    denom = abs(ad_line.iloc[idx_loc-20])
                    slope_val = (slope / denom) * 100 if denom > 0 else 0

            if idx_loc >= 20:
                ad_change_20d = ad_line.iloc[idx_loc] - ad_line.iloc[idx_loc - 20]
            else:
                ad_change_20d = np.nan

            nh = is_new_high.loc[d].sum()
            nl = is_new_low.loc[d].sum()
            
            bo_val = rolling_bo_success_series.loc[d]
            bo_val_safe = round(bo_val, 2) if not np.isnan(bo_val) else np.nan

            breadth_records.append({
                "Date": d.date(),
                "% Above 20 DMA": round((above_20dma.loc[d].sum() / total_valid_stocks) * 100, 2),
                "% Above 200 DMA": round((above_200dma.loc[d].sum() / total_valid_stocks) * 100, 2),
                "New Highs": int(nh), "New Lows": int(nl), "Net New Highs": int(nh - nl),
                "AD Line": int(ad_line.loc[d]), "AD Slope 20D": round(slope_val, 2) if not np.isnan(slope_val) else np.nan,
                "AD Change 20D": round(ad_change_20d, 2) if not np.isnan(ad_change_20d) else np.nan,
                "Rolling BO Success 10D": bo_val_safe,
                "% Breaking < 20 DMA": round((break_below_20dma.loc[d].sum() / total_valid_stocks) * 100, 1)
            })
            
        return breadth_records
    except Exception as e:
        return []

# --- 6. SCANNER ORCHESTRATOR (Highly Vectorized) ---
def scan_stocks(tickers, start_date, end_date, progress_bar, status_text):
    status_text.text("🔌 Downloading Data...")
    raw_data = download_data(tickers)
    
    if raw_data.empty:
        st.error("⚠️ Data download failed. Please check your internet or ticker list.")
        return [], [], [], [], [], [] 
    
    status_text.text("📊 Calculating Breadth...")
    breadth_data = calculate_market_breadth(raw_data, start_date, end_date)
    
    try:
        if isinstance(raw_data.columns, pd.MultiIndex):
            if "^NSEI" in raw_data.columns.levels[0]:
                bench_data = raw_data["^NSEI"]['Close']
            else: bench_data = pd.Series()
        else: bench_data = pd.Series()
    except: bench_data = pd.Series()

    ath_results, pop_results, stage2_results = [], [], []
    current_state_results, cmbf_results = [], []
    
    if isinstance(raw_data.columns, pd.MultiIndex):
        downloaded_tickers = list(set([col[0] for col in raw_data.columns]))
    else: downloaded_tickers = [] 
        
    if "^NSEI" in downloaded_tickers: downloaded_tickers.remove("^NSEI")
    
    if not downloaded_tickers:
        st.warning("⚠️ No stock data found. Check if tickers have '.NS' suffix.")
        return [], [], [], breadth_data, [], []

    total = len(downloaded_tickers)
    
    for idx, ticker in enumerate(downloaded_tickers):
        progress_bar.progress((idx + 1) / total)
        status_text.text(f"Analyzing {ticker}...")
        
        try:
            df = raw_data[ticker].copy()
            if df.empty or len(df) < 260: continue
            
            # LIQUIDITY FILTER (Dollar Volume)
            df['Dollar_Vol'] = df['Close'] * df['Volume']
            avg_dollar_vol = df['Dollar_Vol'].rolling(50).mean().iloc[-1]
            if avg_dollar_vol < 1_00_000_000: continue

            # ==========================================
            # 🔥 PRE-CALCULATE ALL INDICATORS FOR ENTIRE DF
            # ==========================================
            df['SMA20'] = df['Close'].rolling(20).mean()
            df['SMA50'] = df['Close'].rolling(50).mean()
            df['SMA150'] = df['Close'].rolling(150).mean()
            df['SMA200'] = df['Close'].rolling(200).mean()
            df['High_20'] = df['High'].rolling(20).max().shift(1)
            df['High_50'] = df['High'].rolling(50).max().shift(1)
            df['Vol_MA50'] = df['Volume'].rolling(50).mean()
            df['Prev_ATH'] = df['High'].expanding().max().shift(1)
            df['SMA200_Rising'] = df['SMA200'] > df['SMA200'].shift(20)

            # Benchmark alignment & Base Metrics
            bench_aligned = bench_data.reindex(df.index).ffill()
            df['RS_Line'] = df['Close'] / bench_aligned
            df['RS_Mom_Ratio'] = df['RS_Line'] / df['RS_Line'].shift(20)
            df['RS_Mom_Pct'] = (df['RS_Line'] - df['RS_Line'].shift(20)) / df['RS_Line'].shift(20)
            df['RS_Score'] = (df['RS_Mom_Pct'] * 10).clip(0, 1) * 100

            # Volume & Risk Metrics
            df['Vol_Exp'] = df['Volume'] / df['Vol_MA50']
            hl_diff = df['High'] - df['Low']
            close_loc = (df['Close'] - df['Low']) / hl_diff.replace(0, np.nan)
            
            df['Failure_Risk'] = np.where(df['Vol_Exp'] < 1.2, 30, 0) + \
                                 np.where((hl_diff > 0) & (close_loc < 0.5), 20, 0) + \
                                 np.where((df['SMA20'] > 0) & ((df['Close'] - df['SMA20'])/df['SMA20'] > 0.15), 20, 0)
            df['Failure_Risk'] = df['Failure_Risk'].clip(upper=100)

            df['Up_Days_20'] = (df['Close'].diff() > 0).rolling(20).sum()
            df['Higher_Highs_20'] = (df['High'].diff() > 0).rolling(20).sum()
            df['Persist'] = (df['Up_Days_20'] / 20) * 40 + \
                            ((df['Higher_Highs_20'] / 20) * 30).clip(upper=30) + \
                            np.where(df['Vol_Exp'] > 1.2, 10, 0)
            df['Persist'] = df['Persist'].clip(upper=100)

            # CMBF Pre-Calculations
            df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
            df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
            df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
            df['High_52'] = df['High'].rolling(252).max()
            df['Near_High'] = df['Close'] >= 0.95 * df['High_52']
            
            df['RS_20'] = df['RS_Line'].pct_change(20)
            df['RS_50'] = df['RS_Line'].pct_change(50)
            df['RS_Pass'] = (df['RS_20'] > 0) & (df['RS_50'] > 0)
            df['EMA_Stack'] = (df['Close'] > df['EMA20']) & (df['EMA20'] > df['EMA50']) & (df['EMA50'] > df['EMA200'])
            
            delta = df['Close'].diff()
            gain = delta.clip(lower=0).rolling(14).mean()
            loss = -delta.clip(upper=0).rolling(14).mean()
            df['Daily_RSI'] = 100 - (100 / (1 + (gain / loss)))

            weekly = df['Close'].resample("W").last()
            w_gain = weekly.diff().clip(lower=0).rolling(14).mean()
            w_loss = -weekly.diff().clip(upper=0).rolling(14).mean()
            w_rsi = 100 - (100 / (1 + (w_gain / w_loss)))
            df['Weekly_RSI'] = w_rsi.reindex(df.index, method='ffill').fillna(50)

            df['Box_HH'] = df['High'].rolling(20).max()
            df['Box_LL'] = df['Low'].rolling(20).min()
            df['Box_Range_Pct'] = (df['Box_HH'] - df['Box_LL']) / df['Box_LL'].replace(0, np.nan)
            df['Box_Pass'] = df['Box_Range_Pct'] < 0.08

            df['Up_Vol'] = (df['Volume'] * (df['Close'] > df['Close'].shift(1))).rolling(20).sum()
            df['Down_Vol'] = (df['Volume'] * (df['Close'] < df['Close'].shift(1))).rolling(20).sum()
            df['Vol_Pass'] = df['Up_Vol'] > df['Down_Vol']

            tr = np.maximum(df['High'] - df['Low'], np.maximum(abs(df['High'] - df['Close'].shift()), abs(df['Low'] - df['Close'].shift())))
            df['ATR'] = tr.rolling(14).mean()
            df['ATR_Pct'] = df['ATR'] / df['Close'].replace(0, np.nan)

            # --- DATE SLICE ---
            mask_range = (df.index.date >= start_date) & (df.index.date <= end_date)
            range_df = df.loc[mask_range]
            
            if range_df.empty: continue

            # --- 1. ATH CHECK ---
            ath_break = range_df[range_df['Close'] > range_df['Prev_ATH']]
            for event_date, row in ath_break.iterrows():
                record = {
                    "Ticker": ticker.replace(".NS", ""),
                    "ATH Date": event_date.date(),
                    "ATH_Event_Price": row["Close"],
                    "LTP": df.iloc[-1]['Close'],
                    "ATH_Vol_Expansion": round(row["Vol_Exp"], 2),
                    "ATH_Failure_Risk": int(row["Failure_Risk"]),
                    "ATH_Persistence": int(row["Persist"]),
                    "RS Rating": int(row["RS_Score"])
                }
                record["ATH_Return"] = ((record["LTP"] - record["ATH_Event_Price"]) / record["ATH_Event_Price"]) * 100
                ath_results.append(record)

            # --- 2. VOL POP CHECK ---
            pop_mask = (range_df['Close'] > range_df['High_20']) & (range_df['Volume'] > 1.2 * range_df['Vol_MA50'])
            pop_days = range_df[pop_mask]
            for event_date, row in pop_days.iterrows():
                record = {
                    "Ticker": ticker.replace(".NS", ""),
                    "Pop Date": event_date.date(),
                    "Pop_Event_Price": row["Close"],
                    "LTP": df.iloc[-1]['Close'],
                    "Pop_Vol_Expansion": round(row["Vol_Exp"], 2),
                    "Pop_Failure_Risk": int(row["Failure_Risk"]),
                    "Pop_Persistence": int(row["Persist"]),
                    "RS Rating": int(row["RS_Score"])
                }
                record["Pop_Return"] = ((record["LTP"] - record["Pop_Event_Price"]) / record["Pop_Event_Price"]) * 100
                pop_results.append(record)

            # --- 3. STAGE 2 CHECK ---
            s2_dist = (range_df['Close'] - range_df['SMA200']) / range_df['SMA200']
            s2_ext = (s2_dist > 0.02) & (s2_dist < 0.20) 
            s2_trend = (range_df['Close'] > range_df['SMA200']) & (range_df['SMA50'] > range_df['SMA200']) & (range_df['SMA200_Rising']) 
            s2_breakout = (range_df['Close'] > range_df['High_50']) | (range_df['Close'] > range_df['High_20'])
            s2_vol = range_df['Volume'] > 1.3 * range_df['Vol_MA50']
            
            s2_days = range_df[s2_trend & s2_breakout & s2_vol & s2_ext]

            for event_date, row in s2_days.iterrows():
                if (row["RS_Score"] >= 60) and (row["Persist"] >= 50):
                    quality = "Low"
                    if row["RS_Score"] >= 75 and row["Persist"] >= 65 and row["Failure_Risk"] < 30:
                        quality = "High"
                    elif row["RS_Score"] >= 60:
                        quality = "Medium"

                    record = {
                        "Ticker": ticker.replace(".NS", ""),
                        "Stage2 Date": event_date.date(),
                        "S2_Event_Price": row["Close"],
                        "LTP": df.iloc[-1]['Close'],
                        "S2_Vol_Expansion": round(row["Vol_Exp"], 2),
                        "S2_Failure_Risk": int(row["Failure_Risk"]),
                        "S2_Persistence": int(row["Persist"]),
                        "Event RS Mom": round(row["RS_Mom_Ratio"], 4),
                        "RS Rating": int(row["RS_Score"]),
                        "Signal Quality": quality
                    }
                    record["S2_Return"] = ((record["LTP"] - record["S2_Event_Price"]) / record["S2_Event_Price"]) * 100
                    stage2_results.append(record)

            # --- 4. CURRENT STATE (ROCKETS + BASE BUILDER) ---
            df_end = df.loc[:end_date]
            bench_aligned_end = bench_data.reindex(df_end.index).ffill()
            curr_metrics = calculate_advanced_metrics(df_end, bench_aligned_end)
            if curr_metrics:
                curr_metrics["Ticker"] = ticker.replace(".NS", "")
                current_state_results.append(curr_metrics)

            # --- 5. CMBF FILTER (Vectorized) ---
            cmbf_mask = range_df['Near_High'] & range_df['RS_Pass'] & range_df['EMA_Stack'] & range_df['Box_Pass']
            cmbf_days = range_df[cmbf_mask]
            
            for event_date, row in cmbf_days.iterrows():
                rsi_pass = (row['Daily_RSI'] > 60) and (row['Weekly_RSI'] > 60)
                vol_pass = row['Vol_Pass']
                score = int(rsi_pass) + int(vol_pass)
                
                if score == 2: grade = "A"
                elif score == 1: grade = "B"
                else: grade = "Reject"
                
                protocol = "Kinetic" if row['ATR_Pct'] < 0.04 else "Affordable"
                
                cmbf_results.append({
                    "Ticker": ticker.replace(".NS", ""),
                    "CMBF Date": event_date.date(),
                    "Price": row['Close'],
                    "Grade": grade,
                    "Protocol": protocol,
                    "Daily RSI": round(row['Daily_RSI'], 1),
                    "Weekly RSI": round(row['Weekly_RSI'], 1),
                    "Box Range %": round(row['Box_Range_Pct'] * 100, 2),
                    "ATR %": round(row['ATR_Pct'] * 100, 2)
                })

        except Exception: 
            continue

    if not any([ath_results, pop_results, stage2_results, current_state_results, cmbf_results]):
        return [], [], [], breadth_data, [], []
    
    # --- POST-PROCESSING ---
    df_current = pd.DataFrame(current_state_results)
    full_scan_list = []
    
    if not df_current.empty:
        df_current['RS Percentile'] = df_current['RS Raw'].rank(pct=True) * 100
        df_current['RS %'] = ((df_current['RS Percentile'] * 0.7) + (df_current['RS Mom Score'] * 0.3)).fillna(0).astype(int)
        
        df_current['Rocket Score'] = (
            0.30 * df_current['Trend Score'] + 0.25 * df_current['RS %'] +
            0.20 * df_current['Tightness %'] + 0.15 * df_current['Vol Dry Score'] +
            0.10 * df_current['Near Breakout']
        ).round(2)
        
        full_scan_list = df_current.to_dict('records')

    stage2_list = []
    if stage2_results:
        df_s2 = pd.DataFrame(stage2_results)
        stage2_list = df_s2.to_dict('records')

    return ath_results, full_scan_list, pop_results, breadth_data, stage2_list, cmbf_results

# --- 7. UTILS: STYLING ---
def apply_text_styling(val, mode='standard'):
    if isinstance(val, bool):
        return 'color: #00FF88; font-weight: bold;' if val else 'color: #333;'

    if not isinstance(val, (int, float)): 
        if mode == 'quality':
            if val == 'High': return 'color: #00FF88; font-weight: bold;'
            if val == 'Medium': return 'color: #DAA520; font-weight: bold;'
            if val == 'Low': return 'color: #FF0000; font-weight: bold;'
        return ''
    
    green = 'color: #008000; font-weight: bold;' 
    amber = 'color: #DAA520; font-weight: bold;'
    red = 'color: #FF0000; font-weight: bold;' 
    
    if mode == 'standard':
        if val >= 70: return green
        elif val >= 40: return amber
        else: return red
    elif mode == 'inverse': 
        if val <= 30: return green
        elif val <= 60: return amber
        else: return red
    elif mode == 'return': 
        if val > 0: return green
        else: return red
    elif mode == 'vol_expansion':
        if val >= 2.0: return green
        elif val >= 1.2: return amber
        else: return red
    elif mode == 'bo_success':
        if val >= 60: return green
        elif val >= 40: return amber
        else: return red
    return ''

# --- 8. MAIN UI ---
with st.sidebar:
    st.title("⚙️ Configuration")
    file_path = "NiftyTM.txt" 
    tickers = []
    
    NIFTY50_TICKERS = [
        "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "BHARTIARTL", "ITC", "SBIN",
        "LICI", "HINDUNILVR", "LT", "BAJFINANCE", "HCLTECH", "MARUTI", "SUNPHARMA",
        "ADANIENT", "KOTAKBANK", "TITAN", "ONGC", "TATAMOTORS", "NTPC", "AXISBANK",
        "ADANIPORTS", "ULTRACEMCO", "POWERGRID", "BAJAJFINSV", "M&M", "WIPRO",
        "COALINDIA", "TATASTEEL", "ASIANPAINT", "JSWSTEEL", "HDFCLIFE", "SBILIFE",
        "LTIM", "GRASIM", "TECHM", "BRITANNIA", "INDUSINDBK", "HINDALCO", "DIVISLAB",
        "EICHERMOT", "APOLLOHOSP", "TATACONSUM", "NESTLEIND", "DRREDDY", "BAJAJ-AUTO",
        "CIPLA", "HEROMOTOCO", "BPCL"
    ]

    if os.path.exists(file_path):
        with open(file_path, "r") as f: tickers = [line.strip() for line in f.readlines() if line.strip()]
    else:
        tickers = NIFTY50_TICKERS 
    
    if not tickers:
        st.sidebar.warning(f"'{file_path}' not found and fallback failed.")
        uploaded = st.sidebar.file_uploader("Upload Ticker List", type=["txt"])
        if uploaded: tickers = [line.strip() for line in uploaded.read().decode("utf-8").splitlines() if line.strip()]

    st.divider()
    
    preset = st.radio(
        "Analysis Date:", 
        ["Today", "Yesterday", "This Week", "Last Week", "Last Week till Date", "This Month", "This Year", "Custom"],
        index=4
    )
    
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

utc_now = datetime.now(timezone.utc)
ist_offset = timedelta(hours=5, minutes=30)
ist_time = utc_now + ist_offset
last_refreshed = ist_time.strftime("%Y-%m-%d | %H:%M:%S")

col_header, col_refresh = st.columns([3, 1])

with col_header:
    st.markdown('<div class="main-title">Vector IQ</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Breadth. Breakouts. Regime.</div>', unsafe_allow_html=True)

with col_refresh:
    st.markdown(f"""
    <div style="text-align: right; color: gray; font-size: 0.9em; padding-top: 20px;">
        Last Refreshed:<br><b>{last_refreshed}</b>
    </div>
    """, unsafe_allow_html=True)

st.markdown(f'<div class="date-banner">📅 Period: {s_d} — {e_d}</div>', unsafe_allow_html=True)

if 'results' not in st.session_state:
    st.session_state.results = None

if tickers:
    should_run = run_btn or (st.session_state.results is None)
    
    if should_run:
        msg = "Downloading data... this might take 30 seconds..."
        with st.spinner(msg):
            start_t = time.time()
            bar = st.progress(0)
            status = st.empty()
            st.session_state.results = scan_stocks(tickers, s_d, e_d, bar, status)
            bar.empty()
            status.empty()
            end_t = time.time()

if st.session_state.results:
    ath, full_scan, breakouts, breadth, stage2, cmbf = st.session_state.results
    
    if breadth:
        df_breadth = pd.DataFrame(breadth)
        df_breadth["Net New Lows"] = df_breadth["New Lows"] - df_breadth["New Highs"]
        last10 = df_breadth.tail(10)
        
        pct20_series = last10.get("% Above 20 DMA", pd.Series(dtype=float))
        pct200_series = last10.get("% Above 200 DMA", pd.Series(dtype=float))
        adslope_series = last10.get("AD Slope 20D", pd.Series(dtype=float))
        bo_series = last10.get("Rolling BO Success 10D", pd.Series(dtype=float))
        break20_series = last10.get("% Breaking < 20 DMA", pd.Series(dtype=float))
        net_highs_series = last10.get("Net New Highs", pd.Series(dtype=float))

        if not pct200_series.empty:
            c1, c2, c3 = st.columns(3)
            with c1: render_regime_tile("% Above 20 DMA", pct20_series.iloc[-1], pct20_series, 50, True, "%")
            with c2: render_regime_tile("% Above 200 DMA", pct200_series.iloc[-1], pct200_series, 50, True, "%")
            with c3: render_regime_tile("% Breaking < 20 DMA", break20_series.iloc[-1], break20_series, 4, False, "%")
            c4, c5, c6 = st.columns(3)
            with c4: render_regime_tile("AD Slope 20D", adslope_series.iloc[-1], adslope_series.fillna(0), 0, True, "%")
            with c5: render_regime_tile("10D BO Success", bo_series.iloc[-1], bo_series.fillna(0), 50, True, "%")
            with c6: render_regime_tile("Net New Highs", net_highs_series.iloc[-1], net_highs_series.fillna(0), 0, True, "")

        if not df_breadth.empty:
            latest = df_breadth.iloc[-1]

            cond_expansion = (
                latest["% Above 20 DMA"] > 60 and
                latest["AD Slope 20D"] > 0 and
                latest["Rolling BO Success 10D"] > 50
            )

            cond_chop = (
                latest["% Above 20 DMA"] < 45 and
                latest["AD Slope 20D"] < 0 and
                latest["Rolling BO Success 10D"] < 40
            )

            if cond_expansion:
                regime_label = "🟢 EXPANSION REGIME"
            elif cond_chop:
                regime_label = "🔴 CHOP / RISK-OFF"
            else:
                regime_label = "🟡 TRANSITIONAL"

            st.markdown(f"<div class='metric-box'>{regime_label}</div>", unsafe_allow_html=True)

    tab_stage2, tab_ath, tab_pop, tab_trend, tab_base, tab_cmbf, tab_breadth = st.tabs([
        "📈 Stage 2 Breakouts", 
        "⚡ ATH Breakouts", 
        "💥 Volume Poppers", 
        "🔭 Trend Watch",
        "🧱 Base Builder",
        "📦 CMBF",
        "📊 Market Breadth"
    ])

    def copy_tv(data):
        if not data: return
        unique_tickers = list(dict.fromkeys([f"NSE:{x['Ticker']}" for x in data]))
        batches = [", ".join(unique_tickers[i:i+30]) for i in range(0, len(unique_tickers), 30)]
        st.markdown("### 📋 TradingView Watchlist")
        for b in batches: st.code(b, language="text")

    with tab_stage2:
        df_s2 = pd.DataFrame(stage2)
        if df_s2.empty:
            st.info("No Stage 2 breakouts found.")
        else:
            if len(df_s2) >= 10:
                st.warning("⚠️ Signal Clustering Detected — Check Market Regime")

            all_tickers = sorted(df_s2["Ticker"].unique())
            selected_tickers = st.multiselect("Filter by Ticker", all_tickers, key="s2_filter")
            
            if selected_tickers:
                df_s2 = df_s2[df_s2["Ticker"].isin(selected_tickers)]

            entries_count = len(df_s2)
            unique_count = df_s2["Ticker"].nunique()
            st.markdown(f"<div class='metric-box'>{entries_count} Entries | {unique_count} Unique Stocks</div>", unsafe_allow_html=True)
            
            cols = ["Ticker", "Stage2 Date", "Signal Quality", "S2_Event_Price", "LTP", "S2_Return", "RS Rating", "S2_Persistence", "S2_Failure_Risk"]
            styled = df_s2[cols].style.format({
                "S2_Event_Price": "₹ {:.2f}", "LTP": "₹ {:.2f}", "S2_Return": "{:.2f}%"
            })\
            .applymap(lambda v: apply_text_styling(v, 'return'), subset=["S2_Return"])\
            .applymap(lambda v: apply_text_styling(v, 'quality'), subset=["Signal Quality"])\
            .applymap(lambda v: apply_text_styling(v, 'standard'), subset=["S2_Persistence", "RS Rating"])\
            .applymap(lambda v: apply_text_styling(v, 'inverse'), subset=["S2_Failure_Risk"])
            st.dataframe(styled, use_container_width=True, hide_index=True)
            copy_tv(df_s2.to_dict('records'))

    with tab_ath:
        df_ath = pd.DataFrame(ath)
        if df_ath.empty:
            st.info("No ATH breakouts found.")
        else:
            all_tickers = sorted(df_ath["Ticker"].unique())
            selected_tickers = st.multiselect("Filter by Ticker", all_tickers, key="ath_filter")
            
            if selected_tickers:
                df_ath = df_ath[df_ath["Ticker"].isin(selected_tickers)]

            entries_count = len(df_ath)
            unique_count = df_ath["Ticker"].nunique()
            st.markdown(f"<div class='metric-box'>{entries_count} Entries | {unique_count} Unique Stocks</div>", unsafe_allow_html=True)

            cols = ["Ticker", "ATH Date", "ATH_Event_Price", "LTP", "ATH_Return", "RS Rating", "ATH_Persistence", "ATH_Failure_Risk"]
            styled = df_ath[cols].style.format({
                "ATH_Event_Price": "₹ {:.2f}", "LTP": "₹ {:.2f}", "ATH_Return": "{:.2f}%"
            })\
            .applymap(lambda v: apply_text_styling(v, 'return'), subset=["ATH_Return"])\
            .applymap(lambda v: apply_text_styling(v, 'inverse'), subset=["ATH_Failure_Risk"])\
            .applymap(lambda v: apply_text_styling(v, 'standard'), subset=["ATH_Persistence", "RS Rating"])
            st.dataframe(styled, use_container_width=True, hide_index=True)
            copy_tv(df_ath.to_dict('records'))

    with tab_pop:
        df_pop = pd.DataFrame(breakouts)
        if df_pop.empty:
            st.info("No Volume Poppers found.")
        else:
            all_tickers = sorted(df_pop["Ticker"].unique())
            selected_tickers = st.multiselect("Filter by Ticker", all_tickers, key="pop_filter")
            
            if selected_tickers:
                df_pop = df_pop[df_pop["Ticker"].isin(selected_tickers)]

            entries_count = len(df_pop)
            unique_count = df_pop["Ticker"].nunique()
            st.markdown(f"<div class='metric-box'>{entries_count} Entries | {unique_count} Unique Stocks</div>", unsafe_allow_html=True)

            cols = ["Ticker", "Pop Date", "Pop_Event_Price", "LTP", "Pop_Return", "RS Rating", "Pop_Vol_Expansion", "Pop_Failure_Risk", "Pop_Persistence"]
            styled = df_pop[cols].style.format({
                "Pop_Event_Price": "₹ {:.2f}", "LTP": "₹ {:.2f}", "Pop_Return": "{:.2f}%", "Pop_Vol_Expansion": "{:.2f}x"
            })\
            .applymap(lambda v: apply_text_styling(v, 'return'), subset=["Pop_Return"])\
            .applymap(lambda v: apply_text_styling(v, 'vol_expansion'), subset=["Pop_Vol_Expansion"])\
            .applymap(lambda v: apply_text_styling(v, 'inverse'), subset=["Pop_Failure_Risk"])\
            .applymap(lambda v: apply_text_styling(v, 'standard'), subset=["Pop_Persistence", "RS Rating"])
            st.dataframe(styled, use_container_width=True, hide_index=True)
            copy_tv(df_pop.to_dict('records'))

    with tab_trend:
        if full_scan:
            df = pd.DataFrame(full_scan)
            df = df[df['Rocket Score'] >= 50].sort_values(by="Rocket Score", ascending=False)
            
            st.markdown(f"<div class='metric-box'>Found {len(df)} Trend Watch Setups</div>", unsafe_allow_html=True)
            
            cols = ["Ticker", "Price", "Rocket Score", "Trend Score", "RS %", "Tightness %", "Vol Dry Score", "Near Breakout", "Failure Risk", "Persistence"]
            styled = df[cols].style.format({"Price": "₹ {:.2f}", "Rocket Score": "{:.2f}"})\
                .applymap(lambda v: apply_text_styling(v, 'standard'), subset=["Rocket Score", "Trend Score", "RS %", "Tightness %", "Vol Dry Score", "Near Breakout", "Persistence"])\
                .applymap(lambda v: apply_text_styling(v, 'inverse'), subset=["Failure Risk"])
            st.dataframe(styled, use_container_width=True, hide_index=True)
            copy_tv(df.to_dict('records'))
        else: st.info("No Trend Watch Setups found.")

    with tab_base:
        if full_scan:
            df = pd.DataFrame(full_scan)
            df_base = df[df['RS %'] >= 60].copy()
            df_base = df_base[df_base["Basic VCP"] == True]
            df_base = df_base[ (df_base["Within 25% 52W High"] == True) | (df_base["Elite VCP"] == True) ]
            
            st.markdown(f"<div class='metric-box'>Found {len(df_base)} Base Structures (RS >= 60)</div>", unsafe_allow_html=True)
            
            if not df_base.empty:
                df_base = df_base.sort_values(
                    by=["Elite VCP", "Base Duration", "Dist to Breakout %", "RS %"], 
                    ascending=[False, False, True, False]
                )

                cols = ["Ticker", "Price", "Base Duration", "Within 25% 52W High", "Elite VCP", "VCP Range %", "Dist to Breakout %", "RS %", "Vol Expansion"]
                
                styled = df_base[cols].style.format({
                    "Price": "₹ {:.2f}", 
                    "VCP Range %": "{:.2f}%", 
                    "Dist to Breakout %": "{:.2f}%",
                    "Vol Expansion": "{:.2f}x"
                })\
                .applymap(lambda v: apply_text_styling(v, 'standard'), subset=["Elite VCP", "Within 25% 52W High"])\
                .applymap(lambda v: apply_text_styling(v, 'standard'), subset=["RS %"])\
                .applymap(lambda v: apply_text_styling(v, 'vol_expansion'), subset=["Vol Expansion"])
                
                st.dataframe(styled, use_container_width=True, hide_index=True)
                copy_tv(df_base.to_dict('records'))
            else:
                st.warning("No stocks found matching the criteria (RS>=60, Basic VCP + [Elite VCP OR Near 52WH]).")
        else: st.info("No data available.")

    with tab_cmbf:
        df_cmbf = pd.DataFrame(cmbf)
        if df_cmbf.empty:
            st.info("No CMBF setups found.")
        else:
            df_cmbf = df_cmbf[df_cmbf["Grade"] != "Reject"]
            
            if df_cmbf.empty:
                st.info("No valid CMBF setups found after removing Rejects.")
            else:
                all_tickers = sorted(df_cmbf["Ticker"].unique())
                selected_tickers = st.multiselect("Filter by Ticker", all_tickers, key="cmbf_filter")
                
                if selected_tickers:
                    df_cmbf = df_cmbf[df_cmbf["Ticker"].isin(selected_tickers)]

                entries_count = len(df_cmbf)
                unique_count = df_cmbf["Ticker"].nunique()
                st.markdown(f"<div class='metric-box'>{entries_count} Entries | {unique_count} Unique Stocks</div>", unsafe_allow_html=True)
                
                df_cmbf = df_cmbf.sort_values(by=["CMBF Date", "Grade", "ATR %"], ascending=[False, True, True])
                
                cols = ["Ticker", "CMBF Date", "Price", "Grade", "Protocol", "Daily RSI", "Weekly RSI", "Box Range %", "ATR %"]
                styled = df_cmbf[cols].style.format({        
                    "Price": "₹ {:.2f}",            
                    "ATR %": "{:.2f}%",            
                    "Box Range %": "{:.2f}%"        
                })        
                st.dataframe(styled, use_container_width=True, hide_index=True)        
                copy_tv(df_cmbf.to_dict('records'))

    with tab_breadth:
        st.markdown("### 📊 Market Regime & Internals")
        if breadth:
            df = pd.DataFrame(breadth)
            cols = ["Date", "New Highs", "New Lows", "Net New Highs", "AD Slope 20D", "AD Change 20D", "Rolling BO Success 10D", "% Above 20 DMA", "% Breaking < 20 DMA", "% Above 200 DMA"]
            def color_breadth(val):
                if isinstance(val, (int, float)):
                    if val > 70: return 'color: #008000; font-weight: bold;'
                    elif val > 40: return 'color: #DAA520; font-weight: bold;'
                    else: return 'color: #FF0000; font-weight: bold;'
                return ''
            def color_slope(val): return 'color: #008000; font-weight: bold;' if val > 0 else 'color: #FF0000; font-weight: bold;'

            styled = df[cols].style.format({
                "AD Slope 20D": "{:.2f}%", "AD Change 20D": "{:.2f}", "Rolling BO Success 10D": "{:.1f}%",
                "% Above 20 DMA": "{:.2f}%", "% Above 200 DMA": "{:.2f}%", "% Breaking < 20 DMA": "{:.2f}%"
            })\
            .applymap(color_breadth, subset=["% Above 20 DMA", "% Above 200 DMA"])\
            .applymap(lambda v: apply_text_styling(v, 'bo_success'), subset=["Rolling BO Success 10D"])\
            .applymap(color_slope, subset=["AD Slope 20D", "Net New Highs", "AD Change 20D"])
            st.dataframe(styled, use_container_width=True, hide_index=True)
        else: st.info("No Breadth Data.")
