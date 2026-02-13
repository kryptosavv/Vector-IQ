import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import timedelta, date
import os

# --- 1. CONFIGURATION & BRANDING ---
st.set_page_config(
    page_title="NSE Vanguard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for branding
st.markdown("""
    <style>
    .main-title {font-size: 3em; font-weight: bold; color: #FF4B4B;}
    .sub-title {font-size: 1.2em; color: #555;}
    </style>
""", unsafe_allow_html=True)

# --- 2. DATA LOADING LAYER (Cached) ---
@st.cache_data(show_spinner=False, ttl=3600) # Cache data for 1 hour
def download_data(tickers):
    """
    Downloads historical data for all tickers.
    Cached to prevent re-downloading when changing date filters.
    """
    if not tickers:
        return pd.DataFrame()
    
    try:
        # Batch download from Yahoo Finance
        data = yf.download(
            tickers,
            period="max",
            group_by='ticker',
            threads=True,
            progress=False
        )
        return data
    except Exception as e:
        st.error(f"Download API failed: {e}")
        return pd.DataFrame()

# --- 3. LOGIC LAYER (Processing) ---
def process_single_ticker(ticker, df, start_date, end_date):
    """
    Analyzes a single ticker dataframe for ATH and Yearly High breakouts.
    Returns a dict if a breakout is found, else None.
    """
    try:
        # Clean data: Drop rows where OHLC is missing
        df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
        
        if df.empty: return None, None
        
        # Ensure index is datetime
        df.index = pd.to_datetime(df.index)

        # 1. Calculate Historical Levels
        # Prev 52W High (Yearly High) - Rolling 252 days, shifted by 1 to exclude today
        df['Prev_Yearly_High'] = df['High'].rolling(window=252).max().shift(1)
        # Prev All-Time High - Expanding max, shifted by 1
        df['Prev_ATH'] = df['High'].expanding().max().shift(1)
        
        # 2. Filter for User Date Range
        mask_range = (df.index.date >= start_date) & (df.index.date <= end_date)
        range_df = df.loc[mask_range].copy()
        
        if range_df.empty: return None, None

        # 3. Detect Breakouts
        ath_breakouts = range_df[range_df['High'] > range_df['Prev_ATH']]
        yh_breakouts = range_df[range_df['High'] > range_df['Prev_Yearly_High']]

        # 4. Format Result Helper
        def format_result(breakout_df):
            if breakout_df.empty: return None
            
            # First occurrence in the selected window
            first = breakout_df.iloc[0]
            b_date = breakout_df.index[0].date()
            b_price = first['High']
            
            # Safe latest price fetch
            try:
                latest_valid = df.dropna(subset=['Close']).iloc[-1]
                latest_price = latest_valid['Close']
            except:
                latest_price = b_price # Fallback

            ret_pct = ((latest_price - b_price) / b_price) * 100
            
            return {
                "Ticker": ticker.replace(".NS", ""),
                "Breakout Date": b_date,
                "Breakout Price": round(b_price, 2),
                "CMP": round(latest_price, 2),
                "Return (%)": round(ret_pct, 2),
            }

        return format_result(ath_breakouts), format_result(yh_breakouts)

    except Exception:
        return None, None

def scan_stocks(tickers, start_date, end_date, progress_bar, status_text):
    """
    Orchestrates the download and processing.
    """
    # 1. Download Data (Cached)
    status_text.text("🔌 Connecting to NSE Server... (Downloading Data)")
    
    raw_data = download_data(tickers)
    
    # --- CRITICAL ERROR PROTECTION ---
    if raw_data.empty:
        st.error("⚠️ Data download failed or returned empty. Please check your internet or retry.")
        return [], []
    # ---------------------------------

    ath_list = []
    yh_list = []
    
    # Safe Ticker Extraction
    # Handles MultiIndex columns safely to avoid failed downloads
    if isinstance(raw_data.columns, pd.MultiIndex):
        downloaded_tickers = list(set([col[0] for col in raw_data.columns]))
    else:
        # Fallback if only 1 ticker or flat structure
        downloaded_tickers = tickers

    total = len(downloaded_tickers)
    
    # 2. Process Data
    for idx, ticker in enumerate(downloaded_tickers):
        progress_bar.progress((idx + 1) / total)
        status_text.text(f"Analyzing {ticker}...")
        
        try:
            # Extract Ticker DF
            if total == 1:
                df = raw_data.copy()
            else:
                df = raw_data[ticker].copy()
            
            # Run Logic
            ath_res, yh_res = process_single_ticker(ticker, df, start_date, end_date)
            
            if ath_res: ath_list.append(ath_res)
            if yh_res: yh_list.append(yh_res)
            
        except Exception:
            continue
            
    return ath_list, yh_list

# --- 4. HELPER FUNCTIONS ---
def load_tickers():
    file_path = "NiftyTM.txt" 
    tickers = []
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            tickers = [line.strip() for line in f.readlines() if line.strip()]
    
    if not tickers:
        st.sidebar.warning(f"'{file_path}' not found. Please upload it.")
        uploaded_file = st.sidebar.file_uploader("Upload Ticker List (txt)", type=["txt"])
        if uploaded_file is not None:
            string_data = uploaded_file.read().decode("utf-8")
            tickers = [line.strip() for line in string_data.splitlines() if line.strip()]
    return tickers

def generate_tradingview_batches(data_list, batch_size=30):
    if not data_list: return []
    tv_tickers = [f"NSE:{item['Ticker']}" for item in data_list]
    batches = [tv_tickers[i:i + batch_size] for i in range(0, len(tv_tickers), batch_size)]
    return [", ".join(batch) for batch in batches]

# --- 5. UI SETUP ---

# Sidebar for Inputs
with st.sidebar:
    st.title("⚙️ Configuration")
    TICKER_LIST = load_tickers()
    
    st.divider()
    
    st.write("### 📅 Time Period")
    preset = st.radio(
        "Select Range:", 
        ["Today", "Yesterday", "This Week", "Last Week", "This Month", "Custom"], 
    )

    today = date.today()

    if preset == "Today":
        start_d = today
        end_d = today
    elif preset == "Yesterday":
        start_d = today - timedelta(days=1)
        end_d = start_d
    elif preset == "This Week":
        start_d = today - timedelta(days=today.weekday())
        end_d = today
    elif preset == "Last Week":
        start_current = today - timedelta(days=today.weekday())
        start_d = start_current - timedelta(days=7)
        end_d = start_current - timedelta(days=1)
    elif preset == "This Month":
        start_d = date(today.year, today.month, 1)
        end_d = today
    else: # Custom
        start_d = st.date_input("Start", value=today - timedelta(days=7))
        end_d = st.date_input("End", value=today)
        
    st.divider()
    run_btn = st.button("🚀 Run Vanguard Scan", type="primary", use_container_width=True)

# Main Area
st.markdown('<div class="main-title">NSE Vanguard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Advanced Momentum & Breakout Scanner</div>', unsafe_allow_html=True)
st.caption(f"🔎 Scanning Range: **{start_d}** to **{end_d}**")
st.divider()

if not TICKER_LIST:
    st.info("👈 Please upload 'NiftyTM.txt' in the sidebar to start.")
    st.stop()

if run_btn:
    bar = st.progress(0)
    status = st.empty()
    
    # Process
    ath_results, yh_results = scan_stocks(TICKER_LIST, start_d, end_d, bar, status)
    
    bar.empty()
    status.empty()
    
    # Display Results
    tab1, tab2 = st.tabs(["🚀 All-Time Highs (ATH)", "📈 Yearly Highs (52W)"])

    def show_tab_content(results, label):
        if results:
            st.success(f"Found {len(results)} {label} Breakouts!")
            df = pd.DataFrame(results).sort_values(by="Breakout Date", ascending=False)
            
            # Interactive Table
            st.dataframe(
                df, 
                use_container_width=True, 
                hide_index=True,
                column_config={
                    "Return (%)": st.column_config.NumberColumn(format="%.2f %%"),
                    "Breakout Price": st.column_config.NumberColumn(format="₹ %.2f"),
                    "CMP": st.column_config.NumberColumn(format="₹ %.2f"),
                }
            )
            
            st.markdown("### 📋 TradingView Watchlist")
            batches = generate_tradingview_batches(results, 30)
            for i, batch in enumerate(batches):
                st.caption(f"Batch {i+1} ({len(batch.split(','))} stocks)")
                st.code(batch, language="text")
        else:
            st.info(f"No stocks hit {label} in this period.")

    with tab1: show_tab_content(ath_results, "ATH")
    with tab2: show_tab_content(yh_results, "Yearly High")
