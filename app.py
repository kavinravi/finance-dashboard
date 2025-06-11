import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Advanced data collection imports
import yfinance as yf
import requests
from datetime import datetime, timedelta
import time
import os
from dotenv import load_dotenv

# Load environment variables for local development
load_dotenv()

# Machine Learning imports
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Set page config
st.set_page_config(
    page_title="Finance Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

def calculate_mape(actual, predicted):
    """Calculate Mean Absolute Percentage Error"""
    return np.mean(np.abs((actual - predicted) / actual)) * 100

def explain_arima_results():
    """Return user-friendly explanation of ARIMA results"""
    return """
    📊 **Understanding Your ARIMA Results:**
    
    • **MAPE (Mean Absolute Percentage Error)**: How far off predictions are on average, as a percentage. 
      - Under 5%: Excellent accuracy for trading decisions
      - 5-10%: Good accuracy for portfolio planning  
      - 10-20%: Fair accuracy for long-term trends
      - Over 20%: Poor accuracy, use with caution
    
    • **R² (R-Squared)**: How well the model explains price movements (0-100%).
      - Above 80%: Strong predictive power
      - 60-80%: Moderate predictive power
      - Below 60%: Weak predictive power
    
    • **7-Day Forecast**: Predicted prices for the next week with confidence intervals.
      The wider the confidence band, the more uncertain the predictions.
    """

def explain_lstm_results():
    """Return user-friendly explanation of LSTM results"""
    return """
    🧠 **Understanding Your LSTM AI Results:**
    
    • **MAPE (Mean Absolute Percentage Error)**: How accurate the AI predictions are, as a percentage.
      - Under 3%: Excellent for day trading strategies
      - 3-7%: Good for swing trading decisions
      - 7-15%: Fair for medium-term planning
      - Over 15%: Poor accuracy, avoid for short-term decisions
    
    • **Next Day Prediction**: AI's best guess for tomorrow's closing price.
      Consider this alongside your fundamental analysis.
    
    • **Training vs Test Performance**: If test MAPE is much higher than training MAPE,
      the model may not work well on new data (overfitting).
    """

def validate_ticker_format(ticker):
    """Validate ticker symbol format"""
    if not ticker:
        return False, "❌ **Ticker required**: Please enter a ticker symbol."
    
    if len(ticker) < 1 or len(ticker) > 6:
        return False, "❌ **Ticker length**: Ticker must be 1-6 characters (e.g., AAPL, GOOGL)."
    
    if not ticker.isalpha():
        return False, "❌ **Ticker format**: Ticker must contain only letters (A-Z)."
    
    return True, ""

def validate_uploaded_data(df, max_size_mb=10):
    """Validate uploaded CSV data and return clear error messages"""
    required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    errors = []
    
    # Check file size (approximate)
    estimated_size_mb = len(df) * len(df.columns) * 50 / (1024 * 1024)  # Rough estimate
    if estimated_size_mb > max_size_mb:
        errors.append(f"❌ **File too large**: File appears to be ~{estimated_size_mb:.1f}MB. Please keep under {max_size_mb}MB.")
        return False, errors
    
    # Check if required columns exist (case-insensitive)
    df_columns_lower = [col.lower() for col in df.columns]
    required_lower = [col.lower() for col in required_columns]
    
    missing_columns = []
    for req_col in required_columns:
        if req_col.lower() not in df_columns_lower:
            missing_columns.append(req_col)
    
    if missing_columns:
        errors.append(f"❌ **Missing required columns**: {', '.join(missing_columns)}")
        errors.append(f"📋 **Required columns are**: {', '.join(required_columns)}")
        return False, errors
    
    # Check if we have data
    if len(df) == 0:
        errors.append("❌ **Empty file**: The CSV file contains no data rows.")
        return False, errors
    
    # Check date column format (try to parse a few rows)
    date_col = None
    for col in df.columns:
        if col.lower() == 'date':
            date_col = col
            break
    
    if date_col:
        try:
            pd.to_datetime(df[date_col].head(3))
        except:
            errors.append("❌ **Date format issue**: Could not parse the Date column. Please use formats like 'MM/DD/YYYY' or 'YYYY-MM-DD'.")
            return False, errors
    
    # Check if numeric columns contain valid data
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in numeric_cols:
        actual_col = None
        for df_col in df.columns:
            if df_col.lower() == col.lower():
                actual_col = df_col
                break
        
        if actual_col:
            # Try to convert to numeric (handle quotes, commas)
            sample_data = df[actual_col].head(3).astype(str).str.replace('"', '').str.replace(',', '')
            try:
                pd.to_numeric(sample_data)
            except:
                errors.append(f"❌ **{col} column format issue**: Contains non-numeric data. Please ensure {col} contains only numbers.")
                return False, errors
    
    return True, []

@st.cache_data(ttl=300)  # Cache for 5 minutes only
def load_data(use_uploaded=False, uploaded_data=None):
    """Load and preprocess stock data"""
    try:
        if use_uploaded and uploaded_data is not None:
            df = uploaded_data.copy()
        else:
            df = pd.read_csv('data/SPY.csv')
        
        # Normalize column names (handle case-insensitive)
        column_mapping = {}
        for col in df.columns:
            if col.lower() == 'date':
                column_mapping[col] = 'Date'
            elif col.lower() == 'open':
                column_mapping[col] = 'Open'
            elif col.lower() == 'high':
                column_mapping[col] = 'High'
            elif col.lower() == 'low':
                column_mapping[col] = 'Low'
            elif col.lower() == 'close':
                column_mapping[col] = 'Close'
            elif col.lower() == 'volume':
                column_mapping[col] = 'Volume'
        
        df = df.rename(columns=column_mapping)
        
        # Clean the data
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Handle Volume column (remove commas)
        if df['Volume'].dtype == 'object':
            df['Volume'] = df['Volume'].str.replace(',', '').astype(int)
        
        # Handle price columns - check if they're strings or already numeric
        for col in ['Open', 'High', 'Low', 'Close']:
            if df[col].dtype == 'object':
                # If they're strings, remove quotes and convert to float
                df[col] = df[col].str.replace('"', '').astype(float)
            # If they're already numeric, leave them as is
        
        # Sort by date
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Calculate additional technical indicators
        df['Daily_Return'] = df['Close'].pct_change()
        df['High_Low_Pct'] = (df['High'] - df['Low']) / df['Close'] * 100
        df['Price_Change'] = df['Close'] - df['Open']
        df['Volatility'] = df['Daily_Return'].rolling(window=5).std()
        
        return df
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info("Please check that the SPY.csv file is properly formatted.")
        return pd.DataFrame()  # Return empty dataframe to prevent crashes

def create_candlestick_chart(df, ticker="SPY"):
    """Create an interactive candlestick chart"""
    fig = go.Figure(data=go.Candlestick(
        x=df['Date'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name=ticker
    ))
    
    # Set the x-axis range to show full dataset by default
    fig.update_layout(
        title=f"{ticker} Candlestick Chart",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        height=500,
        showlegend=False,
        xaxis=dict(
            range=[df['Date'].min(), df['Date'].max()],
            rangeslider=dict(visible=True),  # Add range slider for easy navigation
            type="date"
        ),
        yaxis=dict(
            range=[df['Low'].min() * 0.98, df['High'].max() * 1.02]  # Add some padding
        )
    )
    
    return fig

def create_volume_chart(df, ticker="SPY"):
    """Create volume chart"""
    fig = px.bar(df, x='Date', y='Volume', 
                 title=f"{ticker} Trading Volume",
                 color='Volume',
                 color_continuous_scale='viridis')
    
    fig.update_layout(
        height=400,
        xaxis=dict(
            range=[df['Date'].min(), df['Date'].max()],
            type="date"
        )
    )
    return fig

def create_correlation_heatmap(df, ticker="SPY"):
    """Create correlation heatmap"""
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Daily_Return', 'High_Low_Pct']
    corr_matrix = df[numeric_cols].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
    plt.title(f'Correlation Matrix of {ticker} Features')
    return fig

def create_pairplot(df, ticker="SPY"):
    """Create seaborn pairplot"""
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    fig = plt.figure(figsize=(12, 10))
    
    # Create pairplot
    g = sns.pairplot(df[numeric_cols], diag_kind='hist', corner=True)
    g.fig.suptitle(f'{ticker} Stock Data Pairplot', y=1.02)
    return g.fig

def prepare_arima_data(df):
    """Prepare data for ARIMA model"""
    # Use closing prices for ARIMA
    ts_data = df.set_index('Date')['Close']
    return ts_data

def fit_arima_model(ts_data, order=(1,1,1)):
    """Fit ARIMA model and make predictions"""
    try:
        # Ensure we have enough data points
        if len(ts_data) < max(order) + 10:
            st.warning("Not enough data points for ARIMA analysis. Need at least 15 data points.")
            return None, None, None
        
        model = ARIMA(ts_data, order=order)
        fitted_model = model.fit()
        
        # Make predictions
        forecast_steps = 7  # Predict next 7 days
        forecast = fitted_model.forecast(steps=forecast_steps)
        conf_int = fitted_model.get_forecast(steps=forecast_steps).conf_int()
        
        return fitted_model, forecast, conf_int
    except Exception as e:
        st.error(f"ARIMA model error: {str(e)}")
        st.info("Try different ARIMA parameters (p, d, q) or ensure your data has enough observations.")
        return None, None, None

def prepare_lstm_data(df, sequence_length=10):
    """Prepare data for LSTM model"""
    # Use closing prices
    data = df['Close'].values.reshape(-1, 1)
    
    # Scale the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    
    # Create sequences
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    return X, y, scaler

class LSTMModel(nn.Module):
    """Enhanced PyTorch LSTM model for better stock prediction"""
    def __init__(self, input_size=1, hidden_size=128, num_layers=3, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Enhanced complexity for better predictions
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.dropout1 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.dropout2 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, 32)
        self.dropout3 = nn.Dropout(0.1)
        self.fc3 = nn.Linear(32, output_size)
        
    def forward(self, x):
        # Initialize hidden state with proper device placement
        device = x.device
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Take the last output and apply multiple dense layers
        out = out[:, -1, :]
        out = self.dropout1(out)
        out = torch.relu(self.fc1(out))
        out = self.dropout2(out)
        out = torch.relu(self.fc2(out))
        out = self.dropout3(out)
        out = self.fc3(out)
        
        return out

def create_lstm_model(sequence_length):
    """Create enhanced PyTorch LSTM model with GPU support"""
    model = LSTMModel(input_size=1, hidden_size=128, num_layers=3, output_size=1)
    
    # Move to best available device (CUDA > MPS > CPU)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    model = model.to(device)
    
    return model, device

def fetch_yfinance_data(ticker, start_date, end_date):
    """Fetch extended historical data using yfinance"""
    try:
        # Create yfinance ticker object
        stock = yf.Ticker(ticker)
        
        # Download historical data
        df = stock.history(start=start_date, end=end_date)
        
        if df.empty:
            return None, f"No data found for {ticker} in the specified date range."
        
        # Reset index to make Date a column
        df = df.reset_index()
        
        # Standardize column names
        df.columns = [col.title() for col in df.columns]
        
        # Add technical indicators
        df['Daily_Return'] = df['Close'].pct_change()
        df['High_Low_Pct'] = (df['High'] - df['Low']) / df['Close'] * 100
        df['Price_Change'] = df['Close'] - df['Open']
        df['Volatility'] = df['Daily_Return'].rolling(window=5).std()
        
        # Add moving averages
        df['MA_10'] = df['Close'].rolling(window=10).mean()
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        df['MA_50'] = df['Close'].rolling(window=50).mean()
        
        # Add RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        return df, None
        
    except Exception as e:
        return None, f"Error fetching data for {ticker}: {str(e)}"

def validate_date_range(start_date, end_date):
    """Validate the date range for data collection"""
    if start_date >= end_date:
        return False, "Start date must be before end date."
    
    if end_date > datetime.now().date():
        return False, "End date cannot be in the future."
    
    # Check if date range is too large (more than 10 years for free APIs)
    date_diff = (end_date - start_date).days
    if date_diff > 3650:  # ~10 years
        return False, "Date range too large. Please select a range within 10 years for optimal performance."
    
    return True, ""

def calculate_advanced_indicators(df):
    """Calculate additional technical indicators"""
    try:
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        
        # MACD
        exp1 = df['Close'].ewm(span=12).mean()
        exp2 = df['Close'].ewm(span=26).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        return df
    except Exception as e:
        st.warning(f"Could not calculate some advanced indicators: {str(e)}")
        return df

@st.cache_data(ttl=600)  # Cache for 10 minutes
def get_ticker_info(ticker):
    """Get basic ticker information using yfinance"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        return {
            'name': info.get('longName', ticker),
            'sector': info.get('sector', 'Unknown'),
            'industry': info.get('industry', 'Unknown'),
            'market_cap': info.get('marketCap', 'N/A'),
            'currency': info.get('currency', 'USD')
        }
    except:
        return {
            'name': ticker,
            'sector': 'Unknown',
            'industry': 'Unknown', 
            'market_cap': 'N/A',
            'currency': 'USD'
        }

def get_api_keys():
    """Securely get API keys from environment variables or Streamlit secrets"""
    try:
        # Try Streamlit secrets first (for cloud deployment)
        if hasattr(st, 'secrets'):
            return {
                'fmp': st.secrets.get('FMP_API_KEY', ''),
                'alpha_vantage': st.secrets.get('ALPHA_VANTAGE_API_KEY', ''),
                'fmp_rate_limit': int(st.secrets.get('FMP_RATE_LIMIT', 60)),
                'alpha_rate_limit': int(st.secrets.get('ALPHA_VANTAGE_RATE_LIMIT', 5))
            }
    except:
        pass
    
    # Fall back to environment variables (for local development)
    return {
        'fmp': os.getenv('FMP_API_KEY', ''),
        'alpha_vantage': os.getenv('ALPHA_VANTAGE_API_KEY', ''),
        'fmp_rate_limit': int(os.getenv('FMP_RATE_LIMIT', 60)),
        'alpha_rate_limit': int(os.getenv('ALPHA_VANTAGE_RATE_LIMIT', 5))
    }

def check_api_keys_available():
    """Check if API keys are properly configured"""
    keys = get_api_keys()
    return {
        'fmp_available': bool(keys['fmp']),
        'alpha_available': bool(keys['alpha_vantage']),
        'keys': keys
    }

@st.cache_data(ttl=300)  # Cache for 5 minutes to respect rate limits
def fetch_fmp_fundamentals(ticker, api_key):
    """Fetch fundamental data from Financial Modeling Prep"""
    try:
        # Company profile
        profile_url = f"https://financialmodelingprep.com/api/v3/profile/{ticker}"
        profile_params = {'apikey': api_key}
        
        profile_response = requests.get(profile_url, params=profile_params, timeout=10)
        
        if profile_response.status_code != 200:
            return None, f"FMP API error: {profile_response.status_code}"
        
        profile_data = profile_response.json()
        if not profile_data:
            return None, f"No fundamental data found for {ticker}"
        
        # Add small delay to respect rate limits
        time.sleep(1)
        
        # Financial ratios
        ratios_url = f"https://financialmodelingprep.com/api/v3/ratios/{ticker}"
        ratios_params = {'apikey': api_key, 'limit': 1}
        
        ratios_response = requests.get(ratios_url, params=ratios_params, timeout=10)
        ratios_data = ratios_response.json() if ratios_response.status_code == 200 else []
        
        time.sleep(1)  # Rate limiting
        
        # Income statement (latest year)
        income_url = f"https://financialmodelingprep.com/api/v3/income-statement/{ticker}"
        income_params = {'apikey': api_key, 'limit': 1}
        
        income_response = requests.get(income_url, params=income_params, timeout=10)
        income_data = income_response.json() if income_response.status_code == 200 else []
        
        # Safely extract data with error handling
        result_data = {
            'profile': profile_data[0] if isinstance(profile_data, list) and profile_data else (profile_data if isinstance(profile_data, dict) else {}),
            'ratios': ratios_data[0] if isinstance(ratios_data, list) and ratios_data else (ratios_data if isinstance(ratios_data, dict) else {}),
            'income': income_data[0] if isinstance(income_data, list) and income_data else (income_data if isinstance(income_data, dict) else {})
        }
        
        return result_data, None
        
    except requests.RequestException as e:
        return None, f"FMP API request failed: {str(e)}"
    except Exception as e:
        return None, f"FMP data processing error: {str(e)}"

@st.cache_data(ttl=300)  # Cache for 5 minutes to respect rate limits
def fetch_alpha_vantage_indicators(ticker, api_key):
    """Fetch additional technical indicators from Alpha Vantage"""
    try:
        indicators = {}
        
        # RSI
        rsi_url = "https://www.alphavantage.co/query"
        rsi_params = {
            'function': 'RSI',
            'symbol': ticker,
            'interval': 'daily',
            'time_period': 14,
            'series_type': 'close',
            'apikey': api_key
        }
        
        time.sleep(12)  # Rate limiting - Alpha Vantage free tier allows 5 requests per minute
        rsi_response = requests.get(rsi_url, params=rsi_params, timeout=15)
        
        if rsi_response.status_code == 200:
            rsi_data = rsi_response.json()
            if 'Technical Analysis: RSI' in rsi_data:
                indicators['rsi'] = rsi_data['Technical Analysis: RSI']
        
        # MACD
        macd_url = "https://www.alphavantage.co/query"
        macd_params = {
            'function': 'MACD',
            'symbol': ticker,
            'interval': 'daily',
            'series_type': 'close',
            'apikey': api_key
        }
        
        time.sleep(12)  # Rate limiting
        macd_response = requests.get(macd_url, params=macd_params, timeout=15)
        
        if macd_response.status_code == 200:
            macd_data = macd_response.json()
            if 'Technical Analysis: MACD' in macd_data:
                indicators['macd'] = macd_data['Technical Analysis: MACD']
        
        return indicators, None
        
    except requests.RequestException as e:
        return None, f"Alpha Vantage API request failed: {str(e)}"
    except Exception as e:
        return None, f"Alpha Vantage data processing error: {str(e)}"

def format_fundamental_data(fmp_data):
    """Format FMP fundamental data for display with robust error handling"""
    if not fmp_data:
        return {}
    
    profile = fmp_data.get('profile', {})
    ratios = fmp_data.get('ratios', {})
    income = fmp_data.get('income', {})
    
    def safe_format_number(value, format_type='currency', decimals=2):
        """Safely format numbers with error handling"""
        try:
            if value is None or value == '':
                return 'N/A'
            
            # Convert to float if it's a string
            if isinstance(value, str):
                # Remove common formatting characters
                clean_value = value.replace(',', '').replace('$', '').replace('%', '')
                if clean_value == '' or clean_value.lower() in ['n/a', 'null', 'none']:
                    return 'N/A'
                value = float(clean_value)
            
            if not isinstance(value, (int, float)):
                return 'N/A'
            
            if format_type == 'currency':
                return f"${value:,.{decimals}f}"
            elif format_type == 'percentage':
                return f"{value:.{decimals}%}"
            elif format_type == 'number':
                return f"{value:,.{decimals}f}"
            elif format_type == 'count':
                return f"{int(value):,}"
            else:
                return f"{value:.{decimals}f}"
                
        except (ValueError, TypeError, AttributeError):
            return 'N/A'
    
    def safe_get_string(data, key, default='N/A'):
        """Safely get string values"""
        value = data.get(key, default)
        return str(value) if value is not None else default
    
    return {
        'Company Info': {
            'Name': safe_get_string(profile, 'companyName'),
            'Sector': safe_get_string(profile, 'sector'),
            'Industry': safe_get_string(profile, 'industry'),
            'Country': safe_get_string(profile, 'country'),
            'Exchange': safe_get_string(profile, 'exchangeShortName'),
            'Market Cap': safe_format_number(profile.get('mktCap'), 'currency', 0),
            'Employees': safe_format_number(profile.get('fullTimeEmployees'), 'count', 0)
        },
        'Valuation Ratios': {
            'P/E Ratio': safe_format_number(ratios.get('priceEarningsRatio'), 'number'),
            'P/B Ratio': safe_format_number(ratios.get('priceToBookRatio'), 'number'),
            'P/S Ratio': safe_format_number(ratios.get('priceToSalesRatio'), 'number'),
            'EV/EBITDA': safe_format_number(ratios.get('enterpriseValueMultiple'), 'number'),
            'Dividend Yield': safe_format_number(ratios.get('dividendYield'), 'percentage')
        },
        'Financial Health': {
            'Current Ratio': safe_format_number(ratios.get('currentRatio'), 'number'),
            'Debt to Equity': safe_format_number(ratios.get('debtEquityRatio'), 'number'),
            'ROE': safe_format_number(ratios.get('returnOnEquity'), 'percentage'),
            'ROA': safe_format_number(ratios.get('returnOnAssets'), 'percentage'),
            'Gross Margin': safe_format_number(ratios.get('grossProfitMargin'), 'percentage')
        },
        'Latest Financials': {
            'Revenue': safe_format_number(income.get('revenue'), 'currency', 0),
            'Net Income': safe_format_number(income.get('netIncome'), 'currency', 0),
            'EPS': safe_format_number(income.get('eps'), 'currency'),
            'Operating Income': safe_format_number(income.get('operatingIncome'), 'currency', 0)
        }
    }

# Benchmark comparison functions
@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_benchmark_data(benchmark_ticker, start_date, end_date):
    """Fetch benchmark data using yfinance"""
    try:
        benchmark = yf.Ticker(benchmark_ticker)
        df = benchmark.history(start=start_date, end=end_date)
        
        if df.empty:
            return None, f"No data found for {benchmark_ticker}"
        
        df = df.reset_index()
        df.columns = [col.title() for col in df.columns]
        
        # Calculate returns
        df['Daily_Return'] = df['Close'].pct_change()
        df['Cumulative_Return'] = (1 + df['Daily_Return']).cumprod() - 1
        
        return df, None
    except Exception as e:
        return None, f"Error fetching {benchmark_ticker}: {str(e)}"

@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_fed_funds_rate_alpha_vantage(api_key, start_date, end_date):
    """Fetch Federal Funds Rate data from Alpha Vantage"""
    try:
        import requests
        import time
        
        url = "https://www.alphavantage.co/query"
        params = {
            'function': 'FEDERAL_FUNDS_RATE',
            'interval': 'daily',
            'apikey': api_key
        }
        
        response = requests.get(url, params=params, timeout=30)
        
        if response.status_code != 200:
            return None, f"Alpha Vantage API error: {response.status_code}"
        
        data = response.json()
        
        if 'Error Message' in data:
            return None, f"Alpha Vantage error: {data['Error Message']}"
        
        if 'Note' in data:
            return None, "Alpha Vantage rate limit exceeded. Please try again later."
        
        if 'data' not in data:
            return None, "No Federal Funds Rate data available from Alpha Vantage"
        
        # Convert to DataFrame
        df = pd.DataFrame(data['data'])
        df['date'] = pd.to_datetime(df['date'])
        df['value'] = pd.to_numeric(df['value'])
        
        # Filter by date range
        df = df[(df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))]
        
        if df.empty:
            return None, "No Federal Funds Rate data available for the specified date range"
        
        # Rename columns to match expected format
        df = df.rename(columns={'date': 'Date', 'value': 'Close'})
        df['Open'] = df['Close']  # For rate data, open = close
        df['High'] = df['Close']
        df['Low'] = df['Close']
        df['Volume'] = 1000000  # Dummy volume for rate data
        df['Daily_Return'] = df['Close'].pct_change()
        
        # Sort by date
        df = df.sort_values('Date').reset_index(drop=True)
        
        return df, None
        
    except Exception as e:
        return None, f"Error fetching Federal Funds Rate: {str(e)}"

def get_available_benchmarks():
    """Return available benchmarks with descriptions"""
    return {
        'FED_FUNDS': 'Federal Funds Rate',
        '^GSPC': 'S&P 500 Index',
        '^IXIC': 'NASDAQ Composite',
        '^DJI': 'Dow Jones Industrial Average',
        '^RUT': 'Russell 2000',
        '^IRX': '13-Week Treasury Bill Rate (Fed Funds Proxy) ⚠️',
        '^TNX': '10-Year Treasury Note Yield ⚠️',
        '^VIX': 'VIX Volatility Index',
        'SPY': 'SPDR S&P 500 ETF',
        'QQQ': 'Invesco QQQ Trust',
        'IWM': 'iShares Russell 2000 ETF',
        'TLT': '20+ Year Treasury Bond ETF',
        'GLD': 'SPDR Gold Shares',
        'USO': 'United States Oil Fund',
        'EFA': 'iShares MSCI EAFE ETF',
        'EEM': 'iShares MSCI Emerging Markets ETF'
    }

def calculate_comparison_metrics(stock_data, benchmark_data, stock_ticker, benchmark_ticker):
    """Calculate comprehensive comparison metrics"""
    try:
        # Align data by date
        stock_df = stock_data.copy()
        benchmark_df = benchmark_data.copy()
        
        # Ensure both have Date column
        if 'Date' not in stock_df.columns:
            stock_df = stock_df.reset_index()
        if 'Date' not in benchmark_df.columns:
            benchmark_df = benchmark_df.reset_index()
        
        # Debug: Check date types and formats before merge
        print(f"DEBUG - Stock dates dtype: {stock_df['Date'].dtype}")
        print(f"DEBUG - Benchmark dates dtype: {benchmark_df['Date'].dtype}")
        print(f"DEBUG - Stock date sample: {stock_df['Date'].head(3).tolist()}")
        print(f"DEBUG - Benchmark date sample: {benchmark_df['Date'].head(3).tolist()}")
        
        # Ensure both Date columns are datetime
        stock_df['Date'] = pd.to_datetime(stock_df['Date'])
        benchmark_df['Date'] = pd.to_datetime(benchmark_df['Date'])
        
        # Normalize dates to remove time components (keep only date part)
        stock_df['Date'] = stock_df['Date'].dt.date
        benchmark_df['Date'] = benchmark_df['Date'].dt.date
        
        # Merge on date
        merged = pd.merge(stock_df[['Date', 'Close', 'Daily_Return']], 
                         benchmark_df[['Date', 'Close', 'Daily_Return']], 
                         on='Date', suffixes=('_stock', '_benchmark'))
        
        print(f"DEBUG - After merge: {len(merged)} overlapping days")
        
        if len(merged) < 10:
            # Provide detailed error information with debugging
            stock_date_range = f"{stock_df['Date'].min()} to {stock_df['Date'].max()}"
            benchmark_date_range = f"{benchmark_df['Date'].min()} to {benchmark_df['Date'].max()}"
            
            # Show sample dates for debugging
            stock_sample_dates = stock_df['Date'].head(5).tolist()
            benchmark_sample_dates = benchmark_df['Date'].head(5).tolist()
            
            error_msg = f"""Insufficient overlapping data for comparison:
            
• {stock_ticker} has {len(stock_df)} data points ({stock_date_range})
• {benchmark_ticker} has {len(benchmark_df)} data points ({benchmark_date_range})
• Only {len(merged)} overlapping trading days found (need at least 10)

**Debugging Info:**
• Stock sample dates: {stock_sample_dates}
• Benchmark sample dates: {benchmark_sample_dates}

This suggests a data processing issue. Try:
- Different comparison period
- Different securities
- Report this as a bug if the date ranges look identical"""
            
            return None, error_msg
        
        # Calculate metrics - ensure we drop NaN values from both series simultaneously
        valid_data = merged[['Daily_Return_stock', 'Daily_Return_benchmark']].dropna()
        
        if len(valid_data) < 10:
            return None, "Insufficient valid return data for comparison"
        
        stock_returns = valid_data['Daily_Return_stock']
        benchmark_returns = valid_data['Daily_Return_benchmark']
        
        # Basic statistics
        stock_total_return = (merged['Close_stock'].iloc[-1] / merged['Close_stock'].iloc[0] - 1) * 100
        benchmark_total_return = (merged['Close_benchmark'].iloc[-1] / merged['Close_benchmark'].iloc[0] - 1) * 100
        
        stock_volatility = stock_returns.std() * np.sqrt(252) * 100  # Annualized
        benchmark_volatility = benchmark_returns.std() * np.sqrt(252) * 100
        
        stock_sharpe = (stock_returns.mean() / stock_returns.std() * np.sqrt(252)) if stock_returns.std() > 0 else 0
        benchmark_sharpe = (benchmark_returns.mean() / benchmark_returns.std() * np.sqrt(252)) if benchmark_returns.std() > 0 else 0
        
        # Beta calculation
        covariance = np.cov(stock_returns, benchmark_returns)[0][1]
        benchmark_variance = np.var(benchmark_returns)
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
        
        # Alpha calculation (excess return over benchmark adjusted for beta)
        alpha = (stock_returns.mean() - beta * benchmark_returns.mean()) * 252 * 100  # Annualized
        
        # Correlation
        correlation = np.corrcoef(stock_returns, benchmark_returns)[0][1] if len(stock_returns) > 1 else 0
        
        # Maximum drawdown - use the same valid data
        stock_cumulative = (1 + stock_returns).cumprod()
        stock_max_dd = ((stock_cumulative / stock_cumulative.cummax()) - 1).min() * 100
        
        benchmark_cumulative = (1 + benchmark_returns).cumprod()
        benchmark_max_dd = ((benchmark_cumulative / benchmark_cumulative.cummax()) - 1).min() * 100
        
        metrics = {
            'period_days': len(valid_data),
            'stock_total_return': stock_total_return,
            'benchmark_total_return': benchmark_total_return,
            'excess_return': stock_total_return - benchmark_total_return,
            'stock_volatility': stock_volatility,
            'benchmark_volatility': benchmark_volatility,
            'stock_sharpe': stock_sharpe,
            'benchmark_sharpe': benchmark_sharpe,
            'beta': beta,
            'alpha': alpha,
            'correlation': correlation,
            'stock_max_drawdown': stock_max_dd,
            'benchmark_max_drawdown': benchmark_max_dd,
            'stock_ticker': stock_ticker,
            'benchmark_ticker': benchmark_ticker
        }
        
        return metrics, merged
    
    except Exception as e:
        return None, f"Error calculating metrics: {str(e)}"

def create_comparison_chart(merged_data, stock_ticker, benchmark_ticker):
    """Create comparison price chart"""
    # Normalize to 100 at start
    stock_normalized = (merged_data['Close_stock'] / merged_data['Close_stock'].iloc[0]) * 100
    benchmark_normalized = (merged_data['Close_benchmark'] / merged_data['Close_benchmark'].iloc[0]) * 100
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=merged_data['Date'],
        y=stock_normalized,
        name=f'{stock_ticker}',
        line=dict(color='blue', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=merged_data['Date'],
        y=benchmark_normalized,
        name=f'{benchmark_ticker}',
        line=dict(color='red', width=2)
    ))
    
    fig.update_layout(
        title=f'{stock_ticker} vs {benchmark_ticker} Performance (Normalized to 100)',
        xaxis_title='Date',
        yaxis_title='Normalized Price',
        height=500,
        hovermode='x unified'
    )
    
    return fig

def create_returns_comparison_chart(merged_data, stock_ticker, benchmark_ticker):
    """Create rolling returns comparison chart"""
    # Calculate 30-day rolling returns
    stock_rolling = merged_data['Daily_Return_stock'].rolling(30).mean() * 100 * 30
    benchmark_rolling = merged_data['Daily_Return_benchmark'].rolling(30).mean() * 100 * 30
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=merged_data['Date'],
        y=stock_rolling,
        name=f'{stock_ticker} (30-day)',
        line=dict(color='blue')
    ))
    
    fig.add_trace(go.Scatter(
        x=merged_data['Date'],
        y=benchmark_rolling,
        name=f'{benchmark_ticker} (30-day)',
        line=dict(color='red')
    ))
    
    fig.update_layout(
        title=f'30-Day Rolling Returns: {stock_ticker} vs {benchmark_ticker}',
        xaxis_title='Date',
        yaxis_title='30-Day Return (%)',
        height=400
    )
    
    return fig

def create_correlation_rolling_chart(merged_data, stock_ticker, benchmark_ticker):
    """Create rolling correlation chart"""
    rolling_corr = merged_data['Daily_Return_stock'].rolling(60).corr(merged_data['Daily_Return_benchmark'])
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=merged_data['Date'],
        y=rolling_corr,
        name=f'60-day Correlation',
        line=dict(color='purple', width=2),
        fill='tonexty'
    ))
    
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", annotation_text="Moderate Correlation (0.5)")
    fig.add_hline(y=0.8, line_dash="dash", line_color="orange", annotation_text="High Correlation (0.8)")
    
    fig.update_layout(
        title=f'Rolling 60-Day Correlation: {stock_ticker} vs {benchmark_ticker}',
        xaxis_title='Date',
        yaxis_title='Correlation Coefficient',
        height=400,
        yaxis=dict(range=[-1, 1])
    )
    
    return fig

def explain_comparison_metrics():
    """Return explanation of comparison metrics"""
    return """
    📊 **Understanding Benchmark Comparison Metrics:**
    
    • **Total Return**: Overall percentage gain/loss over the period
    • **Excess Return**: How much better (or worse) your stock performed vs benchmark
    • **Beta**: Sensitivity to market movements (1.0 = moves with market, >1.0 = more volatile)
    • **Alpha**: Risk-adjusted excess return (positive = outperforming after accounting for risk)
    • **Sharpe Ratio**: Return per unit of risk (higher is better)
    • **Correlation**: How closely prices move together (0-1, where 1 = perfect correlation)
    • **Max Drawdown**: Largest peak-to-trough decline (lower is better)
    • **Volatility**: Annualized standard deviation of returns (risk measure)
    
    **Investment Insights:**
    - **High Beta (>1.2)**: Stock amplifies market movements, higher risk/reward
    - **Positive Alpha**: Stock generates excess returns relative to risk taken
    - **High Correlation (>0.8)**: Stock closely follows benchmark trends
    - **Lower Max Drawdown**: Stock preserves capital better during downturns
    """

def create_overlaid_price_chart(merged_data, stock_ticker, benchmark_ticker):
    """Create overlaid price chart showing both stock and benchmark actual prices"""
    fig = go.Figure()
    
    # Add stock price line
    fig.add_trace(go.Scatter(
        x=merged_data['Date'],
        y=merged_data['Close_stock'],
        name=f'{stock_ticker} Price',
        line=dict(color='blue', width=2),
        yaxis='y'
    ))
    
    # Add benchmark price line on secondary y-axis (if prices are very different)
    price_ratio = merged_data['Close_stock'].mean() / merged_data['Close_benchmark'].mean()
    
    if 0.1 < price_ratio < 10:  # If prices are similar, use same axis
        fig.add_trace(go.Scatter(
            x=merged_data['Date'],
            y=merged_data['Close_benchmark'],
            name=f'{benchmark_ticker} Price',
            line=dict(color='red', width=2),
            yaxis='y'
        ))
        
        fig.update_layout(
            title=f'{stock_ticker} vs {benchmark_ticker} Price History',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            height=500,
            hovermode='x unified'
        )
    else:  # Use dual y-axis for very different price ranges
        fig.add_trace(go.Scatter(
            x=merged_data['Date'],
            y=merged_data['Close_benchmark'],
            name=f'{benchmark_ticker} Price',
            line=dict(color='red', width=2),
            yaxis='y2'
        ))
        
        fig.update_layout(
            title=f'{stock_ticker} vs {benchmark_ticker} Price History (Dual Scale)',
            xaxis_title='Date',
            yaxis=dict(
                title=f'{stock_ticker} Price ($)',
                side='left'
            ),
            yaxis2=dict(
                title=f'{benchmark_ticker} Price ($)',
                side='right',
                overlaying='y'
            ),
            height=500,
            hovermode='x unified'
        )
    
    return fig

def create_percentage_change_chart(merged_data, stock_ticker, benchmark_ticker):
    """Create percentage change comparison chart from starting point"""
    # Calculate cumulative percentage change from first day
    stock_pct_change = ((merged_data['Close_stock'] / merged_data['Close_stock'].iloc[0]) - 1) * 100
    benchmark_pct_change = ((merged_data['Close_benchmark'] / merged_data['Close_benchmark'].iloc[0]) - 1) * 100
    
    fig = go.Figure()
    
    # Add stock percentage change
    fig.add_trace(go.Scatter(
        x=merged_data['Date'],
        y=stock_pct_change,
        name=f'{stock_ticker}',
        line=dict(color='blue', width=2)
    ))
    
    # Add benchmark percentage change
    fig.add_trace(go.Scatter(
        x=merged_data['Date'],
        y=benchmark_pct_change,
        name=f'{benchmark_ticker}',
        line=dict(color='red', width=2)
    ))
    
    # Add zero line for reference
    fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Starting Point (0%)")
    
    fig.update_layout(
        title=f'{stock_ticker} vs {benchmark_ticker} - Percentage Change from Start',
        xaxis_title='Date',
        yaxis_title='Percentage Change (%)',
        height=500,
        hovermode='x unified',
        yaxis=dict(tickformat='.1f')
    )
    
    return fig

def main():
    # Initialize session state for uploaded data and UI state
    if 'uploaded_data' not in st.session_state:
        st.session_state.uploaded_data = None
    if 'current_ticker' not in st.session_state:
        st.session_state.current_ticker = "SPY"
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = 0  # Default to first tab
    

    
    # Dynamic header with current ticker
    st.markdown(f'<h1 class="main-header">📈 Finance Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Dashboard Controls")
    
    # Cache management
    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()
    
    # Reset to default SPY data
    if st.sidebar.button("🏠 Reset to Default (SPY)"):
        st.session_state.uploaded_data = None
        st.session_state.current_ticker = "SPY"
        st.cache_data.clear()
        st.rerun()
    
    st.sidebar.markdown("---")
    
    # File upload section - always visible
    st.sidebar.subheader("📁 Upload Your Data")
    
    uploaded_file = st.sidebar.file_uploader(
        "Choose a CSV file",
        type="csv",
        help="Upload a CSV file with stock data. Required columns: Date, Open, High, Low, Close, Volume",
        key="csv_uploader"
    )
    
    ticker_name = st.sidebar.text_input(
        "Ticker Symbol",
        value=st.session_state.current_ticker if st.session_state.current_ticker != "SPY" else "",
        max_chars=6,
        help="Enter the ticker symbol (e.g., AAPL, NVDA, GOOGL). Must be 1-6 letters only."
    ).upper()
    
    st.sidebar.caption("💡 **Tips:** Keep CSV files under 10MB. Most daily stock data should be much smaller than this.")
    
    # Process uploaded file
    use_uploaded_data = False
    if uploaded_file is not None:
        try:
            # Read uploaded CSV
            uploaded_df = pd.read_csv(uploaded_file)
            
            # Validate ticker format first
            ticker_valid, ticker_error = validate_ticker_format(ticker_name)
            
            if not ticker_valid:
                st.sidebar.error("**Upload Failed**")
                st.sidebar.error(ticker_error)
            else:
                # Validate data
                is_valid, errors = validate_uploaded_data(uploaded_df)
                
                if is_valid:
                    # Store in session state
                    st.session_state.uploaded_data = uploaded_df
                    st.session_state.current_ticker = ticker_name
                    use_uploaded_data = True
                    st.sidebar.success(f"✅ {ticker_name} data loaded successfully!")
                    st.sidebar.info(f"📊 {len(uploaded_df)} trading days loaded")
                else:
                    # Show validation errors
                    st.sidebar.error("**Upload Failed**")
                    for error in errors:
                        st.sidebar.error(error)
                    st.sidebar.markdown("**💡 Tips:**")
                    st.sidebar.markdown("- Ensure your CSV has headers: Date, Open, High, Low, Close, Volume")
                    st.sidebar.markdown("- Date format: MM/DD/YYYY or YYYY-MM-DD")
                    st.sidebar.markdown("- Numbers can have quotes or commas (we'll clean them)")
                
        except Exception as e:
            st.sidebar.error(f"❌ **Error reading file**: {str(e)}")
            st.sidebar.markdown("Please ensure your file is a valid CSV format.")
    
    # Load data (either uploaded or default SPY)
    if use_uploaded_data:
        df = load_data(use_uploaded=True, uploaded_data=st.session_state.uploaded_data)
    elif st.session_state.uploaded_data is not None:
        # Use previously uploaded data (from Advanced Data or file upload)
        df = load_data(use_uploaded=True, uploaded_data=st.session_state.uploaded_data)
        use_uploaded_data = True
        # Don't reset current_ticker here - it should already be set correctly
    else:
        # Use default SPY data only if no uploaded data exists
        df = load_data()
        st.session_state.current_ticker = "SPY"
    
    # Check if data loaded successfully
    if df.empty:
        st.error("Failed to load data. Please check the data file and try again.")
        return
    
    # Create tab navigation with persistent state
    tab_options = ["📊 Data Overview", "📈 Visualizations", "📚 Benchmark Comparison", "🔍 Advanced Data", "🔮 ARIMA Analysis", "🧠 LSTM Analysis"]
    
    # Custom tab selector that maintains state
    selected_tab = st.selectbox(
        "📋 **Navigate to:**",
        options=tab_options,
        index=st.session_state.active_tab,
        key="tab_selector",
        help="Select which analysis you want to view. This selection will persist as you interact with the dashboard!"
    )
    
    # Update session state when tab changes
    if selected_tab:
        st.session_state.active_tab = tab_options.index(selected_tab)
    
    st.markdown("---")  # Visual separator
    
    # Display content based on selected tab
    if selected_tab == "📊 Data Overview":
        st.header("Data Overview")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Current Price", f"${df['Close'].iloc[-1]:.2f}", 
                     f"{df['Daily_Return'].iloc[-1]:.2%}")
        
        with col2:
            st.metric("52W High", f"${df['High'].max():.2f}")
        
        with col3:
            st.metric("52W Low", f"${df['Low'].min():.2f}")
        
        with col4:
            st.metric("Avg Volume", f"{df['Volume'].mean():,.0f}")
        
        # Data table
        st.subheader("Recent Data")
        st.dataframe(df.tail(10), use_container_width=True)
        
        # Basic statistics
        st.subheader("Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)
    
    elif selected_tab == "📈 Visualizations":
        st.header("Stock Visualizations")
        
        # Candlestick chart
        st.subheader("Price Chart")
        candlestick_fig = create_candlestick_chart(df, st.session_state.current_ticker)
        st.plotly_chart(candlestick_fig, use_container_width=True)
        
        # Volume chart
        st.subheader("Volume Chart")
        volume_fig = create_volume_chart(df, st.session_state.current_ticker)
        st.plotly_chart(volume_fig, use_container_width=True)
        
        # Correlation heatmap
        st.subheader("Correlation Analysis")
        corr_fig = create_correlation_heatmap(df, st.session_state.current_ticker)
        st.pyplot(corr_fig)
        
        # Pairplot
        st.subheader("Pairplot Analysis")
        pair_fig = create_pairplot(df, st.session_state.current_ticker)
        st.pyplot(pair_fig)
    
    elif selected_tab == "📚 Benchmark Comparison":
        st.header("📚 Benchmark Comparison")
        
        # Data source indicator
        if st.session_state.uploaded_data is not None:
            if st.session_state.current_ticker != "SPY":
                st.success(f"📊 **Analyzing**: {st.session_state.current_ticker} ({len(df)} data points)")
            else:
                st.info(f"📊 **Analyzing**: {st.session_state.current_ticker} ({len(df)} data points)")
        else:
            st.info(f"📊 **Analyzing**: SPY ({len(df)} data points)")
        
        st.info("💡 **Compare your stock's performance against major market benchmarks** to understand relative risk and return characteristics.")
        
        # Stock ticker input
        st.subheader("📊 Select Stock to Analyze")
        col_ticker1, col_ticker2 = st.columns([3, 1])
        
        with col_ticker1:
            benchmark_stock_ticker = st.text_input(
                "Stock Ticker for Comparison",
                value=st.session_state.current_ticker if st.session_state.current_ticker else "SPY",
                max_chars=10,
                help="Enter any stock ticker (e.g., AAPL, GOOGL, TSLA). This will be compared against the benchmark.",
                key="benchmark_stock_ticker"
            ).upper()
        
        with col_ticker2:
            use_current_data = st.button(
                "📊 Use Current Data",
                help="Use the data currently loaded in the dashboard",
                key="use_current_data_benchmark"
            )
        
        # If user clicked to use current data, update the ticker
        if use_current_data:
            benchmark_stock_ticker = st.session_state.current_ticker
            st.success(f"Using current data: {st.session_state.current_ticker}")
        
        # Benchmark selection
        st.subheader("📈 Select Benchmark")
        col_bench1, col_bench2 = st.columns([2, 1])
        
        with col_bench1:
            benchmarks = get_available_benchmarks()
            
            # Default to S&P 500 if current ticker is not SPY, otherwise use NASDAQ
            default_benchmark = '^GSPC' if benchmark_stock_ticker != 'SPY' else '^IXIC'
            
            selected_benchmark = st.selectbox(
                "Select Benchmark for Comparison",
                options=list(benchmarks.keys()),
                format_func=lambda x: f"{x} - {benchmarks[x]}",
                index=list(benchmarks.keys()).index(default_benchmark),
                key="benchmark_selector",
                help="Choose a benchmark to compare your stock against. S&P 500 is most common for large-cap stocks."
            )
            
            comparison_period = st.selectbox(
                "Comparison Period",
                ["1 Year", "2 Years", "3 Years", "5 Years", "Max Available"],
                index=1,  # Default to 2 years
                key="comparison_period",
                help="Select how far back to compare performance. Longer periods give better statistical significance."
            )
            
            # Custom date range section
            st.write("**📅 Or specify a custom date range:**")
            col_start, col_end = st.columns(2)
            with col_start:
                custom_start_date = st.date_input(
                    "Start Date",
                    value=datetime.now().date() - timedelta(days=365*2),  # Default 2 years
                    max_value=datetime.now().date(),
                    help="Select start date for comparison (overrides preset period)",
                    key="benchmark_custom_start_date"
                )
            
            with col_end:
                custom_end_date = st.date_input(
                    "End Date",
                    value=datetime.now().date(),
                    max_value=datetime.now().date(),
                    help="Select end date for comparison (overrides preset period)",
                    key="benchmark_custom_end_date"
                )
        
        with col_bench2:
            st.write("**Selected Benchmark:**")
            st.write(f"**{selected_benchmark}**")
            st.write(f"*{benchmarks[selected_benchmark]}*")
            
            # Warning for Treasury rates and Federal Funds Rate
            if selected_benchmark in ['^IRX', '^TNX']:
                st.warning("⚠️ **Treasury rates** may have limited data availability and different trading schedules than stocks. Consider using shorter periods (1-2 years) for better results.")
            elif selected_benchmark == 'FED_FUNDS':
                st.info("🏛️ **Federal Funds Rate** data comes directly from Alpha Vantage and provides official Fed policy rates. Requires Alpha Vantage API key.")
            
            # Period mapping
            period_map = {
                "1 Year": 365,
                "2 Years": 730, 
                "3 Years": 1095,
                "5 Years": 1825,
                "Max Available": None
            }
            
            # Show period based on priority
            if custom_start_date and custom_end_date:
                st.write(f"**Period**: {custom_start_date} to {custom_end_date}")
                st.write("📅 *Using custom date range*")
            elif period_map[comparison_period]:
                comparison_start = datetime.now().date() - timedelta(days=period_map[comparison_period])
                comparison_end = datetime.now().date()
                st.write(f"**Period**: {comparison_start} to {comparison_end}")
                st.write("📊 *Using preset period*")
            else:
                st.write("**Period**: Maximum available data")
                st.write("🔄 *Using all available data*")
        
        if st.button("🔍 Run Benchmark Comparison", key="run_benchmark_comparison", type="primary"):
            # Validate ticker input
            ticker_valid, ticker_error = validate_ticker_format(benchmark_stock_ticker)
            if not ticker_valid:
                st.error(ticker_error)
                return
            
            with st.spinner(f"Fetching data for {benchmark_stock_ticker} and {benchmarks[selected_benchmark]}..."):
                
                # Determine date range for comparison - prioritize custom dates
                if custom_start_date and custom_end_date:
                    # Custom dates provided - validate them
                    date_valid, date_error = validate_date_range(custom_start_date, custom_end_date)
                    if not date_valid:
                        st.error(f"Invalid custom date range: {date_error}")
                        return
                    start_date = custom_start_date
                    end_date = custom_end_date
                    st.info(f"📅 **Using custom date range**: {start_date} to {end_date}")
                elif period_map[comparison_period]:
                    # Use preset period
                    start_date = datetime.now().date() - timedelta(days=period_map[comparison_period])
                    end_date = datetime.now().date()
                else:
                    # Use the stock data date range for "Max Available"
                    start_date = df['Date'].min().date()
                    end_date = df['Date'].max().date()
                
                # Fetch stock data (if not using current data)
                if benchmark_stock_ticker == st.session_state.current_ticker and use_current_data:
                    stock_data = df
                else:
                    stock_data, stock_error = fetch_yfinance_data(benchmark_stock_ticker, start_date, end_date)
                    if stock_error:
                        st.error(f"Failed to fetch stock data: {stock_error}")
                        return
                
                # Fetch benchmark data - handle special case of Federal Funds Rate
                if selected_benchmark == 'FED_FUNDS':
                    # Fetch Federal Funds Rate from Alpha Vantage
                    api_keys = get_api_keys()
                    if not api_keys['alpha_vantage']:
                        st.error("🔑 **Alpha Vantage API key required for Federal Funds Rate data**. Please add your API key to `.streamlit/secrets.toml`")
                        st.info("📚 **Get your free Alpha Vantage API key**: https://www.alphavantage.co/support/#api-key")
                        return
                    
                    benchmark_data, benchmark_error = fetch_fed_funds_rate_alpha_vantage(api_keys['alpha_vantage'], start_date, end_date)
                else:
                    benchmark_data, benchmark_error = fetch_benchmark_data(selected_benchmark, start_date, end_date)
                
                if benchmark_error:
                    st.error(f"Failed to fetch benchmark data: {benchmark_error}")
                    return
                
                # Calculate comparison metrics
                metrics_result = calculate_comparison_metrics(
                    stock_data, benchmark_data, 
                    benchmark_stock_ticker, selected_benchmark
                )
                
                if metrics_result[0] is None:
                    st.error(f"Comparison failed: {metrics_result[1]}")
                    return
                
                metrics, merged_data = metrics_result
                
                # Display comparison summary
                st.success(f"✅ **Comparison complete!** Analyzed {metrics['period_days']} overlapping trading days")
                
                # Key metrics display
                st.subheader("📊 Performance Comparison Summary")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    stock_return = metrics['stock_total_return']
                    delta_color = "normal" if stock_return >= 0 else "inverse"
                    st.metric(
                        f"{metrics['stock_ticker']} Total Return", 
                        f"{stock_return:.2f}%",
                        delta_color=delta_color
                    )
                
                with col2:
                    benchmark_return = metrics['benchmark_total_return']
                    delta_color = "normal" if benchmark_return >= 0 else "inverse"
                    st.metric(
                        f"{metrics['benchmark_ticker']} Total Return", 
                        f"{benchmark_return:.2f}%",
                        delta_color=delta_color
                    )
                
                with col3:
                    excess_return = metrics['excess_return']
                    delta_color = "normal" if excess_return >= 0 else "inverse"
                    st.metric(
                        "Excess Return", 
                        f"{excess_return:.2f}%",
                        help="How much better (or worse) your stock performed vs benchmark",
                        delta_color=delta_color
                    )
                
                with col4:
                    correlation = metrics['correlation']
                    st.metric(
                        "Correlation", 
                        f"{correlation:.3f}",
                        help="How closely the stock moves with the benchmark (0-1)"
                    )
                
                # Performance interpretation
                if excess_return > 5:
                    st.success(f"🟢 **Strong Outperformance**: {metrics['stock_ticker']} significantly outperformed {selected_benchmark}")
                elif excess_return > 0:
                    st.info(f"🟡 **Modest Outperformance**: {metrics['stock_ticker']} slightly outperformed {selected_benchmark}")
                elif excess_return > -5:
                    st.warning(f"🟠 **Modest Underperformance**: {metrics['stock_ticker']} slightly underperformed {selected_benchmark}")
                else:
                    st.error(f"🔴 **Significant Underperformance**: {metrics['stock_ticker']} significantly underperformed {selected_benchmark}")
                
                # Detailed metrics
                st.subheader("📈 Detailed Risk & Return Analysis")
                
                # Risk metrics
                col_risk1, col_risk2 = st.columns(2)
                
                with col_risk1:
                    st.write("**📊 Return Metrics:**")
                    st.write(f"• **{metrics['stock_ticker']} Volatility**: {metrics['stock_volatility']:.2f}%")
                    st.write(f"• **{metrics['benchmark_ticker']} Volatility**: {metrics['benchmark_volatility']:.2f}%")
                    st.write(f"• **{metrics['stock_ticker']} Sharpe Ratio**: {metrics['stock_sharpe']:.3f}")
                    st.write(f"• **{metrics['benchmark_ticker']} Sharpe Ratio**: {metrics['benchmark_sharpe']:.3f}")
                
                with col_risk2:
                    st.write("**⚖️ Risk-Adjusted Metrics:**")
                    st.write(f"• **Beta**: {metrics['beta']:.3f}")
                    if metrics['beta'] > 1.2:
                        st.write("  → 🔴 High Beta (Amplifies market moves)")
                    elif metrics['beta'] > 0.8:
                        st.write("  → 🟡 Moderate Beta (Follows market)")
                    else:
                        st.write("  → 🟢 Low Beta (Less volatile than market)")
                    
                    st.write(f"• **Alpha**: {metrics['alpha']:.2f}%")
                    if metrics['alpha'] > 2:
                        st.write("  → 🟢 Positive Alpha (Excess returns)")
                    elif metrics['alpha'] > -2:
                        st.write("  → 🟡 Neutral Alpha")
                    else:
                        st.write("  → 🔴 Negative Alpha (Underperforming)")
                
                # Drawdown analysis
                st.write("**📉 Maximum Drawdown Analysis:**")
                st.write(f"• **{metrics['stock_ticker']} Max Drawdown**: {metrics['stock_max_drawdown']:.2f}%")
                st.write(f"• **{metrics['benchmark_ticker']} Max Drawdown**: {metrics['benchmark_max_drawdown']:.2f}%")
                
                # Visualizations
                st.subheader("📈 Comparison Charts")
                
                # Percentage change chart (no scaling needed)
                st.subheader("📊 Percentage Change Comparison")
                pct_change_chart = create_percentage_change_chart(merged_data, metrics['stock_ticker'], metrics['benchmark_ticker'])
                st.plotly_chart(pct_change_chart, use_container_width=True)
                
                # Overlaid price history chart
                st.subheader("💰 Price History Overlay")
                overlaid_chart = create_overlaid_price_chart(merged_data, metrics['stock_ticker'], metrics['benchmark_ticker'])
                st.plotly_chart(overlaid_chart, use_container_width=True)
                
                # Normalized performance chart
                st.subheader("📊 Normalized Performance (Base 100)")
                comparison_fig = create_comparison_chart(merged_data, metrics['stock_ticker'], metrics['benchmark_ticker'])
                st.plotly_chart(comparison_fig, use_container_width=True)
                
                # Rolling returns comparison
                returns_fig = create_returns_comparison_chart(merged_data, metrics['stock_ticker'], metrics['benchmark_ticker'])
                st.plotly_chart(returns_fig, use_container_width=True)
                
                # Rolling correlation
                correlation_fig = create_correlation_rolling_chart(merged_data, metrics['stock_ticker'], metrics['benchmark_ticker'])
                st.plotly_chart(correlation_fig, use_container_width=True)
                
                # Performance summary table
                st.subheader("📋 Complete Metrics Summary")
                
                summary_data = {
                    'Metric': [
                        'Total Return (%)',
                        'Annualized Volatility (%)', 
                        'Sharpe Ratio',
                        'Maximum Drawdown (%)',
                        'Beta',
                        'Alpha (%)',
                        'Correlation'
                    ],
                    f'{metrics["stock_ticker"]}': [
                        f"{metrics['stock_total_return']:.2f}",
                        f"{metrics['stock_volatility']:.2f}",
                        f"{metrics['stock_sharpe']:.3f}",
                        f"{metrics['stock_max_drawdown']:.2f}",
                        f"{metrics['beta']:.3f}",
                        f"{metrics['alpha']:.2f}",
                        f"{metrics['correlation']:.3f}"
                    ],
                    f'{metrics["benchmark_ticker"]}': [
                        f"{metrics['benchmark_total_return']:.2f}",
                        f"{metrics['benchmark_volatility']:.2f}",
                        f"{metrics['benchmark_sharpe']:.3f}",
                        f"{metrics['benchmark_max_drawdown']:.2f}",
                        "1.000",  # Benchmark beta is always 1
                        "0.00",   # Benchmark alpha is always 0
                        "1.000"   # Benchmark correlation with itself is 1
                    ]
                }
                
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True)
                
                # Investment insights
                st.subheader("💡 Investment Insights")
                
                insights = []
                
                # Beta insights
                if metrics['beta'] > 1.5:
                    insights.append(f"🔴 **High Risk Profile**: {metrics['stock_ticker']} is {metrics['beta']:.1f}x more volatile than {selected_benchmark}")
                elif metrics['beta'] > 1.2:
                    insights.append(f"🟠 **Moderate-High Risk**: {metrics['stock_ticker']} amplifies market movements by {(metrics['beta']-1)*100:.0f}%")
                elif metrics['beta'] < 0.8:
                    insights.append(f"🟢 **Defensive Stock**: {metrics['stock_ticker']} is less volatile than the market (Beta: {metrics['beta']:.2f})")
                
                # Alpha insights
                if metrics['alpha'] > 5:
                    insights.append(f"🟢 **Strong Alpha Generation**: {metrics['stock_ticker']} generates {metrics['alpha']:.1f}% excess returns annually")
                elif metrics['alpha'] < -5:
                    insights.append(f"🔴 **Poor Risk-Adjusted Returns**: {metrics['stock_ticker']} underperforms by {abs(metrics['alpha']):.1f}% annually")
                
                # Correlation insights
                if metrics['correlation'] > 0.9:
                    insights.append(f"📈 **High Market Correlation**: {metrics['stock_ticker']} closely tracks {selected_benchmark} movements")
                elif metrics['correlation'] < 0.3:
                    insights.append(f"📊 **Low Market Correlation**: {metrics['stock_ticker']} provides diversification benefits")
                
                # Sharpe ratio insights
                if metrics['stock_sharpe'] > metrics['benchmark_sharpe'] + 0.2:
                    insights.append(f"⭐ **Superior Risk-Adjusted Returns**: Better Sharpe ratio than benchmark")
                elif metrics['stock_sharpe'] < metrics['benchmark_sharpe'] - 0.2:
                    insights.append(f"⚠️ **Inferior Risk-Adjusted Returns**: Lower Sharpe ratio than benchmark")
                
                for insight in insights:
                    st.write(f"• {insight}")
                
                if not insights:
                    st.write("• 🟡 **Balanced Profile**: Stock shows similar risk-return characteristics to the benchmark")
                
                # Educational content
                with st.expander("❓ Understanding These Metrics"):
                    st.markdown(explain_comparison_metrics())
    
    elif selected_tab == "🔍 Advanced Data":
        st.header("🔍 Advanced Data Collection")
        
        st.info("💡 **Professional Analysis**: Collect extended historical data with advanced technical indicators for institutional-quality analysis.")
        
        # Advanced data collection interface
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📊 Data Collection Parameters")
            
            # Ticker input with validation
            advanced_ticker = st.text_input(
                "Ticker Symbol", 
                value="AAPL",
                max_chars=10,
                help="Enter any valid stock ticker (e.g., AAPL, GOOGL, TSLA, BRK-A)",
                key="advanced_ticker"
            ).upper()
            
            # Date range selection
            col_start, col_end = st.columns(2)
            with col_start:
                start_date = st.date_input(
                    "Start Date",
                    value=datetime.now().date() - timedelta(days=365*2),  # Default 2 years
                    max_value=datetime.now().date(),
                    help="Select how far back to collect data",
                    key="advanced_start_date"
                )
            
            with col_end:
                end_date = st.date_input(
                    "End Date",
                    value=datetime.now().date(),
                    max_value=datetime.now().date(),
                    help="Select the end date for data collection",
                    key="advanced_end_date"
                )
            
            # Data source options
            st.subheader("📡 Data Sources & Indicators")
            
            col_opt1, col_opt2 = st.columns(2)
            with col_opt1:
                include_technical = st.checkbox("Technical Indicators", value=True, help="RSI, MACD, Bollinger Bands, Moving Averages")
                include_volume = st.checkbox("Volume Analysis", value=True, help="Volume-based indicators and analysis")
            
            with col_opt2:
                include_advanced = st.checkbox("Advanced Indicators", value=True, help="Additional indicators for professional analysis")
                export_csv = st.checkbox("Generate CSV Export", value=True, help="Create downloadable CSV file")
            
            # Phase 2: API Data Sources
            st.subheader("🔌 Professional API Data Sources")
            
            # Check API key availability
            api_status = check_api_keys_available()
            
            col_api1, col_api2 = st.columns(2)
            with col_api1:
                include_fundamentals = st.checkbox(
                    "Fundamental Analysis (FMP)", 
                    value=api_status['fmp_available'],
                    disabled=not api_status['fmp_available'],
                    help="P/E ratios, financial statements, company fundamentals"
                )
                if not api_status['fmp_available']:
                    st.caption("⚠️ FMP API key not configured")
            
            with col_api2:
                include_alpha_indicators = st.checkbox(
                    "Enhanced Indicators (Alpha Vantage)", 
                    value=api_status['alpha_available'],
                    disabled=not api_status['alpha_available'],
                    help="Professional-grade RSI, MACD, and additional indicators"
                )
                if not api_status['alpha_available']:
                    st.caption("⚠️ Alpha Vantage API key not configured")
        
        with col2:
            st.subheader("ℹ️ Collection Info")
            
            # API Status Display
            st.write("**🔌 API Status:**")
            fmp_status = "🟢 Active" if api_status['fmp_available'] else "🔴 Not configured"
            alpha_status = "🟢 Active" if api_status['alpha_available'] else "🔴 Not configured"
            st.write(f"• FMP: {fmp_status}")
            st.write(f"• Alpha Vantage: {alpha_status}")
            
            if not api_status['fmp_available'] or not api_status['alpha_available']:
                with st.expander("🔧 How to Configure API Keys"):
                    st.markdown("""
                    **For Streamlit Cloud:**
                    1. Go to your app settings
                    2. Add secrets in the format:
                    ```
                    FMP_API_KEY = "your_fmp_key"
                    ALPHA_VANTAGE_API_KEY = "your_alpha_key"
                    ```
                    
                    **For Local Development:**
                    1. Create a `.env` file in your project root
                    2. Add your API keys:
                    ```
                    FMP_API_KEY=your_fmp_key
                    ALPHA_VANTAGE_API_KEY=your_alpha_key
                    ```
                    """)
            
            st.write("---")
            
            # Show ticker info if valid
            if advanced_ticker and len(advanced_ticker) > 0:
                with st.spinner("Getting ticker info..."):
                    ticker_info = get_ticker_info(advanced_ticker)
                    st.write(f"**Company**: {ticker_info['name']}")
                    st.write(f"**Sector**: {ticker_info['sector']}")
                    st.write(f"**Industry**: {ticker_info['industry']}")
                    if ticker_info['market_cap'] != 'N/A':
                        try:
                            market_cap = ticker_info['market_cap']
                            st.write(f"**Market Cap**: ${market_cap:,.0f}")
                        except:
                            st.write(f"**Market Cap**: {ticker_info['market_cap']}")
            
            # Show date range info
            if start_date and end_date:
                date_range_days = (end_date - start_date).days
                st.write(f"**Date Range**: {date_range_days} days")
                st.write(f"**Period**: {date_range_days/365:.1f} years")
        
        # Data collection button
        if st.button("🚀 Collect Advanced Data", key="collect_advanced_data", type="primary"):
            # Validate inputs
            ticker_valid, ticker_error = validate_ticker_format(advanced_ticker)
            if not ticker_valid:
                st.error(ticker_error)
                return
            
            date_valid, date_error = validate_date_range(start_date, end_date)
            if not date_valid:
                st.error(date_error)
                return
            
            # Collect data
            with st.spinner(f"Collecting data for {advanced_ticker}..."):
                # Fetch yfinance data
                yf_data, yf_error = fetch_yfinance_data(advanced_ticker, start_date, end_date)
                
                if yf_error:
                    st.error(f"Data collection failed: {yf_error}")
                    return
                
                if yf_data is None or yf_data.empty:
                    st.error(f"No data available for {advanced_ticker} in the specified date range.")
                    return
                
                # Add advanced indicators if requested
                if include_advanced:
                    yf_data = calculate_advanced_indicators(yf_data)
                
                # Phase 2: Collect API data
                fundamental_data = None
                alpha_indicators = None
                
                if include_fundamentals and api_status['fmp_available']:
                    with st.spinner("Fetching fundamental data from FMP..."):
                        fundamental_data, fmp_error = fetch_fmp_fundamentals(advanced_ticker, api_status['keys']['fmp'])
                        if fmp_error:
                            st.warning(f"FMP API: {fmp_error}")
                
                if include_alpha_indicators and api_status['alpha_available']:
                    with st.spinner("Fetching enhanced indicators from Alpha Vantage... (this may take ~30 seconds due to rate limits)"):
                        alpha_indicators, alpha_error = fetch_alpha_vantage_indicators(advanced_ticker, api_status['keys']['alpha_vantage'])
                        if alpha_error:
                            st.warning(f"Alpha Vantage API: {alpha_error}")
                
                # Display success metrics
                st.success(f"✅ Successfully collected {len(yf_data)} trading days of data for {advanced_ticker}")
                
                # Show data summary
                col_summary1, col_summary2, col_summary3 = st.columns(3)
                with col_summary1:
                    st.metric("Data Points", f"{len(yf_data):,}")
                with col_summary2:
                    st.metric("Date Range", f"{(end_date - start_date).days} days")
                with col_summary3:
                    indicators_count = len(yf_data.columns)
                    if fundamental_data:
                        indicators_count += 4  # Categories of fundamental data
                    if alpha_indicators:
                        indicators_count += len(alpha_indicators)
                    st.metric("Total Indicators", f"{indicators_count} sources")
                
                # Phase 2: Display Fundamental Analysis
                if fundamental_data:
                    st.subheader("💼 Fundamental Analysis")
                    formatted_fundamentals = format_fundamental_data(fundamental_data)
                    
                    fund_col1, fund_col2 = st.columns(2)
                    
                    with fund_col1:
                        if 'Company Info' in formatted_fundamentals:
                            st.write("**📊 Company Information:**")
                            for key, value in formatted_fundamentals['Company Info'].items():
                                st.write(f"• {key}: {value}")
                        
                        if 'Valuation Ratios' in formatted_fundamentals:
                            st.write("**💰 Valuation Ratios:**")
                            for key, value in formatted_fundamentals['Valuation Ratios'].items():
                                st.write(f"• {key}: {value}")
                    
                    with fund_col2:
                        if 'Financial Health' in formatted_fundamentals:
                            st.write("**🏥 Financial Health:**")
                            for key, value in formatted_fundamentals['Financial Health'].items():
                                st.write(f"• {key}: {value}")
                        
                        if 'Latest Financials' in formatted_fundamentals:
                            st.write("**📈 Latest Financials:**")
                            for key, value in formatted_fundamentals['Latest Financials'].items():
                                st.write(f"• {key}: {value}")
                
                # Show recent data
                st.subheader("📋 Data Preview")
                st.dataframe(yf_data.head(10), use_container_width=True)
                
                # Create visualizations
                st.subheader("📈 Advanced Visualizations")
                
                if include_technical and 'RSI' in yf_data.columns:
                    # RSI chart
                    fig_rsi = go.Figure()
                    fig_rsi.add_trace(go.Scatter(
                        x=yf_data['Date'], 
                        y=yf_data['RSI'], 
                        name='RSI',
                        line=dict(color='purple')
                    ))
                    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
                    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
                    fig_rsi.update_layout(
                        title=f"{advanced_ticker} Relative Strength Index (RSI)",
                        yaxis_title="RSI",
                        height=400
                    )
                    st.plotly_chart(fig_rsi, use_container_width=True)
                
                if include_technical and 'MACD' in yf_data.columns:
                    # MACD chart
                    fig_macd = go.Figure()
                    fig_macd.add_trace(go.Scatter(x=yf_data['Date'], y=yf_data['MACD'], name='MACD', line=dict(color='blue')))
                    fig_macd.add_trace(go.Scatter(x=yf_data['Date'], y=yf_data['MACD_Signal'], name='Signal', line=dict(color='red')))
                    fig_macd.add_trace(go.Bar(x=yf_data['Date'], y=yf_data['MACD_Histogram'], name='Histogram', opacity=0.7))
                    fig_macd.update_layout(
                        title=f"{advanced_ticker} MACD Analysis",
                        yaxis_title="MACD",
                        height=400
                    )
                    st.plotly_chart(fig_macd, use_container_width=True)
                
                # Advanced statistics
                st.subheader("📊 Advanced Statistics")
                stats_col1, stats_col2 = st.columns(2)
                
                with stats_col1:
                    st.write("**Price Statistics:**")
                    st.write(f"• Maximum Price: ${yf_data['Close'].max():.2f}")
                    st.write(f"• Minimum Price: ${yf_data['Close'].min():.2f}")
                    st.write(f"• Average Price: ${yf_data['Close'].mean():.2f}")
                    st.write(f"• Price Volatility: {yf_data['Daily_Return'].std()*100:.2f}%")
                
                with stats_col2:
                    st.write("**Technical Analysis:**")
                    if 'RSI' in yf_data.columns:
                        current_rsi = yf_data['RSI'].iloc[-1]
                        st.write(f"• Current RSI: {current_rsi:.1f}")
                        if current_rsi > 70:
                            st.write("  → 🔴 Potentially Overbought")
                        elif current_rsi < 30:
                            st.write("  → 🟢 Potentially Oversold")
                        else:
                            st.write("  → 🟡 Neutral Territory")
                    
                    if 'MA_50' in yf_data.columns:
                        current_price = yf_data['Close'].iloc[-1]
                        ma_50 = yf_data['MA_50'].iloc[-1]
                        if not pd.isna(ma_50):
                            trend = "Above" if current_price > ma_50 else "Below"
                            st.write(f"• Price vs 50-day MA: {trend}")
                
                # CSV Export
                if export_csv:
                    st.subheader("💾 Data Export")
                    
                    # Create CSV
                    csv_data = yf_data.to_csv(index=False)
                    
                    # Download button
                    st.download_button(
                        label=f"📥 Download {advanced_ticker} Data (CSV)",
                        data=csv_data,
                        file_name=f"{advanced_ticker}_{start_date}_to_{end_date}_advanced.csv",
                        mime="text/csv",
                        help="Download the complete dataset with all indicators"
                    )
                    
                    st.info(f"📋 **CSV Contains**: {len(yf_data.columns)} columns including price data, technical indicators, and advanced metrics.")
                    
                    # Simple workflow instructions
                    st.markdown("---")
                    st.subheader("📈 Use This Data for ARIMA/LSTM Analysis")
                    st.info("**Simple 3-Step Process:**")
                    col_step1, col_step2, col_step3 = st.columns(3)
                    
                    with col_step1:
                        st.markdown("**1️⃣ Download CSV**")
                        st.markdown("Click the download button above")
                    
                    with col_step2:
                        st.markdown("**2️⃣ Upload File**")
                        st.markdown("Use the sidebar file uploader")
                    
                    with col_step3:
                        st.markdown("**3️⃣ Run Analysis**")
                        st.markdown("Go to ARIMA/LSTM tabs")
                    
                    # st.success("✅ **This method is 100% reliable** and avoids any session state issues!")
    
    elif selected_tab == "🔮 ARIMA Analysis":
        st.header("🔮 ARIMA Analysis")
        
        # Data source indicator
        if st.session_state.uploaded_data is not None:
            if st.session_state.current_ticker != "SPY":
                st.success(f"📊 **Using custom data**: {st.session_state.current_ticker} ({len(df)} data points)")
            else:
                st.info(f"📊 **Using uploaded data**: {st.session_state.current_ticker} ({len(df)} data points)")
        else:
            st.info(f"📊 **Using default data**: SPY ({len(df)} data points)")
            with st.expander("💡 Want to analyze different data?"):
                st.markdown("**3-Step Process to Analyze Any Stock:**")
                st.markdown("1. Go to the **'🔍 Advanced Data'** tab")
                st.markdown("2. Enter ticker, select dates, and click **'🚀 Collect Advanced Data'**")
                st.markdown("3. **Download the CSV** and **upload it** using the sidebar")
                st.markdown("4. Return here for analysis!")
                st.markdown("*This simple method is 100% reliable.*")
        
        st.info("💡 **Recommended starting point**: p=1, d=1, q=1 (selected by default) works well for most stocks including SPY.")
        
        # ARIMA parameters with help tooltips
        col1, col2, col3 = st.columns(3)
        with col1:
            p = st.selectbox(
                "AR (p) - AutoRegressive", 
                [0, 1, 2, 3], 
                index=1,
                key="arima_p",
                help="How much yesterday's price influences today's prediction. **p=1**: Good for stable stocks (blue chips, utilities). **p=2**: Better for volatile stocks (tech, growth). **p=3**: For highly unpredictable stocks. Start with p=1 for established companies."
            )
        with col2:
            d = st.selectbox(
                "I (d) - Integration", 
                [0, 1, 2], 
                index=1,
                key="arima_d",
                help="Removes price trends to focus on price changes. **d=1**: Standard for stock prices (recommended). **d=0**: Only for already stable price series. **d=2**: Rarely needed. Keep at d=1 for normal stock analysis."
            )
        with col3:
            q = st.selectbox(
                "MA (q) - Moving Average", 
                [0, 1, 2, 3], 
                index=1,
                key="arima_q",
                help="How much recent market 'shocks' affect predictions. **q=1**: Good for most stocks, captures immediate reactions. **q=2**: For stocks sensitive to news/events. **q=0**: For very predictable stocks. q=1 works well for SPY and major ETFs."
            )
        
        if st.button("Run ARIMA Analysis", key="run_arima"):
            with st.spinner("Training ARIMA model..."):
                ts_data = prepare_arima_data(df)
                model, forecast, conf_int = fit_arima_model(ts_data, order=(p, d, q))
                
                if model is not None:
                    # Calculate additional metrics for finance professionals
                    fitted_values = model.fittedvalues
                    
                    # Check if we have valid fitted values
                    if len(fitted_values) == 0:
                        st.error("ARIMA model produced no fitted values. Try different parameters.")
                        return
                    
                    # Align actual and fitted values to ensure same length
                    # ARIMA fitted values start from the first valid observation
                    start_idx = max(0, len(ts_data) - len(fitted_values))
                    actual_values = ts_data.iloc[start_idx:]
                    
                    # Ensure both arrays have exactly the same length
                    min_len = min(len(actual_values), len(fitted_values))
                    if min_len <= 0:
                        st.error("Cannot align actual and fitted values. Try different ARIMA parameters.")
                        return
                    
                    actual_values = actual_values.iloc[-min_len:]
                    fitted_values = fitted_values[-min_len:]
                    
                    # Calculate MAPE and R² with error handling
                    try:
                        mape = calculate_mape(actual_values.values, fitted_values)
                        r2 = r2_score(actual_values.values, fitted_values)
                    except Exception as e:
                        st.error(f"Error calculating metrics: {str(e)}")
                        return
                    
                    # Finance-friendly metrics display
                    st.subheader("📊 Model Performance for Trading Decisions")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(
                            "MAPE (Prediction Error)", 
                            f"{mape:.2f}%",
                            help="Lower is better. Under 5% is excellent for trading decisions."
                        )
                    with col2:
                        st.metric(
                            "R² (Predictive Power)", 
                            f"{r2*100:.1f}%",
                            help="Higher is better. Above 80% indicates strong predictive power."
                        )
                    
                    # Interpretation for finance professionals
                    if mape < 5:
                        st.success("🟢 **Excellent accuracy** - Suitable for short-term trading strategies")
                    elif mape < 10:
                        st.info("🟡 **Good accuracy** - Suitable for portfolio planning and medium-term decisions")
                    elif mape < 20:
                        st.warning("🟠 **Fair accuracy** - Use for long-term trend analysis only")
                    else:
                        st.error("🔴 **Poor accuracy** - Exercise caution, consider fundamental analysis")
                    
                    # Model summary (collapsed by default)
                    with st.expander("📋 Technical Model Details (Advanced Users)"):
                        st.text(str(model.summary()))
                    
                    # Forecast plot
                    st.subheader("7-Day Forecast")
                    
                    fig, ax = plt.subplots(figsize=(12, 6))
                    
                    # Plot historical data
                    ax.plot(ts_data.index, ts_data.values, label='Historical', color='blue')
                    
                    # Plot forecast
                    forecast_dates = pd.date_range(start=ts_data.index[-1] + pd.Timedelta(days=1), periods=7)
                    ax.plot(forecast_dates, forecast, label='Forecast', color='red', linestyle='--')
                    
                    # Plot confidence intervals
                    ax.fill_between(forecast_dates, conf_int.iloc[:, 0], conf_int.iloc[:, 1], 
                                   color='red', alpha=0.3, label='Confidence Interval')
                    
                    ax.legend()
                    ax.set_title(f'{st.session_state.current_ticker} ARIMA Forecast')
                    ax.set_xlabel('Date')
                    ax.set_ylabel('Price ($)')
                    plt.xticks(rotation=45)
                    
                    st.pyplot(fig)
                    
                    # Forecast values
                    st.subheader("Forecast Values")
                    forecast_df = pd.DataFrame({
                        'Date': forecast_dates,
                        'Forecast': forecast,
                        'Lower_CI': conf_int.iloc[:, 0],
                        'Upper_CI': conf_int.iloc[:, 1]
                    })
                    st.dataframe(forecast_df)
                    
                    # User-friendly explanation
                    with st.expander("❓ How to Interpret These Results"):
                        st.markdown(explain_arima_results())
    


    elif selected_tab == "🧠 LSTM Analysis":
        st.header("🧠 LSTM Analysis")
        
        # Data source indicator
        if st.session_state.uploaded_data is not None:
            if st.session_state.current_ticker != "SPY":
                st.success(f"📊 **Analyzing**: {st.session_state.current_ticker} ({len(df)} data points)")
            else:
                st.info(f"📊 **Analyzing**: {st.session_state.current_ticker} ({len(df)} data points)")
        else:
            st.info(f"📊 **Analyzing**: SPY ({len(df)} data points)")
        
        st.info("🧠 **AI Deep Learning**: Use neural networks to predict stock prices based on historical patterns.")
        
        # Stock ticker input section
        st.subheader("📊 Select Stock to Analyze")
        col_ticker1, col_ticker2 = st.columns([3, 1])
        
        with col_ticker1:
            lstm_stock_ticker = st.text_input(
                "Stock Ticker for LSTM Analysis",
                value=st.session_state.current_ticker if st.session_state.current_ticker else "SPY",
                max_chars=10,
                help="Enter any stock ticker (e.g., AAPL, GOOGL, TSLA). The AI will analyze this stock's price patterns.",
                key="lstm_stock_ticker"
            ).upper()
        
        with col_ticker2:
            use_current_data_lstm = st.button(
                "📊 Use Current Data",
                help="Use the data currently loaded in the dashboard",
                key="use_current_data_lstm"
            )
        
        # If user clicked to use current data, update the ticker
        if use_current_data_lstm:
            lstm_stock_ticker = st.session_state.current_ticker
            st.success(f"Using current data: {st.session_state.current_ticker}")
        
        # Date range selection for fetching new data
        st.subheader("📅 Analysis Period")
        col_date1, col_date2 = st.columns(2)
        
        with col_date1:
            lstm_start_date = st.date_input(
                "Start Date",
                value=datetime.now().date() - timedelta(days=365*2),  # Default 2 years
                max_value=datetime.now().date(),
                help="Select how far back to analyze data",
                key="lstm_start_date"
            )
        
        with col_date2:
            lstm_end_date = st.date_input(
                "End Date", 
                value=datetime.now().date(),
                max_value=datetime.now().date(),
                help="Select the end date for analysis",
                key="lstm_end_date"
            )
        
        st.info("💡 **Recommended starting point**: Sequence Length = 10 days (selected by default) works well for most stocks.")
        
        sequence_length = st.slider(
            "Sequence Length (days)", 
            5, 20, 10,
            key="lstm_sequence_length_tab6",
            help="How many previous trading days the AI model looks at to predict the next day. **5-7 days**: For highly volatile stocks that change quickly. **10-12 days**: Good for most stocks including SPY (captures ~2 weeks). **15-20 days**: For very stable stocks or longer-term patterns. More days = more context but slower predictions."
        )
        
        if st.button("🧠 Run LSTM Analysis", key="run_lstm_tab6", type="primary"):
            # Validate ticker input
            ticker_valid, ticker_error = validate_ticker_format(lstm_stock_ticker)
            if not ticker_valid:
                st.error(ticker_error)
                return
            
            # Validate date range
            date_valid, date_error = validate_date_range(lstm_start_date, lstm_end_date)
            if not date_valid:
                st.error(date_error)
                return
            
            with st.spinner(f"Fetching data and training LSTM model for {lstm_stock_ticker}..."):
                
                # Determine which data to use
                if lstm_stock_ticker == st.session_state.current_ticker and use_current_data_lstm:
                    # Use current data
                    analysis_data = df
                    st.success(f"Using current session data for {lstm_stock_ticker}")
                else:
                    # Fetch new data
                    analysis_data, fetch_error = fetch_yfinance_data(lstm_stock_ticker, lstm_start_date, lstm_end_date)
                    if fetch_error:
                        st.error(f"Failed to fetch data: {fetch_error}")
                        return
                    
                    if analysis_data is None or analysis_data.empty:
                        st.error(f"No data available for {lstm_stock_ticker} in the specified date range.")
                        return
                    
                    st.success(f"Successfully fetched {len(analysis_data)} days of data for {lstm_stock_ticker}")
                
                X, y, scaler = prepare_lstm_data(analysis_data, sequence_length)
                
                if len(X) > 0:
                    # Split data
                    train_size = int(len(X) * 0.8)
                    X_train, X_test = X[:train_size], X[train_size:]
                    y_train, y_test = y[:train_size], y[train_size:]
                    
                    # Create model and get device
                    model, device = create_lstm_model(sequence_length)
                    
                    # Show device info to user
                    if device.type == 'cuda':
                        st.success(f"🚀 Using NVIDIA GPU acceleration: {torch.cuda.get_device_name(0)}")
                    elif device.type == 'mps':
                        st.success("🍎 Using Apple Silicon GPU acceleration (MPS)")
                    else:
                        st.info("🖥️ Using CPU (consider enabling GPU for faster training)")
                    
                    # Convert to PyTorch tensors and move to device
                    X_train_tensor = torch.FloatTensor(X_train).to(device)
                    y_train_tensor = torch.FloatTensor(y_train.reshape(-1, 1)).to(device)
                    X_test_tensor = torch.FloatTensor(X_test).to(device)
                    y_test_tensor = torch.FloatTensor(y_test.reshape(-1, 1)).to(device)
                    
                    criterion = nn.MSELoss()
                    optimizer = optim.Adam(model.parameters(), lr=0.001)
                    
                    # Training with more epochs for better performance
                    model.train()
                    losses = []
                    epochs = 150  # Increased from 50 for better learning
                    
                    # Progress bar for training
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for epoch in range(epochs):
                        # Update progress
                        progress_bar.progress((epoch + 1) / epochs)
                        status_text.text(f'Training epoch {epoch + 1}/{epochs}...')
                        optimizer.zero_grad()
                        outputs = model(X_train_tensor)
                        loss = criterion(outputs, y_train_tensor)
                        loss.backward()
                        optimizer.step()
                        losses.append(loss.item())
                    
                    # Clear progress indicators
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Make predictions
                    model.eval()
                    with torch.no_grad():
                        train_predictions = model(X_train_tensor).cpu().numpy()
                        test_predictions = model(X_test_tensor).cpu().numpy()
                    
                    # Inverse transform
                    train_predictions = scaler.inverse_transform(train_predictions)
                    test_predictions = scaler.inverse_transform(test_predictions)
                    y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
                    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
                    
                    # Calculate finance-friendly metrics
                    train_mape = calculate_mape(y_train_actual.flatten(), train_predictions.flatten())
                    test_mape = calculate_mape(y_test_actual.flatten(), test_predictions.flatten())
                    train_rmse = np.sqrt(mean_squared_error(y_train_actual, train_predictions))
                    test_rmse = np.sqrt(mean_squared_error(y_test_actual, test_predictions))
                    
                    # Display finance-friendly metrics
                    st.subheader("🧠 AI Model Performance for Trading Decisions")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(
                            "Training MAPE", 
                            f"{train_mape:.2f}%",
                            help="How accurate the AI is on data it learned from. Lower is better."
                        )
                    with col2:
                        st.metric(
                            "Testing MAPE", 
                            f"{test_mape:.2f}%",
                            help="How accurate the AI is on new, unseen data. This is the real performance indicator."
                        )
                    
                    # Interpretation for finance professionals
                    if test_mape < 3:
                        st.success("🟢 **Excellent AI accuracy** - Suitable for day trading strategies")
                    elif test_mape < 7:
                        st.info("🟡 **Good AI accuracy** - Suitable for swing trading decisions")
                    elif test_mape < 15:
                        st.warning("🟠 **Fair AI accuracy** - Use for medium-term planning only")
                    else:
                        st.error("🔴 **Poor AI accuracy** - Avoid for short-term decisions")
                    
                    # Check for overfitting
                    if test_mape > train_mape * 1.5:
                        st.warning("⚠️ **Overfitting detected**: AI performs much better on training data than new data. Use predictions cautiously.")
                    
                    # Technical metrics (collapsed)
                    with st.expander("📊 Technical Metrics (Advanced Users)"):
                        col3, col4 = st.columns(2)
                        with col3:
                            st.metric("Train RMSE", f"${train_rmse:.2f}")
                        with col4:
                            st.metric("Test RMSE", f"${test_rmse:.2f}")
                    
                    # Plot results
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
                    
                    # Training loss
                    ax1.plot(losses)
                    ax1.set_title(f'{st.session_state.current_ticker} LSTM Training Loss')
                    ax1.set_xlabel('Epoch')
                    ax1.set_ylabel('Loss')
                    
                    # Predictions vs actual
                    train_dates = df['Date'].iloc[sequence_length:sequence_length+len(train_predictions)]
                    test_dates = df['Date'].iloc[sequence_length+len(train_predictions):sequence_length+len(train_predictions)+len(test_predictions)]
                    
                    ax2.plot(df['Date'], df['Close'], label='Actual', alpha=0.7)
                    ax2.plot(train_dates, train_predictions.flatten(), label='Train Predictions', alpha=0.8)
                    ax2.plot(test_dates, test_predictions.flatten(), label='Test Predictions', alpha=0.8)
                    ax2.legend()
                    ax2.set_title(f'{st.session_state.current_ticker} LSTM Predictions vs Actual')
                    ax2.set_xlabel('Date')
                    ax2.set_ylabel('Price ($)')
                    plt.xticks(rotation=45)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Future prediction
                    st.subheader("Next Day Prediction")
                    last_sequence = torch.FloatTensor(X[-1].reshape(1, sequence_length, 1)).to(device)
                    model.eval()
                    with torch.no_grad():
                        next_prediction = model(last_sequence).cpu().numpy()
                    next_price = scaler.inverse_transform(next_prediction)[0][0]
                    
                    st.success(f"Predicted next day closing price: ${next_price:.2f}")
                    
                    # User-friendly explanation
                    with st.expander("❓ How to Interpret These AI Results"):
                        st.markdown(explain_lstm_results())

if __name__ == "__main__":
    main() 