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

# Machine Learning imports
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Set page config
st.set_page_config(
    page_title="SPY Finance Dashboard",
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

def validate_ticker_format(ticker):
    """Validate ticker symbol format"""
    if not ticker:
        return False, "❌ **Ticker required**: Please enter a ticker symbol."
    
    if len(ticker) < 1 or len(ticker) > 5:
        return False, "❌ **Ticker length**: Ticker must be 1-5 characters (e.g., AAPL, GOOGL)."
    
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
        model = ARIMA(ts_data, order=order)
        fitted_model = model.fit()
        
        # Make predictions
        forecast_steps = 7  # Predict next 7 days
        forecast = fitted_model.forecast(steps=forecast_steps)
        conf_int = fitted_model.get_forecast(steps=forecast_steps).conf_int()
        
        return fitted_model, forecast, conf_int
    except Exception as e:
        st.error(f"ARIMA model error: {str(e)}")
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
    """PyTorch LSTM model"""
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(hidden_size, 25)
        self.fc2 = nn.Linear(25, output_size)
        
    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Take the last output
        out = out[:, -1, :]
        out = self.dropout(out)
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)
        
        return out

def create_lstm_model(sequence_length):
    """Create PyTorch LSTM model"""
    model = LSTMModel(input_size=1, hidden_size=50, num_layers=2, output_size=1)
    return model

def main():
    # Initialize session state for uploaded data
    if 'uploaded_data' not in st.session_state:
        st.session_state.uploaded_data = None
    if 'current_ticker' not in st.session_state:
        st.session_state.current_ticker = "SPY"
    
    # Dynamic header with current ticker
    st.markdown(f'<h1 class="main-header">📈 {st.session_state.current_ticker} Finance Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Dashboard Controls")
    
    # Cache management
    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()
    
    st.sidebar.markdown("---")
    
    # File upload section
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
        max_chars=5,
        help="Enter the ticker symbol (e.g., AAPL, NVDA, GOOGL). Must be 1-5 letters only."
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
        # Use previously uploaded data
        df = load_data(use_uploaded=True, uploaded_data=st.session_state.uploaded_data)
        use_uploaded_data = True
    else:
        # Use default SPY data
        df = load_data()
        st.session_state.current_ticker = "SPY"
    
    # Check if data loaded successfully
    if df.empty:
        st.error("Failed to load data. Please check the data file and try again.")
        return
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Overview", "📈 Visualizations", "🔮 ARIMA Analysis", "🧠 LSTM Analysis"])
    
    with tab1:
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
    
    with tab2:
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
    
    with tab3:
        st.header("ARIMA Time Series Analysis")
        
        st.info("💡 **Recommended starting point**: p=1, d=1, q=1 (selected by default) works well for most stocks including SPY.")
        
        # ARIMA parameters with help tooltips
        col1, col2, col3 = st.columns(3)
        with col1:
            p = st.selectbox(
                "AR (p) - AutoRegressive", 
                [0, 1, 2, 3], 
                index=1,
                help="How much yesterday's price influences today's prediction. **p=1**: Good for stable stocks (blue chips, utilities). **p=2**: Better for volatile stocks (tech, growth). **p=3**: For highly unpredictable stocks. Start with p=1 for established companies."
            )
        with col2:
            d = st.selectbox(
                "I (d) - Integration", 
                [0, 1, 2], 
                index=1,
                help="Removes price trends to focus on price changes. **d=1**: Standard for stock prices (recommended). **d=0**: Only for already stable price series. **d=2**: Rarely needed. Keep at d=1 for normal stock analysis."
            )
        with col3:
            q = st.selectbox(
                "MA (q) - Moving Average", 
                [0, 1, 2, 3], 
                index=1,
                help="How much recent market 'shocks' affect predictions. **q=1**: Good for most stocks, captures immediate reactions. **q=2**: For stocks sensitive to news/events. **q=0**: For very predictable stocks. q=1 works well for SPY and major ETFs."
            )
        
        if st.button("Run ARIMA Analysis"):
            with st.spinner("Training ARIMA model..."):
                ts_data = prepare_arima_data(df)
                model, forecast, conf_int = fit_arima_model(ts_data, order=(p, d, q))
                
                if model is not None:
                    # Model summary
                    st.subheader("Model Summary")
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
    
    with tab4:
        st.header("LSTM Deep Learning Analysis")
        
        st.info("💡 **Recommended starting point**: Sequence Length = 10 days (selected by default) works well for most stocks.")
        
        sequence_length = st.slider(
            "Sequence Length (days)", 
            5, 20, 10,
            help="How many previous trading days the AI model looks at to predict the next day. **5-7 days**: For highly volatile stocks that change quickly. **10-12 days**: Good for most stocks including SPY (captures ~2 weeks). **15-20 days**: For very stable stocks or longer-term patterns. More days = more context but slower predictions."
        )
        
        if st.button("Run LSTM Analysis"):
            with st.spinner("Training LSTM model..."):
                X, y, scaler = prepare_lstm_data(df, sequence_length)
                
                if len(X) > 0:
                    # Split data
                    train_size = int(len(X) * 0.8)
                    X_train, X_test = X[:train_size], X[train_size:]
                    y_train, y_test = y[:train_size], y[train_size:]
                    
                    # Convert to PyTorch tensors
                    X_train_tensor = torch.FloatTensor(X_train)
                    y_train_tensor = torch.FloatTensor(y_train.reshape(-1, 1))
                    X_test_tensor = torch.FloatTensor(X_test)
                    y_test_tensor = torch.FloatTensor(y_test.reshape(-1, 1))
                    
                    # Create model
                    model = create_lstm_model(sequence_length)
                    criterion = nn.MSELoss()
                    optimizer = optim.Adam(model.parameters(), lr=0.001)
                    
                    # Training
                    model.train()
                    losses = []
                    epochs = 50
                    
                    for epoch in range(epochs):
                        optimizer.zero_grad()
                        outputs = model(X_train_tensor)
                        loss = criterion(outputs, y_train_tensor)
                        loss.backward()
                        optimizer.step()
                        losses.append(loss.item())
                    
                    # Make predictions
                    model.eval()
                    with torch.no_grad():
                        train_predictions = model(X_train_tensor).numpy()
                        test_predictions = model(X_test_tensor).numpy()
                    
                    # Inverse transform
                    train_predictions = scaler.inverse_transform(train_predictions)
                    test_predictions = scaler.inverse_transform(test_predictions)
                    y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
                    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
                    
                    # Calculate metrics
                    train_rmse = np.sqrt(mean_squared_error(y_train_actual, train_predictions))
                    test_rmse = np.sqrt(mean_squared_error(y_test_actual, test_predictions))
                    
                    # Display metrics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Train RMSE", f"${train_rmse:.2f}")
                    with col2:
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
                    last_sequence = torch.FloatTensor(X[-1].reshape(1, sequence_length, 1))
                    model.eval()
                    with torch.no_grad():
                        next_prediction = model(last_sequence).numpy()
                    next_price = scaler.inverse_transform(next_prediction)[0][0]
                    
                    st.success(f"Predicted next day closing price: ${next_price:.2f}")

if __name__ == "__main__":
    main() 