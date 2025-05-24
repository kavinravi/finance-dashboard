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
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

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

@st.cache_data
def load_data():
    """Load and preprocess the SPY data"""
    df = pd.read_csv('data/SPY.csv')
    
    # Clean the data
    df['Date'] = pd.to_datetime(df['Date'])
    df['Volume'] = df['Volume'].str.replace(',', '').astype(int)
    
    # Remove dollar signs and convert to float
    for col in ['Open', 'High', 'Low', 'Close']:
        df[col] = df[col].str.replace('"', '').astype(float)
    
    # Sort by date
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Calculate additional technical indicators
    df['Daily_Return'] = df['Close'].pct_change()
    df['High_Low_Pct'] = (df['High'] - df['Low']) / df['Close'] * 100
    df['Price_Change'] = df['Close'] - df['Open']
    df['Volatility'] = df['Daily_Return'].rolling(window=5).std()
    
    return df

def create_candlestick_chart(df):
    """Create an interactive candlestick chart"""
    fig = go.Figure(data=go.Candlestick(
        x=df['Date'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name="SPY"
    ))
    
    fig.update_layout(
        title="SPY Candlestick Chart",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        height=500,
        showlegend=False
    )
    
    return fig

def create_volume_chart(df):
    """Create volume chart"""
    fig = px.bar(df, x='Date', y='Volume', 
                 title="SPY Trading Volume",
                 color='Volume',
                 color_continuous_scale='viridis')
    
    fig.update_layout(height=400)
    return fig

def create_correlation_heatmap(df):
    """Create correlation heatmap"""
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Daily_Return', 'High_Low_Pct']
    corr_matrix = df[numeric_cols].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
    plt.title('Correlation Matrix of SPY Features')
    return fig

def create_pairplot(df):
    """Create seaborn pairplot"""
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    fig = plt.figure(figsize=(12, 10))
    
    # Create pairplot
    g = sns.pairplot(df[numeric_cols], diag_kind='hist', corner=True)
    g.fig.suptitle('SPY Stock Data Pairplot', y=1.02)
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

def create_lstm_model(sequence_length):
    """Create LSTM model"""
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def main():
    # Header
    st.markdown('<h1 class="main-header">📈 SPY Finance Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    
    # Sidebar
    st.sidebar.title("Dashboard Controls")
    
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
        candlestick_fig = create_candlestick_chart(df)
        st.plotly_chart(candlestick_fig, use_container_width=True)
        
        # Volume chart
        st.subheader("Volume Chart")
        volume_fig = create_volume_chart(df)
        st.plotly_chart(volume_fig, use_container_width=True)
        
        # Correlation heatmap
        st.subheader("Correlation Analysis")
        corr_fig = create_correlation_heatmap(df)
        st.pyplot(corr_fig)
        
        # Pairplot
        st.subheader("Pairplot Analysis")
        pair_fig = create_pairplot(df)
        st.pyplot(pair_fig)
    
    with tab3:
        st.header("ARIMA Time Series Analysis")
        
        # ARIMA parameters
        col1, col2, col3 = st.columns(3)
        with col1:
            p = st.selectbox("AR (p)", [0, 1, 2, 3], index=1)
        with col2:
            d = st.selectbox("I (d)", [0, 1, 2], index=1)
        with col3:
            q = st.selectbox("MA (q)", [0, 1, 2, 3], index=1)
        
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
                    ax.set_title('ARIMA Forecast')
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
        
        sequence_length = st.slider("Sequence Length", 5, 20, 10)
        
        if st.button("Run LSTM Analysis"):
            with st.spinner("Training LSTM model..."):
                X, y, scaler = prepare_lstm_data(df, sequence_length)
                
                if len(X) > 0:
                    # Split data
                    train_size = int(len(X) * 0.8)
                    X_train, X_test = X[:train_size], X[train_size:]
                    y_train, y_test = y[:train_size], y[train_size:]
                    
                    # Create and train model
                    model = create_lstm_model(sequence_length)
                    history = model.fit(X_train, y_train, epochs=50, batch_size=1, verbose=0)
                    
                    # Make predictions
                    train_predictions = model.predict(X_train)
                    test_predictions = model.predict(X_test)
                    
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
                    ax1.plot(history.history['loss'])
                    ax1.set_title('Model Training Loss')
                    ax1.set_xlabel('Epoch')
                    ax1.set_ylabel('Loss')
                    
                    # Predictions vs actual
                    train_dates = df['Date'].iloc[sequence_length:sequence_length+len(train_predictions)]
                    test_dates = df['Date'].iloc[sequence_length+len(train_predictions):sequence_length+len(train_predictions)+len(test_predictions)]
                    
                    ax2.plot(df['Date'], df['Close'], label='Actual', alpha=0.7)
                    ax2.plot(train_dates, train_predictions.flatten(), label='Train Predictions', alpha=0.8)
                    ax2.plot(test_dates, test_predictions.flatten(), label='Test Predictions', alpha=0.8)
                    ax2.legend()
                    ax2.set_title('LSTM Predictions vs Actual')
                    ax2.set_xlabel('Date')
                    ax2.set_ylabel('Price ($)')
                    plt.xticks(rotation=45)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Future prediction
                    st.subheader("Next Day Prediction")
                    last_sequence = X[-1].reshape(1, sequence_length, 1)
                    next_prediction = model.predict(last_sequence)
                    next_price = scaler.inverse_transform(next_prediction)[0][0]
                    
                    st.success(f"Predicted next day closing price: ${next_price:.2f}")

if __name__ == "__main__":
    main() 