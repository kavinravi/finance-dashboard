# 📈 SPY Finance Dashboard

A comprehensive finance dashboard for SPY stock analysis featuring interactive visualizations, ARIMA time series forecasting, and LSTM deep learning predictions.

## 🚀 Features

### 📊 Data Overview
- Real-time metrics display (current price, 52-week high/low, average volume)
- Interactive data tables with recent trading data
- Statistical summary of all trading metrics

### 📈 Visualizations
- **Interactive Candlestick Chart**: Professional trading chart with OHLC data
- **Volume Analysis**: Trading volume visualization with color coding
- **Correlation Heatmap**: Correlation analysis between different stock metrics
- **Seaborn Pairplot**: Comprehensive pairwise relationship analysis

### 🔮 ARIMA Analysis
- Interactive ARIMA model configuration (p, d, q parameters)
- 7-day price forecasting with confidence intervals
- Model summary and statistical diagnostics
- Visual forecast plots with historical data overlay

### 🧠 LSTM Deep Learning
- Configurable sequence length for time series modeling
- Train/test split with performance metrics (RMSE)
- Training loss visualization
- Predictions vs actual price comparisons
- Next-day price prediction

## 🛠️ Installation

1. **Clone or navigate to the project directory**
```bash
cd finance-dashboard
```

2. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the dashboard**
```bash
streamlit run app.py
```

4. **Open your browser** to `http://localhost:8501`

## 📁 Project Structure

```
finance-dashboard/
├── data/
│   └── SPY.csv          # SPY stock data (OHLCV format)
├── app.py               # Main Streamlit application
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## 📊 Data Format

The `SPY.csv` file should contain the following columns:
- `Date`: Trading date (MM/DD/YYYY format)
- `Open`: Opening price
- `High`: Highest price of the day
- `Low`: Lowest price of the day
- `Close`: Closing price
- `Volume`: Trading volume

## 🔧 Technical Details

### Dependencies
- **Streamlit**: Web app framework
- **Pandas/NumPy**: Data manipulation
- **Matplotlib/Seaborn**: Static visualizations
- **Plotly**: Interactive charts
- **Scikit-learn**: Data preprocessing
- **Statsmodels**: ARIMA modeling
- **TensorFlow**: LSTM neural networks

### Models

#### ARIMA (AutoRegressive Integrated Moving Average)
- Traditional time series forecasting
- Configurable (p,d,q) parameters
- 7-day ahead forecasting
- Confidence intervals included

#### LSTM (Long Short-Term Memory)
- Deep learning approach for time series
- Configurable sequence length (5-20 days)
- 80/20 train/test split
- Real-time next-day predictions

## 🎯 Usage Guide

1. **Start with Data Overview**: Check your data quality and basic statistics
2. **Explore Visualizations**: Understand price patterns and correlations
3. **Run ARIMA Analysis**: Get traditional statistical forecasts
4. **Try LSTM Analysis**: Compare with deep learning predictions

## 📈 Features Highlights

- **Interactive Interface**: Easy-to-use Streamlit interface with tabs
- **Real-time Analysis**: Run models on-demand with different parameters
- **Professional Charts**: Trading-grade candlestick and volume charts
- **Model Comparison**: Compare ARIMA vs LSTM predictions
- **Performance Metrics**: RMSE and other evaluation metrics included

## 🚀 Getting Started

1. Make sure your `SPY.csv` file is in the `data/` directory
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`
4. Navigate through the tabs to explore different analyses

## 📝 Notes

- The LSTM model trains for 50 epochs (can be adjusted in code)
- ARIMA parameters can be tuned in the interface
- All visualizations are interactive and responsive
- Data is automatically cached for better performance

## 🤝 Contributing

Feel free to fork this project and submit pull requests for improvements!

## 📄 License

This project is open source and available under the MIT License. 