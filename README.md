# 📈 Finance Dashboard

A comprehensive finance dashboard for stock market analysis featuring interactive visualizations, ARIMA time series forecasting, and PyTorch LSTM deep learning predictions. 

**Default Data**: SPY (S&P 500 ETF) with complete 2024 calendar year data (253 trading days).

**Coming Soon**: Multi-stock support, sector analysis, portfolio optimization, and more in-depth analysis (more features or detail).

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

### 📚 Benchmark Comparison
- **Custom Ticker Window**: Ability to search custom tickers for comparison
- **Wide Range of Benchmarks**: Many benchmarks from index funds to fed intereest rates
- **Custom Date Ranges**: Ability to set a custom date range for benchmark comparison
- **In-Depth Analysis**: Various metrics used to offer both analysis and recommendations

### 🔍 Advanced Data
- **Custon Ticker Window**: Ability to search custom tickers for deep analysis
- **RSI & MACD Chart**: RSI & MACD charts to offer deeper insight than just price
- **Customizability**: Ability to check which indicators and features will be presented
- **Exportability**: Option to export the grabbed data as a .csv file 

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

## 📊 Data Information

### Current Dataset Coverage
The dashboard currently analyzes **SPY (S&P 500 ETF) data for the complete 2024 calendar year**:
- **Date Range**: January 2, 2024 → December 31, 2024
- **Trading Days**: 253 complete sessions
- **Data Quality**: High-frequency OHLCV data from MarketWatch

### Future Expansion Plans
- **Multiple Stocks**: Support for individual stocks (AAPL, GOOGL, TSLA, etc.)
- **Sector ETFs**: Technology (XLK), Healthcare (XLV), Finance (XLF), etc.
- **Extended Timeframes**: Multi-year historical analysis
- **Real-time Data**: Live market feeds and intraday analysis
- **Portfolio Analysis**: Multi-asset correlation and optimization

### Data Format
The `SPY.csv` file contains the following columns:
- `Date`: Trading date (MM/DD/YYYY format)
- `Open`: Opening price
- `High`: Highest price of the day
- `Low`: Lowest price of the day
- `Close`: Closing price
- `Volume`: Trading volume

### Dataset Notes
**Update**: Initially used a partial year dataset (Apr 2024 - May 2025), but switched to full 2024 calendar year for better analysis quality. A complete calendar year provides:
- More comprehensive seasonal pattern analysis
- Better model training with complete business cycles
- Avoidance of early 2025 market volatility
- Professional standard for annual financial reporting

**Development Roadmap**: While currently focused on SPY for proof-of-concept, the architecture is designed for easy expansion to multiple assets. The modular design allows for seamless integration of additional stocks, ETFs, and market data sources.

## 🔧 Technical Details

### Dependencies
- **Streamlit**: Web app framework
- **Pandas/NumPy**: Data manipulation
- **Matplotlib/Seaborn**: Static visualizations
- **Plotly**: Interactive charts
- **Scikit-learn**: Data preprocessing
- **Statsmodels**: ARIMA modeling
- **PyTorch**: LSTM neural networks (replaced TensorFlow for better compatibility)

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
5. **Try Advanced Data**: Try exploring different, perhaps more volatile, stock options
6. **Try Benchmark Comparison**: See how any given stock performs when compared to benchmarks

## 📈 Features Highlights

- **Interactive Interface**: Easy-to-use Streamlit interface with tabs
- **Real-time Analysis**: Run models on-demand with different parameters
- **Professional Charts**: Trading-grade candlestick and volume charts
- **Model Comparison**: Compare ARIMA vs LSTM predictions
- **Performance Metrics**: RMSE and other evaluation metrics included

## 🚀 Getting Started

1. Make sure your data file is in the `data/` directory (currently `SPY.csv`)
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`
4. Navigate through the tabs to explore different analyses

**Note**: Currently supports SPY data, but the framework is ready for multi-stock expansion.

## 📝 Notes

- The LSTM model trains for 50 epochs (can be adjusted in code)
- ARIMA parameters can be tuned in the interface
- All visualizations are interactive and responsive
- Data is automatically cached for better performance

## 🤝 Contributing

Feel free to fork this project and submit pull requests for improvements!

## 📄 License

This project is open source and available under the MIT License. 
