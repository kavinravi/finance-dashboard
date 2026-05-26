# 🔐 API Keys Setup Guide

This guide shows you how to securely configure your API keys for the Finance Dashboard's advanced features.

## 🔑 Required API Keys

### 1. Financial Modeling Prep (FMP)
- **Purpose**: Fundamental analysis, financial statements, ratios
- **Free Tier**: 250 requests/day
- **Get your key**: [financialmodelingprep.com](https://financialmodelingprep.com/developer/docs)

### 2. Alpha Vantage
- **Purpose**: Enhanced technical indicators, professional-grade analysis
- **Free Tier**: 5 requests/minute, 500 requests/day
- **Get your key**: [alphavantage.co](https://www.alphavantage.co/support/#api-key)

## 🏠 Local Development Setup

### Step 1: Create .env file
Create a `.env` file in your project root:

```bash
# Financial Modeling Prep API
FMP_API_KEY=your_fmp_api_key_here

# Alpha Vantage API  
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key_here

# Optional: Rate limiting settings
FMP_RATE_LIMIT=60
ALPHA_VANTAGE_RATE_LIMIT=5
```

### Step 2: Verify Setup
Run the app locally and check the "🔍 Advanced Data" tab. You should see:
- 🟢 Active status for both APIs
- Options to enable "Fundamental Analysis" and "Enhanced Indicators"

## ☁️ Streamlit Cloud Deployment

### Step 1: App Settings
1. Go to your Streamlit Cloud dashboard
2. Click on your app
3. Go to "Settings" → "Secrets"

### Step 2: Add Secrets
Add your API keys in this format:

```toml
FMP_API_KEY = "your_fmp_api_key_here"
ALPHA_VANTAGE_API_KEY = "your_alpha_vantage_api_key_here"
FMP_RATE_LIMIT = 60
ALPHA_VANTAGE_RATE_LIMIT = 5
```

### Step 3: Deploy
- Save the secrets
- Your app will automatically restart
- Check the Advanced Data tab for 🟢 Active status

## ⚠️ Security Best Practices

### ✅ DO:
- Store API keys in `.env` (local) or Streamlit Secrets (cloud)
- Keep `.env` file in `.gitignore` (already configured)
- Use environment variables for sensitive data

### ❌ DON'T:
- Commit API keys to Git/GitHub
- Share API keys in code or documentation
- Use production keys in development

## 🔧 Troubleshooting

### API Status Shows "🔴 Not configured"
1. **Local**: Check your `.env` file exists and has correct format
2. **Cloud**: Verify secrets are properly formatted in Streamlit settings
3. **Both**: Ensure no extra spaces or quotes around keys

### Rate Limit Errors
- **FMP**: Free tier allows 250 requests/day
- **Alpha Vantage**: Free tier allows 5 requests/minute
- The app includes built-in rate limiting and caching

### API Response Errors
- **Check ticker validity**: Ensure the stock ticker exists
- **Check API key validity**: Test your keys directly with the API providers
- **Check rate limits**: Wait a few minutes and try again

## 📊 Features Unlocked

With properly configured API keys, you get:

### 🏢 Fundamental Analysis (FMP)
- Company information (sector, industry, market cap)
- Valuation ratios (P/E, P/B, P/S, EV/EBITDA)
- Financial health metrics (ROE, ROA, debt ratios)
- Latest financial statements (revenue, net income, EPS)

### 📈 Enhanced Indicators (Alpha Vantage)
- Professional-grade RSI calculations
- Advanced MACD analysis
- Additional technical indicators
- Cross-verified data sources

## 🚀 Getting Started

1. Get your free API keys from the providers above
2. Set them up using this guide
3. Go to the "🔍 Advanced Data" tab
4. Try collecting data for AAPL or any ticker
5. See the enhanced fundamental and technical analysis!

---

**Need help?** Check the in-app "🔧 How to Configure API Keys" section in the Advanced Data tab. 