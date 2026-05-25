# Kernel

Kernel is an AI financial analyst dashboard built with Streamlit, DSPy, OpenAI, yfinance, pandas, and ReportLab.

## Features

- Multi-stock market dashboard
- Portfolio weights input
- Live market data with `yfinance`
- Return, volatility, Sharpe proxy, max drawdown, concentration, and correlation metrics
- Risk-balanced and momentum-tilted educational weight suggestions
- Conversational financial analysis with DSPy + OpenAI
- Rule-based fallback when no OpenAI key is configured
- Downloadable PDF financial report

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push these files to GitHub.
2. Go to Streamlit Community Cloud.
3. Create a new app.
4. Select your GitHub repository.
5. Set main file path to:

```text
app.py
```

6. Deploy.

Optional secret:

```toml
OPENAI_API_KEY = "your_key"
OPENAI_MODEL = "openai/gpt-4o-mini"
```

Kernel works without an API key using the built-in rule-based fallback.

## Disclaimer

Kernel is an educational portfolio project, not financial advice.
