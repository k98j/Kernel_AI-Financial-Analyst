## 📈 Kernel — AI Financial Analyst Dashboard  
**Tech:** Python · Flask · DSPy · OpenAI · yfinance · pandas · ReportLab

Analyzes public stock tickers and portfolios using live market data, financial metrics, and AI-generated insights. Retrieves one-year price history, computes return, volatility, Sharpe proxy, drawdown, moving averages, correlation, valuation ratios, and portfolio concentration. Uses DSPy reasoning chains with OpenAI to generate structured analyst-style summaries, market reads, risk commentary, and educational recommendations. Includes portfolio weight analysis, risk-balanced allocation suggestions, momentum-tilted weights, conversational financial queries, and automated PDF report generation.

**Key skills:**  Financial analytics · Flask deployment · DSPy · LLM reasoning · Portfolio analysis · PDF reporting

**Live App → (https://kernelapp-f8vc3yfuwn6cqeomgipmqm.streamlit.app/)**


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
