from functools import lru_cache
from io import BytesIO
import os

import pandas as pd
import streamlit as st
import yfinance as yf
from dotenv import load_dotenv
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


load_dotenv()

DEFAULT_TICKERS = "AAPL, MSFT, GOOGL"
DEFAULT_WEIGHTS = "40, 35, 25"
DEFAULT_QUESTION = "Give me a beginner-friendly portfolio and stock analysis."


st.set_page_config(
    page_title="Kernel - AI Financial Analyst",
    page_icon="📈",
    layout="wide",
)


def format_money(value):
    if value is None or pd.isna(value):
        return "N/A"
    return f"${value:,.2f}"


def format_percent(value):
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:.2%}"


def format_ratio(value):
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:.2f}"


def format_market_cap(value):
    if value is None or pd.isna(value):
        return "N/A"
    if value >= 1_000_000_000_000:
        return f"${value / 1_000_000_000_000:.2f}T"
    if value >= 1_000_000_000:
        return f"${value / 1_000_000_000:.2f}B"
    if value >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"
    return f"${value:,.0f}"


def parse_tickers(raw):
    tickers = [item.strip().upper() for item in raw.replace(";", ",").split(",")]
    tickers = [ticker for ticker in tickers if ticker]
    if not tickers:
        raise ValueError("Enter at least one ticker.")
    return list(dict.fromkeys(tickers))[:8]


def parse_weights(raw, count):
    if not raw.strip():
        return [1 / count] * count

    values = [float(item.strip()) for item in raw.replace(";", ",").split(",") if item.strip()]
    if len(values) != count:
        raise ValueError("Weights must match the number of tickers, for example: 40, 35, 25.")
    if any(value < 0 for value in values):
        raise ValueError("Weights cannot be negative.")

    total = sum(values)
    if total <= 0:
        raise ValueError("Weights must add up to more than zero.")
    return [value / total for value in values]


@st.cache_data(ttl=3600, show_spinner=False)
def load_history(ticker):
    history = yf.Ticker(ticker).history(period="1y", auto_adjust=False)
    if history.empty:
        raise ValueError(f"No market data found for {ticker}.")
    return history


@st.cache_data(ttl=3600, show_spinner=False)
def load_info(ticker):
    try:
        return yf.Ticker(ticker).info or {}
    except Exception:
        return {}


def ticker_snapshot(ticker, weight):
    history = load_history(ticker)
    info = load_info(ticker)
    close = history["Close"].dropna()
    volume = history["Volume"].dropna()

    current_price = float(close.iloc[-1])
    start_price = float(close.iloc[0])
    one_year_return = (current_price - start_price) / start_price
    daily_returns = close.pct_change().dropna()
    volatility = float(daily_returns.std() * (252 ** 0.5))

    sma_50 = float(close.rolling(50).mean().iloc[-1])
    sma_200 = float(close.rolling(200).mean().iloc[-1])
    running_max = close.cummax()
    max_drawdown = float(((close - running_max) / running_max).min())

    latest_volume = float(volume.iloc[-1]) if not volume.empty else None
    avg_volume = float(volume.tail(30).mean()) if not volume.empty else None
    pe_ratio = info.get("trailingPE")
    market_cap = info.get("marketCap")

    trend = "Bullish" if current_price >= sma_200 else "Cautious"
    if current_price >= sma_50 >= sma_200:
        trend = "Strong uptrend"
    elif current_price < sma_50 < sma_200:
        trend = "Downtrend"

    return {
        "Ticker": ticker,
        "Company": info.get("longName") or info.get("shortName") or ticker,
        "Sector": info.get("sector", "N/A"),
        "Industry": info.get("industry", "N/A"),
        "Weight": weight,
        "Weight Text": format_percent(weight),
        "Price": current_price,
        "Price Text": format_money(current_price),
        "1Y Return": one_year_return,
        "1Y Return Text": format_percent(one_year_return),
        "Volatility": volatility,
        "Volatility Text": format_percent(volatility),
        "50D SMA Text": format_money(sma_50),
        "200D SMA Text": format_money(sma_200),
        "Max Drawdown": max_drawdown,
        "Max Drawdown Text": format_percent(max_drawdown),
        "Market Cap Text": format_market_cap(market_cap),
        "P/E Text": format_ratio(pe_ratio),
        "Latest Volume": "N/A" if latest_volume is None else f"{latest_volume:,.0f}",
        "30D Avg Volume": "N/A" if avg_volume is None else f"{avg_volume:,.0f}",
        "Trend": trend,
    }


def build_price_frame(tickers):
    series = [load_history(ticker)["Close"].rename(ticker) for ticker in tickers]
    prices = pd.concat(series, axis=1).dropna()
    if prices.empty:
        raise ValueError("Could not align price histories for the selected tickers.")
    return prices


def portfolio_metrics(tickers, weights):
    prices = build_price_frame(tickers)
    returns = prices.pct_change().dropna()
    weights_series = pd.Series(weights, index=tickers)
    portfolio_returns = returns.mul(weights_series, axis=1).sum(axis=1)

    annual_return = float((1 + portfolio_returns.mean()) ** 252 - 1)
    annual_volatility = float(portfolio_returns.std() * (252 ** 0.5))
    sharpe = None if annual_volatility == 0 else annual_return / annual_volatility

    equity_curve = (1 + portfolio_returns).cumprod()
    max_drawdown = float(((equity_curve - equity_curve.cummax()) / equity_curve.cummax()).min())
    concentration = float(sum(weight ** 2 for weight in weights))
    correlation = returns.corr().round(2)

    vol_by_ticker = returns.std() * (252 ** 0.5)
    inverse_vol = 1 / vol_by_ticker.replace(0, float("nan"))
    suggested = (inverse_vol / inverse_vol.sum()).fillna(1 / len(tickers))

    momentum = prices.iloc[-1] / prices.iloc[0] - 1
    momentum_score = momentum.clip(lower=0.01) / vol_by_ticker.replace(0, float("nan"))
    momentum_weights = (momentum_score / momentum_score.sum()).fillna(1 / len(tickers))

    return {
        "Annual Return": annual_return,
        "Annual Return Text": format_percent(annual_return),
        "Annual Volatility": annual_volatility,
        "Annual Volatility Text": format_percent(annual_volatility),
        "Sharpe Proxy": sharpe,
        "Sharpe Proxy Text": format_ratio(sharpe),
        "Max Drawdown": max_drawdown,
        "Max Drawdown Text": format_percent(max_drawdown),
        "Concentration": concentration,
        "Concentration Text": format_ratio(concentration),
        "Correlation": correlation,
        "Risk-Balanced Weights": pd.DataFrame(
            {"Ticker": tickers, "Suggested Weight": [format_percent(float(suggested[t])) for t in tickers]}
        ),
        "Momentum Weights": pd.DataFrame(
            {"Ticker": tickers, "Suggested Weight": [format_percent(float(momentum_weights[t])) for t in tickers]}
        ),
    }


def build_context(snapshots, metrics):
    lines = [
        "KERNEL FINANCIAL DASHBOARD CONTEXT",
        f"Portfolio annualized return: {metrics['Annual Return Text']}",
        f"Portfolio annualized volatility: {metrics['Annual Volatility Text']}",
        f"Portfolio Sharpe proxy: {metrics['Sharpe Proxy Text']}",
        f"Portfolio max drawdown: {metrics['Max Drawdown Text']}",
        f"Concentration index: {metrics['Concentration Text']}",
        "",
        "Ticker rows:",
    ]
    for row in snapshots:
        lines.append(
            f"{row['Ticker']}: weight {row['Weight Text']}, price {row['Price Text']}, "
            f"1Y return {row['1Y Return Text']}, volatility {row['Volatility Text']}, "
            f"max drawdown {row['Max Drawdown Text']}, P/E {row['P/E Text']}, trend {row['Trend']}."
        )
    return "\n".join(lines)


def fallback_analysis(context, question):
    return {
        "Executive Summary": (
            "Kernel reviewed the selected tickers using recent market data, return, volatility, "
            "drawdown, moving-average trend, valuation, and portfolio concentration metrics."
        ),
        "Market Read": (
            "The strongest names are typically those with positive one-year return, price above the "
            "200-day moving average, and controlled drawdown. Weakness appears when volatility and "
            "drawdown rise faster than return."
        ),
        "Portfolio View": (
            "The portfolio should be evaluated through weighted return, annualized volatility, "
            "Sharpe proxy, and ticker concentration. Diversification reduces dependence on one volatile name."
        ),
        "Recommendations": (
            "Compare current weights with inverse-volatility and momentum-tilted weights. Treat the suggestions "
            "as educational diagnostics, not trading instructions."
        ),
        "Caveats": (
            f"Question considered: {question}. This is a rule-based fallback because no OpenAI API key is configured."
        ),
        "Mode": "Rule-based fallback",
    }


def ai_analysis(context, question):
    api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)
    if not api_key:
        return fallback_analysis(context, question)

    try:
        import dspy

        model_name = os.getenv("OPENAI_MODEL") or st.secrets.get("OPENAI_MODEL", "openai/gpt-4o-mini")
        lm = dspy.LM(model_name, api_key=api_key)
        dspy.configure(lm=lm)

        class KernelDashboardAnalyst(dspy.Signature):
            """
            Generate a structured educational financial analysis from factual market and portfolio metrics.
            Do not provide personalized financial advice.
            Do not give direct buy, sell, or hold instructions.
            """

            context: str = dspy.InputField()
            user_question: str = dspy.InputField()

            executive_summary: str = dspy.OutputField()
            market_read: str = dspy.OutputField()
            portfolio_view: str = dspy.OutputField()
            recommendations: str = dspy.OutputField()
            caveats: str = dspy.OutputField()

        analyst = dspy.ChainOfThought(KernelDashboardAnalyst)
        response = analyst(context=context, user_question=question)
        return {
            "Executive Summary": response.executive_summary,
            "Market Read": response.market_read,
            "Portfolio View": response.portfolio_view,
            "Recommendations": response.recommendations,
            "Caveats": response.caveats,
            "Mode": f"DSPy + {model_name}",
        }
    except Exception as exc:
        result = fallback_analysis(context, question)
        result["Mode"] = f"Fallback after AI error: {exc}"
        return result


def make_pdf(snapshots, metrics, analysis, question):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=1.5 * cm, leftMargin=1.5 * cm)
    styles = getSampleStyleSheet()
    story = [
        Paragraph("Kernel AI Financial Analyst Report", styles["Title"]),
        Paragraph("Educational report, not financial advice.", styles["Normal"]),
        Spacer(1, 12),
        Paragraph("Question", styles["Heading2"]),
        Paragraph(question, styles["BodyText"]),
        Spacer(1, 8),
        Paragraph("Portfolio Metrics", styles["Heading2"]),
    ]
    metric_rows = [
        ["Annualized Return", metrics["Annual Return Text"]],
        ["Annualized Volatility", metrics["Annual Volatility Text"]],
        ["Sharpe Proxy", metrics["Sharpe Proxy Text"]],
        ["Max Drawdown", metrics["Max Drawdown Text"]],
        ["Concentration Index", metrics["Concentration Text"]],
    ]
    metric_table = Table(metric_rows, colWidths=[8 * cm, 7 * cm])
    metric_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F4F7FB")),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#CBD5E1")),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.extend([metric_table, Spacer(1, 10), Paragraph("Ticker Snapshot", styles["Heading2"])])

    rows = [["Ticker", "Weight", "Price", "1Y Return", "Volatility", "Trend"]]
    for row in snapshots:
        rows.append([
            row["Ticker"],
            row["Weight Text"],
            row["Price Text"],
            row["1Y Return Text"],
            row["Volatility Text"],
            row["Trend"],
        ])
    table = Table(rows, colWidths=[2 * cm, 2.2 * cm, 2.4 * cm, 2.4 * cm, 2.5 * cm, 3.2 * cm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17202A")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#CBD5E1")),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.extend([table, Spacer(1, 12)])

    for title in ["Executive Summary", "Market Read", "Portfolio View", "Recommendations", "Caveats"]:
        story.append(Paragraph(title, styles["Heading2"]))
        story.append(Paragraph(str(analysis[title]).replace("\n", "<br/>"), styles["BodyText"]))
        story.append(Spacer(1, 8))

    doc.build(story)
    buffer.seek(0)
    return buffer


st.markdown(
    """
    <style>
    .main .block-container { padding-top: 2rem; max-width: 1180px; }
    .kernel-title { font-size: 4rem; line-height: 1; font-weight: 850; margin: 0; }
    .kernel-kicker { color: #08746f; font-weight: 800; letter-spacing: .08em; text-transform: uppercase; }
    .kernel-sub { color: #5d6b78; font-size: 1.05rem; max-width: 820px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="kernel-kicker">Kernel</div>', unsafe_allow_html=True)
st.markdown('<h1 class="kernel-title">AI Financial Analyst Dashboard</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="kernel-sub">Multi-stock market data, portfolio analytics, DSPy reasoning chains, '
    'conversational queries, optimization-style weight suggestions, and automated PDF reports.</p>',
    unsafe_allow_html=True,
)

with st.form("kernel_form"):
    col1, col2 = st.columns(2)
    tickers_raw = col1.text_input("Tickers", DEFAULT_TICKERS, help="Comma-separated tickers, max 8.")
    weights_raw = col2.text_input("Portfolio weights", DEFAULT_WEIGHTS, help="Comma-separated weights. Leave blank for equal weights.")
    question = st.text_area("Ask Kernel", DEFAULT_QUESTION, height=120)
    submitted = st.form_submit_button("Analyze Portfolio", type="primary")

if submitted:
    try:
        with st.spinner("Retrieving market data and building analysis..."):
            tickers = parse_tickers(tickers_raw)
            weights = parse_weights(weights_raw, len(tickers))
            snapshots = [ticker_snapshot(ticker, weight) for ticker, weight in zip(tickers, weights)]
            metrics = portfolio_metrics(tickers, weights)
            context = build_context(snapshots, metrics)
            analysis = ai_analysis(context, question)

        st.caption(f"Insight engine: {analysis['Mode']}")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Annual Return", metrics["Annual Return Text"])
        m2.metric("Annual Volatility", metrics["Annual Volatility Text"])
        m3.metric("Sharpe Proxy", metrics["Sharpe Proxy Text"])
        m4.metric("Max Drawdown", metrics["Max Drawdown Text"])

        st.subheader("AI Executive Summary")
        st.write(analysis["Executive Summary"])

        tab1, tab2, tab3, tab4 = st.tabs(["Dashboard", "Portfolio", "AI Analysis", "PDF Report"])

        with tab1:
            dashboard_df = pd.DataFrame([
                {
                    "Ticker": row["Ticker"],
                    "Company": row["Company"],
                    "Weight": row["Weight Text"],
                    "Price": row["Price Text"],
                    "1Y Return": row["1Y Return Text"],
                    "Volatility": row["Volatility Text"],
                    "Max Drawdown": row["Max Drawdown Text"],
                    "P/E": row["P/E Text"],
                    "Trend": row["Trend"],
                }
                for row in snapshots
            ])
            st.dataframe(dashboard_df, use_container_width=True, hide_index=True)

            st.subheader("Microstructure Proxy")
            micro_df = pd.DataFrame([
                {
                    "Ticker": row["Ticker"],
                    "Sector": row["Sector"],
                    "Industry": row["Industry"],
                    "Market Cap": row["Market Cap Text"],
                    "Latest Volume": row["Latest Volume"],
                    "30D Avg Volume": row["30D Avg Volume"],
                    "50D SMA": row["50D SMA Text"],
                    "200D SMA": row["200D SMA Text"],
                }
                for row in snapshots
            ])
            st.dataframe(micro_df, use_container_width=True, hide_index=True)

        with tab2:
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Risk-Balanced Weights")
                st.caption("Inverse-volatility allocation. Educational, not a recommendation.")
                st.dataframe(metrics["Risk-Balanced Weights"], use_container_width=True, hide_index=True)
            with c2:
                st.subheader("Momentum-Tilted Weights")
                st.caption("Positive one-year return divided by volatility. Educational, not a recommendation.")
                st.dataframe(metrics["Momentum Weights"], use_container_width=True, hide_index=True)

            st.subheader("Correlation Matrix")
            st.dataframe(metrics["Correlation"], use_container_width=True)

        with tab3:
            st.subheader("Market Read")
            st.write(analysis["Market Read"])
            st.subheader("Portfolio View")
            st.write(analysis["Portfolio View"])
            st.subheader("Educational Recommendations")
            st.write(analysis["Recommendations"])
            st.subheader("Caveats")
            st.write(analysis["Caveats"])

        with tab4:
            pdf = make_pdf(snapshots, metrics, analysis, question)
            st.download_button(
                "Download PDF Report",
                data=pdf,
                file_name="kernel_financial_report.pdf",
                mime="application/pdf",
            )

        st.info("Kernel is an educational portfolio project, not financial advice.")

    except Exception as exc:
        st.error(str(exc))
else:
    st.info("Enter tickers and weights, then click Analyze Portfolio.")
