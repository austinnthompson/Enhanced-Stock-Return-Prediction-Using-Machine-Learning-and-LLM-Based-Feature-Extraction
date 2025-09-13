1) fix code
2) update read me


# Enhanced-Stock-Return-Prediction-Using-Machine-Learning-and-LLM-Based-Feature-Extraction 

## Executive Summary

This project develops an advanced stock forecasting system that integrates quantitative historical price data with qualitative insights extracted from financial news using Retrieval-Augmented Generation (RAG) and Large Language Models (LLMs). Focused on stocks like AAPL and TSLA, it fetches 1-year weekly historical prices via yFinance, collects news articles using NewsAPI and GDELT, embeds them with Sentence Transformers, and builds a FAISS index for efficient retrieval.

An LLM (GPT-4), via OpenAI API, queries the RAG system to derive features such as sentiment (positive/negative/neutral), risk level (low/medium/high), growth drivers, and analyst notes. These are merged with price data, enhanced with VADER sentiment scores, RSI, MACD, and rolling statistics. A stacking regressor model (combining Gradient Boosting, Linear Regression, and ElasticNet) predicts next-period log returns.

The system supports scenario analysis by adjusting features (e.g., high volume, positive sentiment) to estimate drift (mu) and volatility (sigma), followed by Monte Carlo simulations for 1-week price forecasts. Bootstrapping expands the dataset for robust training. Results include MAE/RMSE metrics, cross-validation scores, and visualizations of paths and distributions.

Key improvements noted: Hyperparameter tuning, modular code organization, SQL storage for data, and conditional model updates. This approach bridges traditional finance with AI, enabling nuanced predictions beyond pure quantitative models.

## Project Overview

A Jupyter Notebook-based pipeline for AI-driven stock analysis, combining historical data, and ML for forecasting.

### Key Features
- **Data Collection**: Weekly stock prices (yFinance) and news (NewsAPI/GDELT).
- **RAG System**: Embeddings and FAISS for retrieving relevant financial texts.
- **LLM Feature Extraction**: GPT-4 derives sentiment, risk, and growth insights.
- **Feature Engineering**: VADER scores, technical indicators (RSI, MACD), and categorical encoding.
- **ML Modeling**: Stacking regressor for predicting log returns; scenario-based adjustments.
- **Simulation & Visualization**: Monte Carlo paths and outcome distributions using Matplotlib.
- **Scalability Notes**: Suggestions for SQL integration and hyperparameter tuning.

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-repo/stock-ai-analysis.git
   cd stock-ai-analysis
   ```

2. Install dependencies (Python 3.11 recommended):
   ```
   pip install pandas numpy yfinance requests newsapi-python openai sentence-transformers faiss-cpu vaderSentiment scikit-learn matplotlib tqdm
   ```

3. Set API keys in the notebook:
   - NewsAPI: `NEWSAPI_KEY = "your_key"`
   - OpenAI: `OPENAI_API_KEY = "your_key"`

## Usage

1. Open the Jupyter Notebook: `jupyter notebook Project_draft_thompson.ipynb`
2. Run cells sequentially to:
   - Fetch and process data (Steps 1-5).
   - Train models and run simulations (Steps 6-8).
3. Customize:
   - Tickers: Edit `TICKERS = ["AAPL", "TSLA"]`
   - Scenario: Update `SCENARIO` dictionary for feature adjustments.
   - Training: Select ticker in `TRAINING_TICKER` and use bootstrapped data.

## Results Example

For TSLA (scenario: high volume, positive sentiment, high risk):
- Drift (mu): 0.2215
- Volatility (sigma): 0.1970
- Expected Price: $371.22 (from starting $333.87)
- Visuals: Monte Carlo paths and histogram.

## Limitations & Improvements
- API rate limits may affect news fetching.
- No real-time data; historical focus.
- Suggestions: Hyperparameter tuning, SQL for data storage, modular cells for better organization.
