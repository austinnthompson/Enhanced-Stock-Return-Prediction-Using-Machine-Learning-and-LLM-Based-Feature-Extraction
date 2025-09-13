2) upload code
3) update read me
4) upload powerpoint
5) upload html file example


# Enhanced Stock Return Prediction Using Machine Learning and LLM-Based Feature Extraction

## Executive Summary

This project presents a sophisticated stock forecasting framework that synergizes quantitative historical price data with qualitative insights derived from financial news, leveraging Retrieval-Augmented Generation (RAG) and Large Language Models (LLMs). Targeting stocks such as AAPL and TSLA, the system retrieves 1-year weekly historical price data via yFinance and collects news articles through NewsAPI and GDELT. These texts are embedded using Sentence Transformers and indexed with FAISS for efficient retrieval.

A Large Language Model (GPT-4), accessed via the OpenAI API, queries the RAG system to extract qualitative features, including sentiment (positive, negative, neutral), risk level (low, medium, high), growth drivers, and analyst notes. These features are integrated with price data and enriched with technical indicators (RSI, MACD, rolling averages, and momentum) and VADER sentiment scores. A stacking regressor model, combining Gradient Boosting, Linear Regression, and ElasticNet, predicts next-period log returns with robust cross-validation.

The framework enables scenario analysis by modifying input features (e.g., high trading volume, positive sentiment) to estimate drift (mu) and volatility (sigma). Monte Carlo simulations generate 1-week price forecasts, visualized through price paths and distribution plots using Matplotlib. Bootstrapping enhances dataset robustness by expanding sample size for training.

This methodology transcends traditional financial modeling by incorporating AI-driven qualitative analysis, offering nuanced predictions. Its versatility extends beyond finance, applicable to any domain requiring feature extraction from large contextual datasets, such as market research or policy analysis.

## Project Overview

A Jupyter Notebook-based pipeline for AI-driven stock analysis, integrating historical data, qualitative insights, and machine learning for accurate forecasting.

## Repository Files
1. `project_code.ipynb`: Core Python code (download to run).
2. `project_code.html`: HTML export of the notebook (download to view).
3. `executive_presentation.pptx`: Executive PowerPoint and overview.

### Key Features
- **Data Collection**: Weekly stock prices via yFinance and news articles from NewsAPI and GDELT.
- **RAG System**: Sentence Transformer embeddings with FAISS for efficient document retrieval.
- **LLM Feature Extraction**: GPT-4 extracts sentiment, risk, growth drivers, and analyst notes.
- **Feature Engineering**: Includes VADER sentiment scores, technical indicators (RSI, MACD, momentum), and categorical encoding.
- **ML Modeling**: Stacking regressor predicts log returns; supports scenario-based forecasting.
- **Simulation & Visualization**: Monte Carlo simulations with Matplotlib for price paths and outcome distributions.

## Installation

1. Download the Jupyter Notebook (`project_code.ipynb`).
2. Install dependencies (Python 3.11 recommended):
   ```
   pip install pandas numpy yfinance requests newsapi-python openai sentence-transformers faiss-cpu vaderSentiment scikit-learn matplotlib tqdm
   ```
3. Configure API keys in the notebook:
   - NewsAPI: `NEWSAPI_KEY = "your_key"`
   - OpenAI: `OPENAI_API_KEY = "your_key"`

## Usage

1. Launch the Jupyter Notebook: `jupyter notebook project_code.ipynb`
2. Execute cells sequentially to:
   - Fetch and process data (Steps 1–5).
   - Train models and run scenario-based simulations (Steps 6–8).
3. Customize as needed:
   - Tickers: Modify `TICKERS = ["AAPL", "TSLA"]`.
   - Scenario: Adjust `SCENARIO` dictionary for feature-based predictions.
   - Training: Select ticker in `TRAINING_TICKER` and leverage bootstrapped data.

## Limitations & Improvements
- **API Constraints**: NewsAPI and GDELT rate limits may restrict data collection.
- **Data Scope**: Focused on historical data; real-time integration not included.
- **Potential Enhancements**:
  - Implement SQL databases for scalable data storage and retrieval.
  - Optimize data query functions to fetch only new data, reducing API call costs.
  - Explore additional data sources (e.g., SEC filings, social media) for richer qualitative features.
