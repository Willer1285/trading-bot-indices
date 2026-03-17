"""
LLM-based Sentiment Analysis for Trading
Uses pre-trained language models to analyze financial news and market sentiment
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from loguru import logger
from pathlib import Path
import pickle
from datetime import datetime, timedelta

try:
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        pipeline
    )
    import torch
    LLM_AVAILABLE = True
except ImportError:
    logger.warning("transformers not installed. LLM sentiment analysis will not be available.")
    LLM_AVAILABLE = False

try:
    from newsapi import NewsApiClient
    NEWS_API_AVAILABLE = True
except ImportError:
    logger.warning("newsapi-python not installed. News fetching will not be available.")
    NEWS_API_AVAILABLE = False

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    logger.warning("yfinance not installed. Yahoo Finance news will not be available.")
    YFINANCE_AVAILABLE = False


class LLMSentimentAnalyzer:
    """
    LLM-based sentiment analyzer for financial markets.

    Uses pre-trained models like:
    - FinBERT: Financial sentiment analysis
    - DistilBERT: General sentiment analysis
    - Twitter-RoBERTa: Social media sentiment

    Analyzes news, social media, and market commentary to generate
    sentiment scores that can improve trading decisions.
    """

    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        news_api_key: Optional[str] = None,
        cache_hours: int = 1
    ):
        """
        Initialize LLM sentiment analyzer.

        Args:
            model_name: Hugging Face model name
                - "ProsusAI/finbert": Financial sentiment (recommended)
                - "distilbert-base-uncased-finetuned-sst-2-english": General sentiment
                - "cardiffnlp/twitter-roberta-base-sentiment": Social media sentiment
            news_api_key: API key for NewsAPI.org
            cache_hours: Hours to cache sentiment scores
        """
        if not LLM_AVAILABLE:
            raise ImportError("transformers and torch are required for LLM sentiment")

        self.name = "LLM_Sentiment"
        self.model_name = model_name
        self.news_api_key = news_api_key
        self.cache_hours = cache_hours

        self.model = None
        self.tokenizer = None
        self.sentiment_pipeline = None
        self.news_client = None

        # Sentiment cache
        self.sentiment_cache = {}
        self.cache_timestamps = {}

        self.is_fitted = True  # LLM is pre-trained

        logger.info(f"Initialized LLM sentiment analyzer with model={model_name}")

    def load_model(self):
        """Load pre-trained sentiment model."""
        if self.model is not None:
            return  # Already loaded

        logger.info(f"Loading LLM model: {self.model_name}...")

        device = 0 if torch.cuda.is_available() else -1

        try:
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                device=device,
                truncation=True,
                max_length=512
            )

            logger.success(f"LLM model loaded successfully on device={device}")

        except Exception as e:
            logger.error(f"Failed to load LLM model: {e}")
            raise

        # Initialize news API if available
        if NEWS_API_AVAILABLE and self.news_api_key:
            try:
                self.news_client = NewsApiClient(api_key=self.news_api_key)
                logger.info("NewsAPI client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize NewsAPI: {e}")

    def fetch_news(
        self,
        symbol: str,
        days_back: int = 1,
        max_articles: int = 20
    ) -> List[Dict]:
        """
        Fetch recent news articles for a symbol.

        Args:
            symbol: Trading symbol (e.g., "SPY", "NQ100")
            days_back: Number of days to look back
            max_articles: Maximum number of articles to fetch

        Returns:
            List of news articles with titles and descriptions
        """
        articles = []

        # Try NewsAPI first
        if self.news_client is not None:
            try:
                from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')

                # Map indices to search queries
                query_map = {
                    'SPY': 'S&P 500 OR SPY',
                    'NQ': 'NASDAQ OR NQ100',
                    'US30': 'Dow Jones OR US30',
                    'US100': 'NASDAQ 100',
                    'GER40': 'DAX OR GER40',
                    'UK100': 'FTSE OR UK100',
                }

                query = query_map.get(symbol.upper(), symbol)

                response = self.news_client.get_everything(
                    q=query,
                    from_param=from_date,
                    language='en',
                    sort_by='relevancy',
                    page_size=max_articles
                )

                for article in response.get('articles', [])[:max_articles]:
                    articles.append({
                        'title': article.get('title', ''),
                        'description': article.get('description', ''),
                        'content': article.get('content', ''),
                        'source': article.get('source', {}).get('name', 'Unknown'),
                        'published_at': article.get('publishedAt', '')
                    })

                logger.info(f"Fetched {len(articles)} articles from NewsAPI for {symbol}")

            except Exception as e:
                logger.warning(f"Failed to fetch news from NewsAPI: {e}")

        # Try Yahoo Finance as fallback
        if len(articles) == 0 and YFINANCE_AVAILABLE:
            try:
                ticker = yf.Ticker(symbol)
                news = ticker.news

                for item in news[:max_articles]:
                    articles.append({
                        'title': item.get('title', ''),
                        'description': item.get('summary', ''),
                        'content': item.get('summary', ''),
                        'source': item.get('publisher', 'Yahoo Finance'),
                        'published_at': datetime.fromtimestamp(item.get('providerPublishTime', 0)).isoformat()
                    })

                logger.info(f"Fetched {len(articles)} articles from Yahoo Finance for {symbol}")

            except Exception as e:
                logger.warning(f"Failed to fetch news from Yahoo Finance: {e}")

        return articles

    def analyze_text(self, text: str) -> Dict:
        """
        Analyze sentiment of a single text.

        Args:
            text: Text to analyze

        Returns:
            Dictionary with sentiment label and score
        """
        if self.sentiment_pipeline is None:
            self.load_model()

        if not text or len(text.strip()) == 0:
            return {'label': 'neutral', 'score': 0.5}

        try:
            # Get sentiment prediction
            result = self.sentiment_pipeline(text[:512])[0]  # Truncate to max length

            # Normalize labels
            label = result['label'].lower()

            # Map labels to standardized format
            if 'positive' in label or 'bullish' in label:
                sentiment = 'positive'
                score = result['score']
            elif 'negative' in label or 'bearish' in label:
                sentiment = 'negative'
                score = result['score']
            else:
                sentiment = 'neutral'
                score = 0.5

            return {
                'label': sentiment,
                'score': score,
                'raw_label': result['label']
            }

        except Exception as e:
            logger.error(f"Error analyzing text sentiment: {e}")
            return {'label': 'neutral', 'score': 0.5}

    def get_sentiment_score(
        self,
        symbol: str,
        use_cache: bool = True
    ) -> float:
        """
        Get aggregated sentiment score for a symbol.

        Args:
            symbol: Trading symbol
            use_cache: Whether to use cached sentiment

        Returns:
            Sentiment score between -1 (very bearish) and 1 (very bullish)
        """
        # Check cache
        if use_cache and symbol in self.sentiment_cache:
            cache_time = self.cache_timestamps.get(symbol)
            if cache_time and (datetime.now() - cache_time).total_seconds() < self.cache_hours * 3600:
                logger.debug(f"Using cached sentiment for {symbol}")
                return self.sentiment_cache[symbol]

        # Fetch news
        articles = self.fetch_news(symbol, days_back=1, max_articles=10)

        if len(articles) == 0:
            logger.warning(f"No news found for {symbol}, returning neutral sentiment")
            return 0.0

        # Analyze each article
        sentiments = []

        for article in articles:
            # Combine title and description for better context
            text = f"{article['title']}. {article['description']}"

            sentiment = self.analyze_text(text)

            # Convert to numeric score
            if sentiment['label'] == 'positive':
                score = sentiment['score']
            elif sentiment['label'] == 'negative':
                score = -sentiment['score']
            else:
                score = 0.0

            sentiments.append(score)

        # Aggregate sentiment scores
        if len(sentiments) > 0:
            # Weighted average (recent news weighted more)
            weights = np.exp(np.linspace(-1, 0, len(sentiments)))
            weights = weights / weights.sum()
            aggregated_score = np.average(sentiments, weights=weights)
        else:
            aggregated_score = 0.0

        # Cache result
        self.sentiment_cache[symbol] = aggregated_score
        self.cache_timestamps[symbol] = datetime.now()

        logger.info(f"Sentiment for {symbol}: {aggregated_score:.3f} "
                   f"(based on {len(articles)} articles)")

        return aggregated_score

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Generate trading probabilities based on sentiment.

        Args:
            df: DataFrame with symbol information

        Returns:
            Probability array with shape (n_samples, 3) for [SELL, HOLD, BUY]
        """
        # Extract symbol from dataframe
        symbol = df.attrs.get('symbol', 'SPY')  # Default to SPY if not specified

        # Get sentiment score
        sentiment = self.get_sentiment_score(symbol)

        # Convert sentiment to probabilities
        n_samples = len(df)
        probas = np.zeros((n_samples, 3))

        if sentiment > 0.3:  # Bullish
            # More bullish = higher buy probability
            buy_prob = 0.5 + (sentiment * 0.4)  # 0.5 to 0.9
            sell_prob = 0.5 - (sentiment * 0.4)  # 0.1 to 0.5
            hold_prob = 1 - buy_prob - sell_prob

            probas[:] = [sell_prob, hold_prob, buy_prob]

        elif sentiment < -0.3:  # Bearish
            # More bearish = higher sell probability
            sell_prob = 0.5 + (-sentiment * 0.4)
            buy_prob = 0.5 - (-sentiment * 0.4)
            hold_prob = 1 - sell_prob - buy_prob

            probas[:] = [sell_prob, hold_prob, buy_prob]

        else:  # Neutral
            probas[:] = [0.25, 0.5, 0.25]

        return probas

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Generate trading signals based on sentiment.

        Args:
            df: Input features

        Returns:
            Signal predictions (0=SELL, 1=HOLD, 2=BUY)
        """
        probas = self.predict_proba(df)
        predictions = np.argmax(probas, axis=1)
        return predictions

    def get_sentiment_features(self, symbol: str) -> Dict:
        """
        Get detailed sentiment features for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Dictionary with sentiment features
        """
        articles = self.fetch_news(symbol, days_back=1, max_articles=20)

        if len(articles) == 0:
            return {
                'sentiment_score': 0.0,
                'sentiment_positive_ratio': 0.5,
                'sentiment_negative_ratio': 0.5,
                'news_volume': 0,
                'sentiment_std': 0.0
            }

        # Analyze all articles
        sentiments = []
        positive_count = 0
        negative_count = 0

        for article in articles:
            text = f"{article['title']}. {article['description']}"
            sentiment = self.analyze_text(text)

            if sentiment['label'] == 'positive':
                score = sentiment['score']
                positive_count += 1
            elif sentiment['label'] == 'negative':
                score = -sentiment['score']
                negative_count += 1
            else:
                score = 0.0

            sentiments.append(score)

        # Calculate features
        sentiments = np.array(sentiments)

        features = {
            'sentiment_score': np.mean(sentiments),
            'sentiment_positive_ratio': positive_count / len(articles) if len(articles) > 0 else 0.5,
            'sentiment_negative_ratio': negative_count / len(articles) if len(articles) > 0 else 0.5,
            'news_volume': len(articles),
            'sentiment_std': np.std(sentiments),
            'sentiment_min': np.min(sentiments),
            'sentiment_max': np.max(sentiments)
        }

        return features

    def save(self, path: str):
        """Save sentiment analyzer metadata."""
        metadata = {
            'is_fitted': self.is_fitted,
            'model_name': self.model_name,
            'cache_hours': self.cache_hours,
            'sentiment_cache': self.sentiment_cache,
            'cache_timestamps': self.cache_timestamps
        }

        with open(path, 'wb') as f:
            pickle.dump(metadata, f)

        logger.info(f"LLM sentiment analyzer metadata saved to {path}")

    def load(self, path: str):
        """Load sentiment analyzer metadata."""
        with open(path, 'rb') as f:
            metadata = pickle.load(f)

        self.is_fitted = metadata['is_fitted']
        self.model_name = metadata['model_name']
        self.cache_hours = metadata['cache_hours']
        self.sentiment_cache = metadata.get('sentiment_cache', {})
        self.cache_timestamps = metadata.get('cache_timestamps', {})

        # Model will be loaded on first use
        logger.info(f"LLM sentiment analyzer metadata loaded from {path}")


# ============================================================================
# SENTIMENT FEATURE ENGINEERING
# ============================================================================

def add_sentiment_features(df: pd.DataFrame, symbol: str, analyzer: LLMSentimentAnalyzer) -> pd.DataFrame:
    """
    Add sentiment features to a dataframe.

    Args:
        df: DataFrame with OHLC data
        symbol: Trading symbol
        analyzer: LLM sentiment analyzer

    Returns:
        DataFrame with additional sentiment features
    """
    logger.info(f"Adding sentiment features for {symbol}...")

    # Get sentiment features
    sentiment_features = analyzer.get_sentiment_features(symbol)

    # Add as columns (broadcast to all rows)
    for key, value in sentiment_features.items():
        df[key] = value

    logger.info(f"Added {len(sentiment_features)} sentiment features")

    return df
