"""
Data Analysis and Web Monitoring - Tests

This module contains tests for the Data Analysis and Web Monitoring component.
"""

import unittest
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import time

from ..data_analysis import get_data_analysis_system
from ..knowledge_base import get_knowledge_base

class TestDataAnalysisSystem(unittest.TestCase):
    """Test cases for the DataAnalysisSystem class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a test data directory
        self.test_dir = Path(os.path.dirname(os.path.abspath(__file__))) / 'test_data'
        self.test_dir.mkdir(exist_ok=True)
        
        # Initialize Knowledge Base with test directory
        self.kb = get_knowledge_base(self.test_dir)
        
        # Initialize Data Analysis System
        self.data_analysis = get_data_analysis_system(data_update_interval=60)
    
    def test_get_market_data(self):
        """Test retrieving market data."""
        # Get market data for a symbol
        symbol = 'AAPL'
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        data = self.data_analysis.get_market_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            interval='daily'
        )
        
        # Check that data was retrieved
        self.assertFalse(data.empty)
        self.assertIn('open', data.columns)
        self.assertIn('high', data.columns)
        self.assertIn('low', data.columns)
        self.assertIn('close', data.columns)
        self.assertIn('volume', data.columns)
    
    def test_get_news(self):
        """Test retrieving news."""
        # Get news for a symbol
        symbol = 'AAPL'
        days = 7
        
        news = self.data_analysis.get_news(
            symbol=symbol,
            days=days
        )
        
        # Check that news was retrieved
        self.assertIsInstance(news, list)
        if news:  # News might be empty if no news available
            self.assertIn('symbol', news[0])
            self.assertIn('title', news[0])
            self.assertIn('timestamp', news[0])
    
    def test_get_social_sentiment(self):
        """Test retrieving social sentiment."""
        # Get social sentiment for a symbol
        symbol = 'AAPL'
        days = 7
        
        sentiment = self.data_analysis.get_social_sentiment(
            symbol=symbol,
            days=days
        )
        
        # Check that sentiment was retrieved
        self.assertIsInstance(sentiment, dict)
        if sentiment:  # Sentiment might be empty if no data available
            self.assertIn('symbol', sentiment)
            self.assertIn('data', sentiment)
    
    def test_get_economic_indicators(self):
        """Test retrieving economic indicators."""
        # Get economic indicators
        indicators = ['GDP', 'CPI', 'UNEMPLOYMENT']
        days = 30
        
        data = self.data_analysis.get_economic_indicators(
            indicators=indicators,
            days=days
        )
        
        # Check that indicators were retrieved
        self.assertIsInstance(data, dict)
        for indicator in indicators:
            self.assertIn(indicator, data)
            self.assertFalse(data[indicator].empty)
    
    def test_get_sec_filings(self):
        """Test retrieving SEC filings."""
        # Get SEC filings for a symbol
        symbol = 'AAPL'
        filing_types = ['10-K', '10-Q', '8-K']
        days = 90
        
        filings = self.data_analysis.get_sec_filings(
            symbol=symbol,
            filing_types=filing_types,
            days=days
        )
        
        # Check that filings were retrieved
        self.assertIsInstance(filings, list)
        if filings:  # Filings might be empty if no filings available
            self.assertIn('symbol', filings[0])
            self.assertIn('type', filings[0])
            self.assertIn('date', filings[0])
    
    def test_analyze_technical_indicators(self):
        """Test analyzing technical indicators."""
        # Analyze technical indicators for a symbol
        symbol = 'AAPL'
        days = 100
        
        analysis = self.data_analysis.analyze_technical_indicators(
            symbol=symbol,
            days=days
        )
        
        # Check that analysis was performed
        self.assertIsInstance(analysis, dict)
        if analysis:  # Analysis might be empty if no data available
            self.assertIn('trend', analysis)
            self.assertIn('momentum', analysis)
            self.assertIn('overbought_oversold', analysis)
            self.assertIn('indicators', analysis)
    
    def test_analyze_fundamental_data(self):
        """Test analyzing fundamental data."""
        # Analyze fundamental data for a symbol
        symbol = 'AAPL'
        
        analysis = self.data_analysis.analyze_fundamental_data(
            symbol=symbol
        )
        
        # Check that analysis was performed
        self.assertIsInstance(analysis, dict)
        if analysis:  # Analysis might be empty if no data available
            self.assertIn('symbol', analysis)
            self.assertIn('valuation', analysis)
            self.assertIn('growth', analysis)
            self.assertIn('profitability', analysis)
    
    def test_analyze_sentiment(self):
        """Test analyzing sentiment."""
        # Analyze sentiment for a symbol
        symbol = 'AAPL'
        days = 7
        
        analysis = self.data_analysis.analyze_sentiment(
            symbol=symbol,
            days=days
        )
        
        # Check that analysis was performed
        self.assertIsInstance(analysis, dict)
        if analysis:  # Analysis might be empty if no data available
            self.assertIn('symbol', analysis)
            self.assertIn('combined_sentiment', analysis)
            self.assertIn('sentiment_category', analysis)
    
    def test_analyze_correlations(self):
        """Test analyzing correlations."""
        # Analyze correlations between symbols
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        days = 100
        
        analysis = self.data_analysis.analyze_correlations(
            symbols=symbols,
            days=days
        )
        
        # Check that analysis was performed
        self.assertIsInstance(analysis, dict)
        if analysis:  # Analysis might be empty if no data available
            self.assertIn('correlation_matrix', analysis)
            self.assertIn('betas', analysis)
            self.assertIn('symbols', analysis)
    
    def test_analyze_volatility(self):
        """Test analyzing volatility."""
        # Analyze volatility for symbols
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        days = 100
        
        analysis = self.data_analysis.analyze_volatility(
            symbols=symbols,
            days=days
        )
        
        # Check that analysis was performed
        self.assertIsInstance(analysis, dict)
        if analysis:  # Analysis might be empty if no data available
            self.assertIn('volatility', analysis)
            self.assertIn('average_volatility', analysis)
            self.assertIn('lowest_volatility', analysis)
            self.assertIn('highest_volatility', analysis)
    
    def test_schedule_data_update(self):
        """Test scheduling data updates."""
        # Schedule a data update
        data_type = 'market_data'
        params = {
            'symbol': 'AAPL',
            'start_date': datetime.now() - timedelta(days=30),
            'end_date': datetime.now(),
            'interval': 'daily'
        }
        
        update_id = self.data_analysis.schedule_data_update(
            data_type=data_type,
            params=params,
            interval=3600
        )
        
        # Check that update was scheduled
        self.assertNotEqual(update_id, "")
        self.assertIn(update_id, self.data_analysis.scheduled_updates)
    
    def test_cancel_data_update(self):
        """Test cancelling data updates."""
        # Schedule a data update
        data_type = 'market_data'
        params = {
            'symbol': 'AAPL',
            'start_date': datetime.now() - timedelta(days=30),
            'end_date': datetime.now(),
            'interval': 'daily'
        }
        
        update_id = self.data_analysis.schedule_data_update(
            data_type=data_type,
            params=params,
            interval=3600
        )
        
        # Cancel the update
        result = self.data_analysis.cancel_data_update(update_id)
        
        # Check that update was cancelled
        self.assertTrue(result)
        self.assertNotIn(update_id, self.data_analysis.scheduled_updates)
    
    def tearDown(self):
        """Clean up after tests."""
        # Shutdown data analysis system
        self.data_analysis.shutdown()


class TestDataSources(unittest.TestCase):
    """Test cases for the data sources."""
    
    def setUp(self):
        """Set up test environment."""
        # Initialize Data Analysis System
        self.data_analysis = get_data_analysis_system(data_update_interval=60)
    
    def test_market_data_source(self):
        """Test market data source."""
        # Get market data source
        market_data_source = self.data_analysis.data_sources['market_data']
        
        # Get data from source
        symbol = 'AAPL'
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        data = market_data_source.get_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            interval='daily'
        )
        
        # Check that data was retrieved
        self.assertFalse(data.empty)
        self.assertIn('open', data.columns)
        self.assertIn('high', data.columns)
        self.assertIn('low', data.columns)
        self.assertIn('close', data.columns)
        self.assertIn('volume', data.columns)
    
    def test_news_data_source(self):
        """Test news data source."""
        # Get news data source
        news_data_source = self.data_analysis.data_sources['news']
        
        # Get data from source
        symbol = 'AAPL'
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        data = news_data_source.get_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date
        )
        
        # Check that data was retrieved
        self.assertIsInstance(data, list)
        if data:  # Data might be empty if no news available
            self.assertIn('symbol', data[0])
            self.assertIn('title', data[0])
            self.assertIn('timestamp', data[0])
    
    def test_social_sentiment_data_source(self):
        """Test social sentiment data source."""
        # Get social sentiment data source
        social_sentiment_data_source = self.data_analysis.data_sources['social_sentiment']
        
        # Get data from source
        symbol = 'AAPL'
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        data = social_sentiment_data_source.get_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date
        )
        
        # Check that data was retrieved
        self.assertIsInstance(data, dict)
        if data:  # Data might be empty if no sentiment available
            self.assertIn('symbol', data)
            self.assertIn('data', data)
    
    def test_economic_indicators_data_source(self):
        """Test economic indicators data source."""
        # Get economic indicators data source
        economic_indicators_data_source = self.data_analysis.data_sources['economic_indicators']
        
        # Get data from source
        indicator = 'GDP'
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        data = economic_indicators_data_source.get_data(
            indicator=indicator,
            start_date=start_date,
            end_date=end_date
        )
        
        # Check that data was retrieved
        self.assertFalse(data.empty)
        self.assertIn('value', data.columns)
        self.assertIn('unit', data.columns)
    
    def test_sec_filings_data_source(self):
        """Test SEC filings data source."""
        # Get SEC filings data source
        sec_filings_data_source = self.data_analysis.data_sources['sec_filings']
        
        # Get data from source
        symbol = 'AAPL'
        filing_types = ['10-K', '10-Q', '8-K']
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)
        
        data = sec_filings_data_source.get_data(
            symbol=symbol,
            filing_types=filing_types,
            start_date=start_date,
            end_date=end_date
        )
        
        # Check that data was retrieved
        self.assertIsInstance(data, list)
        if data:  # Data might be empty if no filings available
            self.assertIn('symbol', data[0])
            self.assertIn('type', data[0])
            self.assertIn('date', data[0])
    
    def tearDown(self):
        """Clean up after tests."""
        # Shutdown data analysis system
        self.data_analysis.shutdown()


class TestDataProcessors(unittest.TestCase):
    """Test cases for the data processors."""
    
    def setUp(self):
        """Set up test environment."""
        # Initialize Data Analysis System
        self.data_analysis = get_data_analysis_system(data_update_interval=60)
    
    def test_technical_analysis_processor(self):
        """Test technical analysis processor."""
        # Get technical analysis processor
        technical_analysis_processor = self.data_analysis.data_processors['technical_analysis']
        
        # Create test data
        dates = pd.date_range(start=datetime.now() - timedelta(days=100), end=datetime.now(), freq='D')
        
        # Generate random data
        base_price = 100.0
        volatility = 0.02
        
        opens = [base_price]
        highs = [base_price * (1 + np.random.uniform(0, volatility))]
        lows = [base_price * (1 - np.random.uniform(0, volatility))]
        closes = [base_price * (1 + np.random.normal(0, volatility))]
        
        for i in range(1, len(dates)):
            prev_close = closes[i-1]
            daily_return = np.random.normal(0.0005, volatility)
            
            open_price = prev_close * (1 + np.random.normal(0, volatility/2))
            close_price = open_price * (1 + daily_return)
            high_price = max(open_price, close_price) * (1 + np.random.uniform(0, volatility))
            low_price = min(open_price, close_price) * (1 - np.random.uniform(0, volatility))
            
            opens.append(open_price)
            highs.append(high_price)
            lows.append(low_price)
            closes.append(close_price)
        
        volumes = np.random.randint(100000, 1000000, size=len(dates))
        
        # Create DataFrame
        df = pd.DataFrame({
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes,
            'adjclose': closes  # Simplified
        }, index=dates)
        
        # Process data
        result = technical_analysis_processor.process_data(df)
        
        # Check that processing was performed
        self.assertIsInstance(result, dict)
        self.assertIn('trend', result)
        self.assertIn('momentum', result)
        self.assertIn('overbought_oversold', result)
        self.assertIn('indicators', result)
    
    def test_fundamental_analysis_processor(self):
        """Test fundamental analysis processor."""
        # Get fundamental analysis processor
        fundamental_analysis_processor = self.data_analysis.data_processors['fundamental_analysis']
        
        # Create test data
        data = {
            'symbol': 'AAPL',
            'filings': [
                {
                    'symbol': 'AAPL',
                    'type': '10-K',
                    'title': 'Annual Report',
                    'date': datetime.now().isoformat(),
                    'url': 'https://www.sec.gov/edgar/search/'
                }
            ]
        }
        
        # Process data
        result = fundamental_analysis_processor.process_data(data)
        
        # Check that processing was performed
        self.assertIsInstance(result, dict)
        self.assertIn('symbol', result)
        self.assertIn('valuation', result)
        self.assertIn('growth', result)
        self.assertIn('profitability', result)
        self.assertIn('financial_health', result)
    
    def test_sentiment_analysis_processor(self):
        """Test sentiment analysis processor."""
        # Get sentiment analysis processor
        sentiment_analysis_processor = self.data_analysis.data_processors['sentiment_analysis']
        
        # Create test data
        data = {
            'symbol': 'AAPL',
            'news': [
                {
                    'symbol': 'AAPL',
                    'title': 'Positive news about AAPL',
                    'source': 'Test Source',
                    'author': 'Test Author',
                    'url': 'https://example.com',
                    'summary': 'This is a positive test news item about AAPL.',
                    'timestamp': datetime.now().isoformat(),
                    'sentiment': 0.5
                }
            ],
            'social_sentiment': {
                'symbol': 'AAPL',
                'source': 'social_media',
                'data': [
                    {
                        'date': datetime.now().strftime('%Y-%m-%d'),
                        'sentiment': 0.3,
                        'volume': 500
                    }
                ],
                'summary': {
                    'average_sentiment': 0.3,
                    'sentiment_trend': 'positive',
                    'volume_trend': 'increasing'
                }
            }
        }
        
        # Process data
        result = sentiment_analysis_processor.process_data(data)
        
        # Check that processing was performed
        self.assertIsInstance(result, dict)
        self.assertIn('symbol', result)
        self.assertIn('combined_sentiment', result)
        self.assertIn('sentiment_category', result)
        self.assertIn('news_sentiment', result)
        self.assertIn('social_sentiment', result)
    
    def test_correlation_analysis_processor(self):
        """Test correlation analysis processor."""
        # Get correlation analysis processor
        correlation_analysis_processor = self.data_analysis.data_processors['correlation_analysis']
        
        # Create test data
        dates = pd.date_range(start=datetime.now() - timedelta(days=100), end=datetime.now(), freq='D')
        
        data = {}
        
        for symbol in ['AAPL', 'MSFT', 'GOOGL']:
            # Generate random data
            base_price = 100.0
            volatility = 0.02
            
            opens = [base_price]
            highs = [base_price * (1 + np.random.uniform(0, volatility))]
            lows = [base_price * (1 - np.random.uniform(0, volatility))]
            closes = [base_price * (1 + np.random.normal(0, volatility))]
            
            for i in range(1, len(dates)):
                prev_close = closes[i-1]
                daily_return = np.random.normal(0.0005, volatility)
                
                open_price = prev_close * (1 + np.random.normal(0, volatility/2))
                close_price = open_price * (1 + daily_return)
                high_price = max(open_price, close_price) * (1 + np.random.uniform(0, volatility))
                low_price = min(open_price, close_price) * (1 - np.random.uniform(0, volatility))
                
                opens.append(open_price)
                highs.append(high_price)
                lows.append(low_price)
                closes.append(close_price)
            
            volumes = np.random.randint(100000, 1000000, size=len(dates))
            
            # Create DataFrame
            df = pd.DataFrame({
                'open': opens,
                'high': highs,
                'low': lows,
                'close': closes,
                'volume': volumes,
                'adjclose': closes  # Simplified
            }, index=dates)
            
            data[symbol] = df
        
        # Process data
        result = correlation_analysis_processor.process_data(data)
        
        # Check that processing was performed
        self.assertIsInstance(result, dict)
        self.assertIn('correlation_matrix', result)
        self.assertIn('betas', result)
        self.assertIn('symbols', result)
    
    def test_volatility_analysis_processor(self):
        """Test volatility analysis processor."""
        # Get volatility analysis processor
        volatility_analysis_processor = self.data_analysis.data_processors['volatility_analysis']
        
        # Create test data
        dates = pd.date_range(start=datetime.now() - timedelta(days=100), end=datetime.now(), freq='D')
        
        data = {}
        
        for symbol in ['AAPL', 'MSFT', 'GOOGL']:
            # Generate random data
            base_price = 100.0
            volatility = 0.02
            
            opens = [base_price]
            highs = [base_price * (1 + np.random.uniform(0, volatility))]
            lows = [base_price * (1 - np.random.uniform(0, volatility))]
            closes = [base_price * (1 + np.random.normal(0, volatility))]
            
            for i in range(1, len(dates)):
                prev_close = closes[i-1]
                daily_return = np.random.normal(0.0005, volatility)
                
                open_price = prev_close * (1 + np.random.normal(0, volatility/2))
                close_price = open_price * (1 + daily_return)
                high_price = max(open_price, close_price) * (1 + np.random.uniform(0, volatility))
                low_price = min(open_price, close_price) * (1 - np.random.uniform(0, volatility))
                
                opens.append(open_price)
                highs.append(high_price)
                lows.append(low_price)
                closes.append(close_price)
            
            volumes = np.random.randint(100000, 1000000, size=len(dates))
            
            # Create DataFrame
            df = pd.DataFrame({
                'open': opens,
                'high': highs,
                'low': lows,
                'close': closes,
                'volume': volumes,
                'adjclose': closes  # Simplified
            }, index=dates)
            
            data[symbol] = df
        
        # Process data
        result = volatility_analysis_processor.process_data(data)
        
        # Check that processing was performed
        self.assertIsInstance(result, dict)
        self.assertIn('volatility', result)
        self.assertIn('average_volatility', result)
        self.assertIn('lowest_volatility', result)
        self.assertIn('highest_volatility', result)
    
    def tearDown(self):
        """Clean up after tests."""
        # Shutdown data analysis system
        self.data_analysis.shutdown()


if __name__ == '__main__':
    unittest.main()
