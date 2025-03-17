"""
AI Trading Agent - Tests

This module contains tests for the AI Trading Agent component.
"""

import unittest
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

from ..ai_trading_agent.agent import AITradingAgent, MomentumStrategy, SectorRotationStrategy, SwingTradingStrategy, MeanReversionStrategy
from ..ai_trading_agent.utils import (
    calculate_technical_indicators,
    detect_trend,
    calculate_volatility,
    calculate_z_score,
    calculate_relative_strength,
    detect_support_resistance,
    calculate_risk_reward_ratio,
    calculate_position_size
)
from ..knowledge_base import get_knowledge_base

class TestAITradingAgent(unittest.TestCase):
    """Test cases for the AITradingAgent class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a test data directory
        self.test_dir = Path(os.path.dirname(os.path.abspath(__file__))) / 'test_data'
        self.test_dir.mkdir(exist_ok=True)
        
        # Initialize Knowledge Base with test directory
        self.kb = get_knowledge_base(self.test_dir)
        
        # Create test market data
        self.create_test_market_data()
        
        # Initialize AI Trading Agent
        self.agent = AITradingAgent()
    
    def create_test_market_data(self):
        """Create test market data for testing."""
        # Generate dates for the past 200 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=200)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Create a trending market (AAPL)
        trend_data = pd.DataFrame({
            'open': np.linspace(100, 150, len(dates)) + np.random.normal(0, 1, len(dates)),
            'high': np.linspace(102, 152, len(dates)) + np.random.normal(0, 1, len(dates)),
            'low': np.linspace(98, 148, len(dates)) + np.random.normal(0, 1, len(dates)),
            'close': np.linspace(100, 150, len(dates)) + np.random.normal(0, 1, len(dates)),
            'volume': np.random.randint(1000000, 5000000, len(dates))
        }, index=dates)
        
        # Create a mean-reverting market (XYZ)
        mean = 100
        std = 10
        mean_reversion_data = pd.DataFrame({
            'open': np.random.normal(mean, std, len(dates)),
            'high': np.random.normal(mean, std, len(dates)) + 2,
            'low': np.random.normal(mean, std, len(dates)) - 2,
            'close': np.random.normal(mean, std, len(dates)),
            'volume': np.random.randint(1000000, 5000000, len(dates))
        }, index=dates)
        
        # Create a sector ETF (XLK)
        sector_data = pd.DataFrame({
            'open': np.linspace(80, 120, len(dates)) + np.random.normal(0, 1, len(dates)),
            'high': np.linspace(82, 122, len(dates)) + np.random.normal(0, 1, len(dates)),
            'low': np.linspace(78, 118, len(dates)) + np.random.normal(0, 1, len(dates)),
            'close': np.linspace(80, 120, len(dates)) + np.random.normal(0, 1, len(dates)),
            'volume': np.random.randint(1000000, 5000000, len(dates))
        }, index=dates)
        
        # Store data in Knowledge Base
        self.kb.store_market_data('AAPL', trend_data)
        self.kb.store_market_data('XYZ', mean_reversion_data)
        self.kb.store_market_data('XLK', sector_data)
    
    def test_analyze_market(self):
        """Test market analysis functionality."""
        # Analyze market data
        analysis = self.agent.analyze_market(['AAPL', 'XYZ', 'XLK'])
        
        # Check that analysis contains results for each symbol
        self.assertIn('AAPL', analysis)
        self.assertIn('XYZ', analysis)
        self.assertIn('XLK', analysis)
        
        # Check that analysis contains results for each strategy
        for symbol in analysis:
            self.assertIn('momentum', analysis[symbol])
            self.assertIn('sector_rotation', analysis[symbol])
            self.assertIn('swing_trading', analysis[symbol])
            self.assertIn('mean_reversion', analysis[symbol])
    
    def test_generate_signals(self):
        """Test signal generation functionality."""
        # Analyze market first
        self.agent.analyze_market(['AAPL', 'XYZ', 'XLK'])
        
        # Generate signals
        signals = self.agent.generate_signals()
        
        # Check that signals were generated
        self.assertIsInstance(signals, list)
        
        # Check signal structure if any signals were generated
        if signals:
            signal = signals[0]
            self.assertIn('symbol', signal)
            self.assertIn('direction', signal)
            self.assertIn('strategy', signal)
            self.assertIn('strength', signal)
            self.assertIn('timeframe', signal)
    
    def test_get_active_signals(self):
        """Test retrieving active signals."""
        # Analyze market and generate signals
        self.agent.analyze_market(['AAPL', 'XYZ', 'XLK'])
        self.agent.generate_signals()
        
        # Get active signals
        active_signals = self.agent.get_active_signals()
        
        # Check that active signals is a list
        self.assertIsInstance(active_signals, list)
    
    def test_get_market_view(self):
        """Test retrieving market view."""
        # Analyze market first
        self.agent.analyze_market(['AAPL', 'XYZ', 'XLK'])
        
        # Get market view
        market_view = self.agent.get_market_view()
        
        # Check market view structure
        self.assertIn('timestamp', market_view)
        self.assertIn('symbols', market_view)
        self.assertIn('market_regime', market_view)
        self.assertIn('risk_assessment', market_view)
        
        # Check symbol-specific data
        symbols = market_view['symbols']
        for symbol in ['AAPL', 'XYZ', 'XLK']:
            if symbol in symbols:
                self.assertIn('price', symbols[symbol])
                self.assertIn('trend', symbols[symbol])
                self.assertIn('volatility', symbols[symbol])
                self.assertIn('rsi', symbols[symbol])


class TestTradingStrategies(unittest.TestCase):
    """Test cases for the trading strategy classes."""
    
    def setUp(self):
        """Set up test environment."""
        # Create test data
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        
        # Create a trending market
        self.trend_data = pd.DataFrame({
            'open': np.linspace(100, 150, 100) + np.random.normal(0, 1, 100),
            'high': np.linspace(102, 152, 100) + np.random.normal(0, 1, 100),
            'low': np.linspace(98, 148, 100) + np.random.normal(0, 1, 100),
            'close': np.linspace(100, 150, 100) + np.random.normal(0, 1, 100),
            'volume': np.random.randint(1000000, 5000000, 100)
        }, index=dates)
        
        # Add indicators
        self.trend_data = calculate_technical_indicators(self.trend_data)
        
        # Create a mean-reverting market
        mean = 100
        std = 10
        self.mean_reversion_data = pd.DataFrame({
            'open': np.random.normal(mean, std, 100),
            'high': np.random.normal(mean, std, 100) + 2,
            'low': np.random.normal(mean, std, 100) - 2,
            'close': np.random.normal(mean, std, 100),
            'volume': np.random.randint(1000000, 5000000, 100)
        }, index=dates)
        
        # Add indicators
        self.mean_reversion_data = calculate_technical_indicators(self.mean_reversion_data)
        
        # Initialize strategies
        self.momentum_strategy = MomentumStrategy()
        self.sector_rotation_strategy = SectorRotationStrategy()
        self.swing_trading_strategy = SwingTradingStrategy()
        self.mean_reversion_strategy = MeanReversionStrategy()
    
    def test_momentum_strategy(self):
        """Test momentum strategy."""
        # Analyze data
        analysis = self.momentum_strategy.analyze(self.trend_data)
        
        # Check analysis structure
        self.assertIn('momentum_score', analysis)
        self.assertIn('rsi', analysis)
        self.assertIn('signal', analysis)
        self.assertIn('strength', analysis)
        
        # Generate signals
        signals = self.momentum_strategy.generate_signals('AAPL', self.trend_data)
        
        # Check that signals is a list
        self.assertIsInstance(signals, list)
    
    def test_sector_rotation_strategy(self):
        """Test sector rotation strategy."""
        # Analyze data
        analysis = self.sector_rotation_strategy.analyze(self.trend_data)
        
        # Check analysis structure
        self.assertIn('performance', analysis)
        self.assertIn('relative_strength', analysis)
        self.assertIn('signal', analysis)
        self.assertIn('strength', analysis)
        
        # Generate signals
        signals = self.sector_rotation_strategy.generate_signals('XLK', self.trend_data)
        
        # Check that signals is a list
        self.assertIsInstance(signals, list)
    
    def test_swing_trading_strategy(self):
        """Test swing trading strategy."""
        # Analyze data
        analysis = self.swing_trading_strategy.analyze(self.trend_data)
        
        # Check analysis structure
        self.assertIn('rsi', analysis)
        self.assertIn('bb_upper', analysis)
        self.assertIn('bb_lower', analysis)
        self.assertIn('signal', analysis)
        self.assertIn('strength', analysis)
        
        # Generate signals
        signals = self.swing_trading_strategy.generate_signals('AAPL', self.trend_data)
        
        # Check that signals is a list
        self.assertIsInstance(signals, list)
    
    def test_mean_reversion_strategy(self):
        """Test mean reversion strategy."""
        # Analyze data
        analysis = self.mean_reversion_strategy.analyze(self.mean_reversion_data)
        
        # Check analysis structure
        self.assertIn('z_score', analysis)
        self.assertIn('mean', analysis)
        self.assertIn('std', analysis)
        self.assertIn('signal', analysis)
        self.assertIn('strength', analysis)
        
        # Generate signals
        signals = self.mean_reversion_strategy.generate_signals('XYZ', self.mean_reversion_data)
        
        # Check that signals is a list
        self.assertIsInstance(signals, list)


class TestUtils(unittest.TestCase):
    """Test cases for utility functions."""
    
    def setUp(self):
        """Set up test environment."""
        # Create test data
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        
        self.data = pd.DataFrame({
            'open': np.linspace(100, 150, 100) + np.random.normal(0, 1, 100),
            'high': np.linspace(102, 152, 100) + np.random.normal(0, 1, 100),
            'low': np.linspace(98, 148, 100) + np.random.normal(0, 1, 100),
            'close': np.linspace(100, 150, 100) + np.random.normal(0, 1, 100),
            'volume': np.random.randint(1000000, 5000000, 100)
        }, index=dates)
    
    def test_calculate_technical_indicators(self):
        """Test technical indicator calculation."""
        # Calculate indicators
        data_with_indicators = calculate_technical_indicators(self.data)
        
        # Check that indicators were added
        self.assertIn('sma_20', data_with_indicators.columns)
        self.assertIn('ema_12', data_with_indicators.columns)
        self.assertIn('macd', data_with_indicators.columns)
        self.assertIn('rsi', data_with_indicators.columns)
        self.assertIn('bb_upper', data_with_indicators.columns)
        self.assertIn('atr', data_with_indicators.columns)
    
    def test_detect_trend(self):
        """Test trend detection."""
        # Calculate indicators first
        data_with_indicators = calculate_technical_indicators(self.data)
        
        # Detect trend
        trend = detect_trend(data_with_indicators)
        
        # Check that trend is one of the expected values
        self.assertIn(trend, ['bullish', 'bearish', 'neutral'])
    
    def test_calculate_volatility(self):
        """Test volatility calculation."""
        # Calculate volatility
        volatility = calculate_volatility(self.data)
        
        # Check that volatility is a positive number
        self.assertGreaterEqual(volatility, 0)
    
    def test_calculate_z_score(self):
        """Test Z-score calculation."""
        # Calculate Z-score
        z_score = calculate_z_score(self.data['close'])
        
        # Check that Z-score is a number
        self.assertIsInstance(z_score, float)
    
    def test_detect_support_resistance(self):
        """Test support and resistance detection."""
        # Detect support and resistance
        levels = detect_support_resistance(self.data)
        
        # Check that levels contains support and resistance lists
        self.assertIn('support', levels)
        self.assertIn('resistance', levels)
        self.assertIsInstance(levels['support'], list)
        self.assertIsInstance(levels['resistance'], list)
    
    def test_calculate_risk_reward_ratio(self):
        """Test risk-reward ratio calculation."""
        # Calculate risk-reward ratio for a long position
        ratio_long = calculate_risk_reward_ratio(100, 110, 95)
        
        # Check that ratio is correct
        self.assertAlmostEqual(ratio_long, 2.0)
        
        # Calculate risk-reward ratio for a short position
        ratio_short = calculate_risk_reward_ratio(100, 90, 105)
        
        # Check that ratio is correct
        self.assertAlmostEqual(ratio_short, 2.0)
    
    def test_calculate_position_size(self):
        """Test position size calculation."""
        # Calculate position size for a long position
        size_long = calculate_position_size(100000, 0.01, 100, 95)
        
        # Check that size is correct
        self.assertAlmostEqual(size_long, 200.0)
        
        # Calculate position size for a short position
        size_short = calculate_position_size(100000, 0.01, 100, 105)
        
        # Check that size is correct
        self.assertAlmostEqual(size_short, 200.0)


if __name__ == '__main__':
    unittest.main()
