"""
Trading Knowledge Base - Tests

This module contains tests for the Knowledge Base component.
"""

import unittest
import os
import shutil
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

from ..knowledge_base.knowledge_base import KnowledgeBase
from ..knowledge_base.models import MarketData, TradingSignal, Order, PerformanceMetrics
from ..knowledge_base.utils import (
    convert_timeframe_to_timedelta,
    resample_market_data,
    calculate_performance_metrics,
    filter_by_date_range,
    merge_market_data
)

class TestKnowledgeBase(unittest.TestCase):
    """Test cases for the KnowledgeBase class."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = Path(os.path.dirname(os.path.abspath(__file__))) / 'test_data'
        self.test_dir.mkdir(exist_ok=True)
        self.kb = KnowledgeBase(self.test_dir)
    
    def tearDown(self):
        """Clean up test environment."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def test_market_data_store(self):
        """Test storing and retrieving market data."""
        # Create test data
        dates = pd.date_range(start='2023-01-01', periods=10, freq='D')
        data = pd.DataFrame({
            'open': np.random.rand(10) * 100 + 100,
            'high': np.random.rand(10) * 100 + 110,
            'low': np.random.rand(10) * 100 + 90,
            'close': np.random.rand(10) * 100 + 105,
            'volume': np.random.randint(1000, 10000, 10)
        }, index=dates)
        
        # Store data
        self.kb.store_market_data('AAPL', data)
        
        # Retrieve data
        retrieved = self.kb.get_market_data('AAPL')
        
        # Check data integrity
        self.assertEqual(len(retrieved), len(data))
        pd.testing.assert_frame_equal(retrieved, data)
        
        # Test time range filtering
        start_time = dates[2]
        end_time = dates[7]
        filtered = self.kb.get_market_data('AAPL', start_time, end_time)
        self.assertEqual(len(filtered), 6)  # 6 days from index 2 to 7 inclusive
    
    def test_signal_store(self):
        """Test storing and retrieving trading signals."""
        # Create test signal
        signal = {
            'symbol': 'AAPL',
            'direction': 'BUY',
            'strategy': 'momentum',
            'strength': 0.85,
            'timeframe': 'daily',
            'metadata': {'reason': 'Strong uptrend'}
        }
        
        # Store signal
        signal_id = self.kb.store_trading_signal(signal)
        
        # Verify ID was generated
        self.assertIsNotNone(signal_id)
        
        # Retrieve signal
        signals = self.kb.get_trading_signals({'symbol': 'AAPL'})
        
        # Check signal integrity
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0]['symbol'], 'AAPL')
        self.assertEqual(signals[0]['direction'], 'BUY')
        self.assertEqual(signals[0]['strategy'], 'momentum')
        
        # Test filtering
        signals = self.kb.get_trading_signals({'strategy': 'value'})
        self.assertEqual(len(signals), 0)
    
    def test_order_store(self):
        """Test storing and retrieving orders."""
        # Create test order
        order = {
            'symbol': 'AAPL',
            'order_type': 'LIMIT',
            'direction': 'BUY',
            'quantity': 100,
            'price': 150.25,
            'status': 'PENDING'
        }
        
        # Store order
        order_id = self.kb.store_order(order)
        
        # Verify ID was generated
        self.assertIsNotNone(order_id)
        
        # Retrieve order
        orders = self.kb.get_orders({'symbol': 'AAPL'})
        
        # Check order integrity
        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0]['symbol'], 'AAPL')
        self.assertEqual(orders[0]['order_type'], 'LIMIT')
        self.assertEqual(orders[0]['price'], 150.25)
        
        # Test filtering
        orders = self.kb.get_orders({'status': 'FILLED'})
        self.assertEqual(len(orders), 0)
    
    def test_performance_store(self):
        """Test storing and retrieving performance metrics."""
        # Create test metrics
        metrics = {
            'return_pct': 15.7,
            'win_rate': 0.68,
            'sharpe_ratio': 1.8,
            'max_drawdown': 8.5,
            'profit_factor': 2.3,
            'trades_count': 45
        }
        
        # Store metrics
        self.kb.store_strategy_performance('momentum', metrics)
        
        # Retrieve metrics
        retrieved = self.kb.get_strategy_performance('momentum')
        
        # Check metrics integrity
        self.assertEqual(retrieved['return_pct'], 15.7)
        self.assertEqual(retrieved['win_rate'], 0.68)
        self.assertEqual(retrieved['sharpe_ratio'], 1.8)
        
        # Test non-existent strategy
        empty = self.kb.get_strategy_performance('value')
        self.assertEqual(empty, {})
    
    def test_metadata_store(self):
        """Test storing and retrieving metadata."""
        # Store metadata
        self.kb.store_metadata('last_update', '2023-01-15T12:30:45')
        self.kb.store_metadata('version', '1.0.0')
        
        # Retrieve metadata
        last_update = self.kb.get_metadata('last_update')
        version = self.kb.get_metadata('version')
        
        # Check metadata integrity
        self.assertEqual(last_update, '2023-01-15T12:30:45')
        self.assertEqual(version, '1.0.0')
        
        # Test non-existent key
        none_value = self.kb.get_metadata('non_existent')
        self.assertIsNone(none_value)


class TestUtils(unittest.TestCase):
    """Test cases for utility functions."""
    
    def test_convert_timeframe_to_timedelta(self):
        """Test converting timeframe strings to timedelta objects."""
        self.assertEqual(convert_timeframe_to_timedelta('minute'), timedelta(minutes=1))
        self.assertEqual(convert_timeframe_to_timedelta('hourly'), timedelta(hours=1))
        self.assertEqual(convert_timeframe_to_timedelta('daily'), timedelta(days=1))
        self.assertEqual(convert_timeframe_to_timedelta('weekly'), timedelta(days=7))
        self.assertEqual(convert_timeframe_to_timedelta('monthly'), timedelta(days=30))
        
        # Test default for unknown timeframe
        self.assertEqual(convert_timeframe_to_timedelta('unknown'), timedelta(days=1))
    
    def test_resample_market_data(self):
        """Test resampling market data to different timeframes."""
        # Create hourly test data
        dates = pd.date_range(start='2023-01-01', periods=24, freq='H')
        hourly_data = pd.DataFrame({
            'open': np.random.rand(24) * 10 + 100,
            'high': np.random.rand(24) * 10 + 105,
            'low': np.random.rand(24) * 10 + 95,
            'close': np.random.rand(24) * 10 + 102,
            'volume': np.random.randint(100, 1000, 24)
        }, index=dates)
        
        # Resample to daily
        daily_data = resample_market_data(hourly_data, 'daily')
        
        # Check resampling results
        self.assertEqual(len(daily_data), 2)  # 24 hours = 2 days
        self.assertEqual(daily_data.index.freq, pd.tseries.frequencies.to_offset('D'))
        
        # Check aggregation logic
        for day in daily_data.index:
            day_data = hourly_data[hourly_data.index.date == day.date()]
            self.assertEqual(daily_data.loc[day, 'open'], day_data.iloc[0]['open'])
            self.assertEqual(daily_data.loc[day, 'high'], day_data['high'].max())
            self.assertEqual(daily_data.loc[day, 'low'], day_data['low'].min())
            self.assertEqual(daily_data.loc[day, 'close'], day_data.iloc[-1]['close'])
            self.assertEqual(daily_data.loc[day, 'volume'], day_data['volume'].sum())
    
    def test_calculate_performance_metrics(self):
        """Test calculating performance metrics from trades."""
        # Create test trades
        trades = [
            {'profit_loss': 500},
            {'profit_loss': -200},
            {'profit_loss': 300},
            {'profit_loss': 700},
            {'profit_loss': -150}
        ]
        
        # Calculate metrics
        metrics = calculate_performance_metrics(trades)
        
        # Check metric calculations
        self.assertEqual(metrics['trades_count'], 5)
        self.assertEqual(metrics['winning_trades'], 3)
        self.assertEqual(metrics['losing_trades'], 2)
        self.assertAlmostEqual(metrics['win_rate'], 0.6)
        self.assertAlmostEqual(metrics['return_pct'], 1.15)  # (100000 + 1150) / 100000 - 1 = 0.0115 * 100 = 1.15%
        self.assertGreater(metrics['sharpe_ratio'], 0)
        self.assertGreater(metrics['profit_factor'], 1)
        
        # Test empty trades list
        empty_metrics = calculate_performance_metrics([])
        self.assertEqual(empty_metrics['trades_count'], 0)
        self.assertEqual(empty_metrics['return_pct'], 0.0)
    
    def test_filter_by_date_range(self):
        """Test filtering data by date range."""
        # Create test data
        data = [
            {'timestamp': '2023-01-01T12:00:00', 'value': 1},
            {'timestamp': '2023-01-02T12:00:00', 'value': 2},
            {'timestamp': '2023-01-03T12:00:00', 'value': 3},
            {'timestamp': '2023-01-04T12:00:00', 'value': 4},
            {'timestamp': '2023-01-05T12:00:00', 'value': 5}
        ]
        
        # Test start date filter
        start_date = datetime(2023, 1, 3)
        filtered = filter_by_date_range(data, start_date=start_date)
        self.assertEqual(len(filtered), 3)
        self.assertEqual(filtered[0]['value'], 3)
        
        # Test end date filter
        end_date = datetime(2023, 1, 3)
        filtered = filter_by_date_range(data, end_date=end_date)
        self.assertEqual(len(filtered), 3)
        self.assertEqual(filtered[-1]['value'], 3)
        
        # Test both start and end date
        filtered = filter_by_date_range(data, 
                                      start_date=datetime(2023, 1, 2), 
                                      end_date=datetime(2023, 1, 4))
        self.assertEqual(len(filtered), 3)
        self.assertEqual(filtered[0]['value'], 2)
        self.assertEqual(filtered[-1]['value'], 4)
    
    def test_merge_market_data(self):
        """Test merging multiple market data DataFrames."""
        # Create test DataFrames
        dates1 = pd.date_range(start='2023-01-01', periods=5, freq='D')
        df1 = pd.DataFrame({
            'open': np.random.rand(5) * 10 + 100,
            'high': np.random.rand(5) * 10 + 105,
            'low': np.random.rand(5) * 10 + 95,
            'close': np.random.rand(5) * 10 + 102,
            'volume': np.random.randint(100, 1000, 5)
        }, index=dates1)
        
        dates2 = pd.date_range(start='2023-01-03', periods=5, freq='D')
        df2 = pd.DataFrame({
            'open': np.random.rand(5) * 10 + 100,
            'high': np.random.rand(5) * 10 + 105,
            'low': np.random.rand(5) * 10 + 95,
            'close': np.random.rand(5) * 10 + 102,
            'volume': np.random.randint(100, 1000, 5)
        }, index=dates2)
        
        # Merge DataFrames
        merged = merge_market_data([df1, df2])
        
        # Check merged result
        self.assertEqual(len(merged), 7)  # 5 + 5 - 3 overlapping days
        self.assertTrue(all(date in merged.index for date in dates1))
        self.assertTrue(all(date in merged.index for date in dates2))
        
        # Check that overlapping dates use the second DataFrame's values
        for date in dates2:
            pd.testing.assert_series_equal(merged.loc[date], df2.loc[date])


if __name__ == '__main__':
    unittest.main()
