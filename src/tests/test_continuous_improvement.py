import unittest
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the module to test
from continuous_improvement.continuous_improvement import (
    ContinuousImprovementSystem,
    SignalQualityModel,
    PositionSizingModel,
    ExitTimingModel,
    MarketRegimeModel,
    StrategySelectionModel,
    PerformanceMetrics,
    StrategyOptimizer,
    ParameterTuner,
    FeedbackLoop
)

class TestContinuousImprovementSystem(unittest.TestCase):
    """Test cases for the Continuous Improvement System."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock dependencies
        self.mock_kb = MagicMock()
        self.mock_ai_agent = MagicMock()
        self.mock_trading_system = MagicMock()
        self.mock_data_analysis = MagicMock()
        
        # Create temporary directory for models
        self.test_models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_models')
        os.makedirs(self.test_models_dir, exist_ok=True)
        
        # Create patches for dependencies
        self.kb_patch = patch('continuous_improvement.continuous_improvement.get_knowledge_base', 
                             return_value=self.mock_kb)
        self.ai_agent_patch = patch('continuous_improvement.continuous_improvement.get_ai_trading_agent', 
                                   return_value=self.mock_ai_agent)
        self.trading_system_patch = patch('continuous_improvement.continuous_improvement.get_algorithmic_trading_system', 
                                         return_value=self.mock_trading_system)
        self.data_analysis_patch = patch('continuous_improvement.continuous_improvement.get_data_analysis_system', 
                                        return_value=self.mock_data_analysis)
        
        # Start patches
        self.kb_patch.start()
        self.ai_agent_patch.start()
        self.trading_system_patch.start()
        self.data_analysis_patch.start()
        
        # Create system instance
        self.ci_system = ContinuousImprovementSystem(
            models_dir=self.test_models_dir,
            performance_review_interval=3600  # 1 hour for testing
        )
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Stop patches
        self.kb_patch.stop()
        self.ai_agent_patch.stop()
        self.trading_system_patch.stop()
        self.data_analysis_patch.stop()
        
        # Clean up temporary directory
        for file in os.listdir(self.test_models_dir):
            os.remove(os.path.join(self.test_models_dir, file))
        os.rmdir(self.test_models_dir)
    
    def test_initialization(self):
        """Test system initialization."""
        self.assertIsNotNone(self.ci_system)
        self.assertEqual(self.ci_system.performance_review_interval, 3600)
        self.assertEqual(str(self.ci_system.models_dir), self.test_models_dir)
        
        # Check if learning models are initialized
        self.assertIn('signal_quality', self.ci_system.learning_models)
        self.assertIn('position_sizing', self.ci_system.learning_models)
        self.assertIn('exit_timing', self.ci_system.learning_models)
        self.assertIn('market_regime', self.ci_system.learning_models)
        self.assertIn('strategy_selection', self.ci_system.learning_models)
    
    def test_evaluate_signal_quality(self):
        """Test signal quality evaluation."""
        # Create test signal
        signal = {
            'symbol': 'AAPL',
            'strategy': 'momentum',
            'direction': 'BUY',
            'strength': 0.7,
            'timeframe': 'daily',
            'metadata': {
                'rsi': 65,
                'momentum_score': 0.8,
                'volatility': 0.15,
                'volume_ratio': 1.2
            }
        }
        
        # Test evaluation
        quality_score = self.ci_system.evaluate_signal_quality(signal)
        
        # Since model is not trained, should return default value
        self.assertEqual(quality_score, 0.5)
    
    def test_optimize_position_size(self):
        """Test position size optimization."""
        # Create test signal and account info
        signal = {
            'symbol': 'AAPL',
            'strategy': 'momentum',
            'direction': 'BUY',
            'strength': 0.7,
            'timeframe': 'daily',
            'metadata': {
                'rsi': 65,
                'momentum_score': 0.8,
                'volatility': 0.15,
                'volume_ratio': 1.2
            }
        }
        
        account_info = {
            'net_liquidation_value': 100000,
            'buying_power': 200000,
            'cash_balance': 50000,
            'day_trades_remaining': 3,
            'leverage': 1.0
        }
        
        # Test optimization
        position_size = self.ci_system.optimize_position_size(signal, account_info)
        
        # Since model is not trained, should return default value
        self.assertEqual(position_size, 0.05)
    
    def test_recommend_exit_timing(self):
        """Test exit timing recommendation."""
        # Create test position and market data
        position = {
            'symbol': 'AAPL',
            'strategy': 'momentum',
            'direction': 'BUY',
            'average_cost': 150.0,
            'days_held': 5,
            'size_pct': 0.05
        }
        
        # Create sample market data
        dates = pd.date_range(end=datetime.now(), periods=30)
        market_data = pd.DataFrame({
            'open': np.random.uniform(145, 155, 30),
            'high': np.random.uniform(150, 160, 30),
            'low': np.random.uniform(140, 150, 30),
            'close': np.random.uniform(145, 155, 30),
            'volume': np.random.uniform(1000000, 5000000, 30)
        }, index=dates)
        
        # Test recommendation
        recommendation = self.ci_system.recommend_exit_timing(position, market_data)
        
        # Check recommendation structure
        self.assertIn('symbol', recommendation)
        self.assertIn('exit_probability', recommendation)
        self.assertIn('optimal_exit_days', recommendation)
        self.assertIn('timestamp', recommendation)
        
        # Since model is not trained, should return default values
        self.assertEqual(recommendation['symbol'], 'AAPL')
        self.assertEqual(recommendation['exit_probability'], 0.5)
        self.assertEqual(recommendation['optimal_exit_days'], 5.0)
    
    def test_detect_market_regime(self):
        """Test market regime detection."""
        # Create sample market data
        dates = pd.date_range(end=datetime.now(), periods=30)
        spy_data = pd.DataFrame({
            'open': np.random.uniform(400, 410, 30),
            'high': np.random.uniform(405, 415, 30),
            'low': np.random.uniform(395, 405, 30),
            'close': np.random.uniform(400, 410, 30),
            'volume': np.random.uniform(10000000, 50000000, 30)
        }, index=dates)
        
        qqq_data = pd.DataFrame({
            'open': np.random.uniform(300, 310, 30),
            'high': np.random.uniform(305, 315, 30),
            'low': np.random.uniform(295, 305, 30),
            'close': np.random.uniform(300, 310, 30),
            'volume': np.random.uniform(8000000, 40000000, 30)
        }, index=dates)
        
        market_data = {
            'SPY': spy_data,
            'QQQ': qqq_data
        }
        
        # Test detection
        regime_info = self.ci_system.detect_market_regime(market_data)
        
        # Check regime info structure
        self.assertIn('regime', regime_info)
        self.assertIn('confidence', regime_info)
        self.assertIn('timestamp', regime_info)
    
    def test_select_optimal_strategy(self):
        """Test optimal strategy selection."""
        # Create test symbol and market regime
        symbol = 'AAPL'
        market_regime = {
            'regime': 'bullish',
            'confidence': 0.8,
            'timestamp': datetime.now().isoformat()
        }
        
        # Test selection
        strategy = self.ci_system.select_optimal_strategy(symbol, market_regime)
        
        # Since model is not trained, should use heuristic approach
        self.assertIn(strategy, ['momentum', 'sector_rotation', 'swing_trading', 'mean_reversion'])
    
    def test_calculate_performance_metrics(self):
        """Test performance metrics calculation."""
        # Create sample orders and positions
        start_date = datetime.now() - timedelta(days=30)
        end_date = datetime.now()
        
        orders = [
            {
                'symbol': 'AAPL',
                'strategy': 'momentum',
                'direction': 'BUY',
                'quantity': 10,
                'price': 150.0,
                'timestamp': (start_date + timedelta(days=1)).isoformat(),
                'pnl': 500.0
            },
            {
                'symbol': 'MSFT',
                'strategy': 'sector_rotation',
                'direction': 'BUY',
                'quantity': 5,
                'price': 300.0,
                'timestamp': (start_date + timedelta(days=5)).isoformat(),
                'pnl': -200.0
            },
            {
                'symbol': 'GOOGL',
                'strategy': 'swing_trading',
                'direction': 'BUY',
                'quantity': 2,
                'price': 2500.0,
                'timestamp': (start_date + timedelta(days=10)).isoformat(),
                'pnl': 300.0
            }
        ]
        
        positions = [
            {
                'symbol': 'AAPL',
                'quantity': 10,
                'average_cost': 150.0,
                'market_value': 1600.0
            },
            {
                'symbol': 'AMZN',
                'quantity': 3,
                'average_cost': 3300.0,
                'market_value': 10200.0
            }
        ]
        
        # Mock Knowledge Base methods
        self.mock_kb.get_orders.return_value = orders
        self.mock_kb.get_positions.return_value = positions
        
        # Test metrics calculation
        metrics = self.ci_system.calculate_performance_metrics(start_date, end_date)
        
        # Check metrics structure
        self.assertIn('start_date', metrics)
        self.assertIn('end_date', metrics)
        self.assertIn('total_trades', metrics)
        self.assertIn('winning_trades', metrics)
        self.assertIn('losing_trades', metrics)
        self.assertIn('win_rate', metrics)
        self.assertIn('total_return', metrics)
        
        # Check specific values
        self.assertEqual(metrics['total_trades'], 3)
        self.assertEqual(metrics['winning_trades'], 2)
        self.assertEqual(metrics['losing_trades'], 1)
        self.assertAlmostEqual(metrics['win_rate'], 2/3)
        self.assertEqual(metrics['total_return'], 600.0)
    
    def test_process_trade_feedback(self):
        """Test trade feedback processing."""
        # Create test trade result
        trade_result = {
            'symbol': 'AAPL',
            'strategy': 'momentum',
            'entry_price': 150.0,
            'exit_price': 160.0,
            'entry_time': (datetime.now() - timedelta(days=5)).isoformat(),
            'exit_time': datetime.now().isoformat(),
            'position_size': 0.05,
            'pnl': 500.0,
            'pnl_pct': 0.0667
        }
        
        # Test feedback processing
        self.ci_system.process_trade_feedback(trade_result)
        
        # Since this is mostly a logging operation, just verify it doesn't raise exceptions
        pass
    
    def test_shutdown(self):
        """Test system shutdown."""
        # Test shutdown
        self.ci_system.shutdown()
        
        # Verify running flag is set to False
        self.assertFalse(self.ci_system.running)


class TestLearningModels(unittest.TestCase):
    """Test cases for the learning models."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for models
        self.test_models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_models')
        os.makedirs(self.test_models_dir, exist_ok=True)
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Clean up temporary directory
        for file in os.listdir(self.test_models_dir):
            os.remove(os.path.join(self.test_models_dir, file))
        os.rmdir(self.test_models_dir)
    
    def test_signal_quality_model(self):
        """Test signal quality model."""
        # Create model
        model = SignalQualityModel(self.test_models_dir)
        
        # Create test signal
        signal = {
            'symbol': 'AAPL',
            'strategy': 'momentum',
            'direction': 'BUY',
            'strength': 0.7,
            'timeframe': 'daily',
            'metadata': {
                'rsi': 65,
                'momentum_score': 0.8,
                'volatility': 0.15,
                'volume_ratio': 1.2
            }
        }
        
        # Test feature preparation
        features = model.prepare_features(signal)
        
        # Check feature shape
        self.assertEqual(features.shape, (1, 8))
        
        # Test prediction
        quality_score = model.predict(features)
        
        # Since model is not trained, should return default value
        self.assertEqual(quality_score, 0.5)
        
        # Test update
        feedback_data = {
            'symbol': 'AAPL',
            'strategy': 'momentum',
            'entry_price': 150.0,
            'exit_price': 160.0,
            'entry_time': (datetime.now() - timedelta(days=5)).isoformat(),
            'exit_time': datetime.now().isoformat(),
            'position_size': 0.05,
            'pnl': 500.0,
            'pnl_pct': 0.0667
        }
        
        # Mock Knowledge Base
        model.kb = MagicMock()
        model.kb.get_signal_by_id.return_value = signal
        
        # Update model
        model.update(feedback_data)
        
        # Check if training data was added
        self.assertEqual(len(model.training_data), 1)
    
    def test_position_sizing_model(self):
        """Test position sizing model."""
        # Create model
        model = PositionSizingModel(self.test_models_dir)
        
        # Create test signal and account info
        signal = {
            'symbol': 'AAPL',
            'strategy': 'momentum',
            'direction': 'BUY',
            'strength': 0.7,
            'timeframe': 'daily',
            'metadata': {
                'rsi': 65,
                'momentum_score': 0.8,
                'volatility': 0.15,
                'volume_ratio': 1.2
            }
        }
        
        account_info = {
            'net_liquidation_value': 100000,
            'buying_power': 200000,
            'cash_balance': 50000,
            'day_trades_remaining': 3,
            'leverage': 1.0
        }
        
        # Test feature preparation
        features = model.prepare_features(signal, account_info)
        
        # Check feature shape
        self.assertEqual(features.shape, (1, 13))
        
        # Test prediction
        position_size = model.predict(features)
        
        # Since model is not trained, should return default value
        self.assertEqual(position_size, 0.05)
    
    def test_exit_timing_model(self):
        """Test exit timing model."""
        # Create model
        model = ExitTimingModel(self.test_models_dir)
        
        # Create test position and market data
        position = {
            'symbol': 'AAPL',
            'strategy': 'momentum',
            'direction': 'BUY',
            'average_cost': 150.0,
            'days_held': 5,
            'size_pct': 0.05
        }
        
        # Create sample market data
        dates = pd.date_range(end=datetime.now(), periods=30)
        market_data = pd.DataFrame({
            'open': np.random.uniform(145, 155, 30),
            'high': np.random.uniform(150, 160, 30),
            'low': np.random.uniform(140, 150, 30),
            'close': np.random.uniform(145, 155, 30),
            'volume': np.random.uniform(1000000, 5000000, 30)
        }, index=dates)
        
        # Test feature preparation
        features = model.prepare_features(position, market_data)
        
        # Check feature shape
        self.assertEqual(features.shape, (1, 10))
        
        # Test prediction
        exit_prob, exit_days = model.predict(features)
        
        # Since model is not trained, should return default values
        self.assertEqual(exit_prob, 0.5)
        self.assertEqual(exit_days, 5.0)
    
    def test_market_regime_model(self):
        """Test market regime model."""
        # Create model
        model = MarketRegimeModel(self.test_models_dir)
        
        # Create sample market data
        dates = pd.date_range(end=datetime.now(), periods=30)
        spy_data = pd.DataFrame({
            'open': np.random.uniform(400, 410, 30),
            'high': np.random.uniform(405, 415, 30),
            'low': np.random.uniform(395, 405, 30),
            'close': np.random.uniform(400, 410, 30),
            'volume': np.random.uniform(10000000, 50000000, 30)
        }, index=dates)
        
        market_data = {
            'SPY': spy_data
        }
        
        # Test feature preparation
        features = model.prepare_features(market_data)
        
        # Check feature shape
        self.assertEqual(features.shape, (1, 10))
        
        # Test prediction
        regime, confidence = model.predict(features)
        
        # Check regime is one of the defined types
        self.assertIn(regime, ['bullish', 'bearish', 'sideways', 'volatile', 'unknown'])
        
        # Check confidence is between 0 and 1
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
    
    def test_strategy_selection_model(self):
        """Test strategy selection model."""
        # Create model
        model = StrategySelectionModel(self.test_models_dir)
        
        # Create test symbol and market regime
        symbol = 'AAPL'
        market_regime = {
            'regime': 'bullish',
            'confidence': 0.8,
            'timestamp': datetime.now().isoformat()
        }
        
        # Mock Knowledge Base
        model.kb = MagicMock()
        
        # Create sample market data
        dates = pd.date_range(end=datetime.now(), periods=30)
        market_data = pd.DataFrame({
            'open': np.random.uniform(145, 155, 30),
            'high': np.random.uniform(150, 160, 30),
            'low': np.random.uniform(140, 150, 30),
            'close': np.random.uniform(145, 155, 30),
            'volume': np.random.uniform(1000000, 5000000, 30)
        }, index=dates)
        
        model.kb.get_market_data.return_value = market_data
        
        # Test feature preparation
        features = model.prepare_features(symbol, market_regime)
        
        # Check feature shape
        self.assertEqual(features.shape, (1, 10))
        
        # Test prediction
        strategy, confidence = model.predict(features)
        
        # Check strategy is one of the defined types
        self.assertIn(strategy, ['momentum', 'sector_rotation', 'swing_trading', 'mean_reversion'])
        
        # Check confidence is between 0 and 1
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)


class TestPerformanceMetrics(unittest.TestCase):
    """Test cases for the Performance Metrics class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.metrics = PerformanceMetrics()
    
    def test_calculate_metrics(self):
        """Test metrics calculation."""
        # Create sample orders and positions
        start_date = datetime.now() - timedelta(days=30)
        end_date = datetime.now()
        
        orders = [
            {
                'symbol': 'AAPL',
                'strategy': 'momentum',
                'direction': 'BUY',
                'quantity': 10,
                'price': 150.0,
                'timestamp': (start_date + timedelta(days=1)).isoformat(),
                'pnl': 500.0
            },
            {
                'symbol': 'MSFT',
                'strategy': 'sector_rotation',
                'direction': 'BUY',
                'quantity': 5,
                'price': 300.0,
                'timestamp': (start_date + timedelta(days=5)).isoformat(),
                'pnl': -200.0
            },
            {
                'symbol': 'GOOGL',
                'strategy': 'swing_trading',
                'direction': 'BUY',
                'quantity': 2,
                'price': 2500.0,
                'timestamp': (start_date + timedelta(days=10)).isoformat(),
                'pnl': 300.0
            }
        ]
        
        positions = [
            {
                'symbol': 'AAPL',
                'quantity': 10,
                'average_cost': 150.0,
                'market_value': 1600.0
            },
            {
                'symbol': 'AMZN',
                'quantity': 3,
                'average_cost': 3300.0,
                'market_value': 10200.0
            }
        ]
        
        # Test metrics calculation
        metrics = self.metrics.calculate_metrics(orders, positions, start_date, end_date)
        
        # Check metrics structure
        self.assertIn('start_date', metrics)
        self.assertIn('end_date', metrics)
        self.assertIn('total_trades', metrics)
        self.assertIn('winning_trades', metrics)
        self.assertIn('losing_trades', metrics)
        self.assertIn('win_rate', metrics)
        self.assertIn('total_return', metrics)
        
        # Check specific values
        self.assertEqual(metrics['total_trades'], 3)
        self.assertEqual(metrics['winning_trades'], 2)
        self.assertEqual(metrics['losing_trades'], 1)
        self.assertAlmostEqual(metrics['win_rate'], 2/3)
        self.assertEqual(metrics['total_return'], 600.0)


class TestStrategyOptimizer(unittest.TestCase):
    """Test cases for the Strategy Optimizer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.optimizer = StrategyOptimizer()
    
    def test_optimize_momentum_params(self):
        """Test momentum strategy parameter optimization."""
        # Create current parameters
        current_params = {
            'lookback_period': 20,
            'momentum_threshold': 0.05
        }
        
        # Create performance data with poor win rate
        poor_performance = {
            'win_rate': 0.3,
            'profit_factor': 0.8
        }
        
        # Test optimization with poor performance
        optimized_params = self.optimizer._optimize_momentum_params(current_params, poor_performance)
        
        # Check if parameters were adjusted
        self.assertGreater(optimized_params['lookback_period'], current_params['lookback_period'])
        self.assertGreater(optimized_params['momentum_threshold'], current_params['momentum_threshold'])
        
        # Create performance data with good win rate
        good_performance = {
            'win_rate': 0.7,
            'profit_factor': 2.0
        }
        
        # Test optimization with good performance
        optimized_params = self.optimizer._optimize_momentum_params(current_params, good_performance)
        
        # Check if parameters were adjusted
        self.assertLess(optimized_params['lookback_period'], current_params['lookback_period'])
        self.assertLess(optimized_params['momentum_threshold'], current_params['momentum_threshold'])
    
    def test_optimize_sector_rotation_params(self):
        """Test sector rotation strategy parameter optimization."""
        # Create current parameters
        current_params = {
            'top_sectors_count': 3,
            'rotation_period_days': 30
        }
        
        # Create performance data with poor win rate
        poor_performance = {
            'win_rate': 0.3,
            'profit_factor': 0.8
        }
        
        # Test optimization with poor performance
        optimized_params = self.optimizer._optimize_sector_rotation_params(current_params, poor_performance)
        
        # Check if parameters were adjusted
        self.assertLess(optimized_params['top_sectors_count'], current_params['top_sectors_count'])
        self.assertGreater(optimized_params['rotation_period_days'], current_params['rotation_period_days'])


class TestParameterTuner(unittest.TestCase):
    """Test cases for the Parameter Tuner class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tuner = ParameterTuner()
    
    def test_tune_risk_parameters(self):
        """Test risk parameter tuning."""
        # Create current parameters
        current_params = {
            'max_position_size_pct': 0.05,
            'max_sector_exposure_pct': 0.3,
            'max_leverage': 1.0
        }
        
        # Create performance data with high drawdown
        high_drawdown_performance = {
            'win_rate': 0.5,
            'profit_factor': 1.2,
            'max_drawdown': 0.2
        }
        
        # Test tuning with high drawdown
        tuned_params = self.tuner.tune_risk_parameters(current_params, high_drawdown_performance)
        
        # Check if parameters were adjusted
        self.assertLess(tuned_params['max_position_size_pct'], current_params['max_position_size_pct'])
        self.assertLess(tuned_params['max_sector_exposure_pct'], current_params['max_sector_exposure_pct'])
        self.assertLessEqual(tuned_params['max_leverage'], current_params['max_leverage'])
        
        # Create performance data with low drawdown and good performance
        low_drawdown_performance = {
            'win_rate': 0.7,
            'profit_factor': 2.0,
            'max_drawdown': 0.03
        }
        
        # Test tuning with low drawdown and good performance
        tuned_params = self.tuner.tune_risk_parameters(current_params, low_drawdown_performance)
        
        # Check if parameters were adjusted
        self.assertGreater(tuned_params['max_position_size_pct'], current_params['max_position_size_pct'])
        self.assertGreater(tuned_params['max_sector_exposure_pct'], current_params['max_sector_exposure_pct'])
        self.assertGreaterEqual(tuned_params['max_leverage'], current_params['max_leverage'])


if __name__ == '__main__':
    unittest.main()
