import unittest
import os
import sys
import json
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the module to test
from main.trading_system import TradingSystem

class TestTradingSystem(unittest.TestCase):
    """Test cases for the Trading System."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock dependencies
        self.mock_kb = MagicMock()
        self.mock_ai_agent = MagicMock()
        self.mock_trading_system = MagicMock()
        self.mock_data_analysis = MagicMock()
        self.mock_continuous_improvement = MagicMock()
        
        # Create test configuration
        self.test_config = {
            "ib_host": "127.0.0.1",
            "ib_port": 7497,
            "ib_client_id": 1,
            "main_loop_interval": 1,  # Short interval for testing
            "full_update_interval": 86400,
            "signal_quality_threshold": 0.6,
            "signal_max_age": 3600,
            "feedback_max_age": 604800,
            "symbol_universe": ["AAPL", "MSFT", "AMZN", "SPY"],
            "watchlist": ["AAPL", "MSFT", "SPY"]
        }
        
        # Create patches for dependencies
        self.kb_patch = patch('main.trading_system.get_knowledge_base', 
                             return_value=self.mock_kb)
        self.ai_agent_patch = patch('main.trading_system.get_ai_trading_agent', 
                                   return_value=self.mock_ai_agent)
        self.trading_system_patch = patch('main.trading_system.get_algorithmic_trading_system', 
                                         return_value=self.mock_trading_system)
        self.data_analysis_patch = patch('main.trading_system.get_data_analysis_system', 
                                        return_value=self.mock_data_analysis)
        self.ci_patch = patch('main.trading_system.get_continuous_improvement_system', 
                             return_value=self.mock_continuous_improvement)
        
        # Start patches
        self.kb_patch.start()
        self.ai_agent_patch.start()
        self.trading_system_patch.start()
        self.data_analysis_patch.start()
        self.ci_patch.start()
        
        # Create system instance with test configuration
        self.system = TradingSystem(config_file=None)
        self.system.config = self.test_config
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Stop patches
        self.kb_patch.stop()
        self.ai_agent_patch.stop()
        self.trading_system_patch.stop()
        self.data_analysis_patch.stop()
        self.ci_patch.stop()
    
    def test_initialization(self):
        """Test system initialization."""
        self.assertIsNotNone(self.system)
        self.assertEqual(self.system.config, self.test_config)
        self.assertFalse(self.system.running)
        self.assertFalse(self.system.paused)
    
    def test_start_stop(self):
        """Test starting and stopping the system."""
        # Test start
        self.system.start()
        self.assertTrue(self.system.running)
        self.assertFalse(self.system.paused)
        self.assertIsNotNone(self.system.main_thread)
        
        # Test stop
        self.system.stop()
        self.assertFalse(self.system.running)
    
    def test_pause_resume(self):
        """Test pausing and resuming the system."""
        # Start system
        self.system.start()
        
        # Test pause
        self.system.pause()
        self.assertTrue(self.system.paused)
        
        # Test resume
        self.system.resume()
        self.assertFalse(self.system.paused)
        
        # Stop system
        self.system.stop()
    
    def test_get_status(self):
        """Test getting system status."""
        # Mock component status methods
        self.mock_kb.get_status = MagicMock(return_value={"status": "ok"})
        self.mock_ai_agent.get_status = MagicMock(return_value={"status": "ok"})
        self.mock_trading_system.get_status = MagicMock(return_value={"status": "ok"})
        self.mock_data_analysis.get_status = MagicMock(return_value={"status": "ok"})
        self.mock_continuous_improvement.get_status = MagicMock(return_value={"status": "ok"})
        
        # Get status
        status = self.system.get_status()
        
        # Check status structure
        self.assertIn("running", status)
        self.assertIn("paused", status)
        self.assertIn("timestamp", status)
        self.assertIn("components", status)
        
        # Check component status
        self.assertIn("knowledge_base", status["components"])
        self.assertIn("ai_trading_agent", status["components"])
        self.assertIn("algorithmic_trading", status["components"])
        self.assertIn("data_analysis", status["components"])
        self.assertIn("continuous_improvement", status["components"])
    
    def test_execute_command(self):
        """Test executing commands."""
        # Start system
        self.system.start()
        
        # Mock internal command execution
        self.system._execute_command_internal = MagicMock(return_value={"status": "success", "data": "test"})
        
        # Execute command
        result = self.system.execute_command("test_command", arg1="value1")
        
        # Check result
        self.assertEqual(result, {"status": "success", "data": "test"})
        
        # Verify internal execution was called
        self.system._execute_command_internal.assert_called_once_with("test_command", arg1="value1")
        
        # Stop system
        self.system.stop()
    
    def test_execute_command_internal(self):
        """Test internal command execution."""
        # Test get_status command
        result = self.system._execute_command_internal("get_status")
        self.assertEqual(result["status"], "success")
        self.assertIn("data", result)
        
        # Test get_portfolio command
        self.mock_trading_system.get_portfolio = MagicMock(return_value=[{"symbol": "AAPL"}])
        result = self.system._execute_command_internal("get_portfolio")
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["data"], [{"symbol": "AAPL"}])
        
        # Test get_market_data command with missing symbol
        result = self.system._execute_command_internal("get_market_data")
        self.assertEqual(result["status"], "error")
        
        # Test get_market_data command with valid symbol
        import pandas as pd
        market_data = pd.DataFrame({"close": [100, 101, 102]})
        self.mock_kb.get_market_data = MagicMock(return_value=market_data)
        result = self.system._execute_command_internal("get_market_data", symbol="AAPL")
        self.assertEqual(result["status"], "success")
        self.assertIn("data", result)
    
    def test_initialize_session(self):
        """Test session initialization."""
        # Mock component methods
        self.mock_trading_system.connect = MagicMock()
        self.mock_trading_system.update_portfolio = MagicMock()
        self.mock_trading_system.update_account_info = MagicMock()
        
        # Mock market data update and regime detection
        self.system._update_market_data = MagicMock()
        self.system._detect_market_regime = MagicMock()
        
        # Initialize session
        self.system._initialize_session()
        
        # Verify methods were called
        self.mock_trading_system.connect.assert_called_once()
        self.mock_trading_system.update_portfolio.assert_called_once()
        self.mock_trading_system.update_account_info.assert_called_once()
        self.system._update_market_data.assert_called_once_with(force_full_update=True)
        self.system._detect_market_regime.assert_called_once()
    
    def test_update_market_data(self):
        """Test market data update."""
        # Mock component methods
        self.mock_kb.get_last_market_data_update = MagicMock(return_value=datetime.now() - timedelta(hours=1))
        self.mock_trading_system.get_portfolio = MagicMock(return_value=[{"symbol": "AAPL"}])
        self.mock_data_analysis.update_market_data = MagicMock()
        self.mock_kb.set_last_market_data_update = MagicMock()
        
        # Update market data
        self.system._update_market_data()
        
        # Verify methods were called
        self.mock_kb.get_last_market_data_update.assert_called_once()
        self.mock_trading_system.get_portfolio.assert_called_once()
        self.mock_data_analysis.update_market_data.assert_called_once()
        self.mock_kb.set_last_market_data_update.assert_called_once()
    
    def test_generate_trading_signals(self):
        """Test trading signal generation."""
        # Mock component methods
        self.system._detect_market_regime = MagicMock(return_value={"regime": "bullish", "confidence": 0.8})
        self.mock_trading_system.get_portfolio = MagicMock(return_value=[{"symbol": "AAPL"}])
        self.mock_continuous_improvement.select_optimal_strategy = MagicMock(return_value="momentum")
        self.mock_ai_agent.generate_signal = MagicMock(return_value={"symbol": "AAPL", "direction": "BUY"})
        self.mock_continuous_improvement.evaluate_signal_quality = MagicMock(return_value=0.8)
        self.mock_kb.store_signal = MagicMock()
        
        # Generate signals
        signals = self.system._generate_trading_signals()
        
        # Verify methods were called
        self.system._detect_market_regime.assert_called_once()
        self.mock_trading_system.get_portfolio.assert_called_once()
        self.mock_continuous_improvement.select_optimal_strategy.assert_called_once()
        self.mock_ai_agent.generate_signal.assert_called_once()
        self.mock_continuous_improvement.evaluate_signal_quality.assert_called_once()
        self.mock_kb.store_signal.assert_called_once()
        
        # Check signals
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0]["symbol"], "AAPL")
        self.assertEqual(signals[0]["direction"], "BUY")
        self.assertEqual(signals[0]["quality"], 0.8)
    
    def test_execute_trading_signals(self):
        """Test trading signal execution."""
        # Mock component methods
        self.mock_kb.get_recent_signals = MagicMock(return_value=[
            {"id": "1", "symbol": "AAPL", "direction": "BUY", "executed": False}
        ])
        self.mock_trading_system.get_account_info = MagicMock(return_value={"net_liquidation_value": 100000})
        self.mock_continuous_improvement.optimize_position_size = MagicMock(return_value=0.05)
        self.mock_trading_system.execute_signal = MagicMock(return_value="order123")
        self.mock_kb.update_signal = MagicMock()
        
        # Execute signals
        self.system._execute_trading_signals()
        
        # Verify methods were called
        self.mock_kb.get_recent_signals.assert_called_once()
        self.mock_trading_system.get_account_info.assert_called_once()
        self.mock_continuous_improvement.optimize_position_size.assert_called_once()
        self.mock_trading_system.execute_signal.assert_called_once()
        self.mock_kb.update_signal.assert_called_once()
    
    def test_process_feedback(self):
        """Test feedback processing."""
        # Mock component methods
        self.mock_kb.get_completed_trades = MagicMock(return_value=[
            {"id": "1", "symbol": "AAPL", "pnl": 500, "feedback_processed": False}
        ])
        self.mock_continuous_improvement.process_trade_feedback = MagicMock()
        self.mock_kb.update_trade = MagicMock()
        
        # Process feedback
        self.system._process_feedback()
        
        # Verify methods were called
        self.mock_kb.get_completed_trades.assert_called_once()
        self.mock_continuous_improvement.process_trade_feedback.assert_called_once()
        self.mock_kb.update_trade.assert_called_once()
    
    def test_detect_market_regime(self):
        """Test market regime detection."""
        # Mock component methods
        import pandas as pd
        market_data = pd.DataFrame({"close": [100, 101, 102]})
        self.mock_kb.get_market_data = MagicMock(return_value=market_data)
        self.mock_continuous_improvement.detect_market_regime = MagicMock(return_value={"regime": "bullish", "confidence": 0.8})
        self.mock_kb.store_market_regime = MagicMock()
        
        # Detect market regime
        regime_info = self.system._detect_market_regime()
        
        # Verify methods were called
        self.mock_kb.get_market_data.assert_called()
        self.mock_continuous_improvement.detect_market_regime.assert_called_once()
        self.mock_kb.store_market_regime.assert_called_once()
        
        # Check regime info
        self.assertEqual(regime_info["regime"], "bullish")
        self.assertEqual(regime_info["confidence"], 0.8)
    
    def test_load_config(self):
        """Test configuration loading."""
        # Create temporary config file
        config_file = "test_config.json"
        test_config = {
            "ib_host": "192.168.1.1",
            "ib_port": 4001,
            "symbol_universe": ["AAPL", "MSFT"]
        }
        
        with open(config_file, "w") as f:
            json.dump(test_config, f)
        
        try:
            # Load config
            config = self.system._load_config(config_file)
            
            # Check config
            self.assertEqual(config["ib_host"], "192.168.1.1")
            self.assertEqual(config["ib_port"], 4001)
            self.assertEqual(config["symbol_universe"], ["AAPL", "MSFT"])
            
            # Check default values are preserved
            self.assertIn("main_loop_interval", config)
            
        finally:
            # Clean up
            if os.path.exists(config_file):
                os.remove(config_file)


if __name__ == '__main__':
    unittest.main()
