"""
Algorithmic Trading System - Tests

This module contains tests for the Algorithmic Trading System component.
"""

import unittest
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import time

from ..algorithmic_trading.trading_system import AlgorithmicTradingSystem, RiskControls
from ..algorithmic_trading.models import (
    Order, Position, AccountInfo, ExecutionResult, PortfolioStatus, RiskControlSettings,
    OrderStatus, OrderType, OrderDirection, TimeInForce
)
from ..knowledge_base import get_knowledge_base
from ..ai_trading_agent import get_ai_trading_agent

class TestAlgorithmicTradingSystem(unittest.TestCase):
    """Test cases for the AlgorithmicTradingSystem class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a test data directory
        self.test_dir = Path(os.path.dirname(os.path.abspath(__file__))) / 'test_data'
        self.test_dir.mkdir(exist_ok=True)
        
        # Initialize Knowledge Base with test directory
        self.kb = get_knowledge_base(self.test_dir)
        
        # Initialize AI Trading Agent
        self.ai_agent = get_ai_trading_agent()
        
        # Initialize Algorithmic Trading System
        self.trading_system = AlgorithmicTradingSystem(paper_trading=True, max_positions=10)
    
    def test_connect_to_ib(self):
        """Test connection to Interactive Brokers."""
        # Connect to IB
        result = self.trading_system.connect_to_ib(host='localhost', port=7497, client_id=1)
        
        # Check that connection was successful
        self.assertTrue(result)
        self.assertTrue(self.trading_system.ib_connection['connected'])
    
    def test_disconnect_from_ib(self):
        """Test disconnection from Interactive Brokers."""
        # Connect to IB first
        self.trading_system.connect_to_ib(host='localhost', port=7497, client_id=1)
        
        # Disconnect from IB
        result = self.trading_system.disconnect_from_ib()
        
        # Check that disconnection was successful
        self.assertTrue(result)
        self.assertFalse(self.trading_system.ib_connection['connected'])
    
    def test_process_signal(self):
        """Test processing of trading signals."""
        # Connect to IB first
        self.trading_system.connect_to_ib(host='localhost', port=7497, client_id=1)
        
        # Create a test signal
        signal = {
            'symbol': 'AAPL',
            'direction': 'BUY',
            'strategy': 'momentum',
            'strength': 0.8,
            'timeframe': 'daily',
            'expiration': (datetime.now() + timedelta(days=5)).isoformat(),
            'metadata': {
                'momentum_score': 0.15,
                'rsi': 35
            }
        }
        
        # Process signal
        order_id = self.trading_system.process_signal(signal)
        
        # Check that order was created
        self.assertNotEqual(order_id, "")
        self.assertIn(order_id, self.trading_system.active_orders)
    
    def test_execute_order(self):
        """Test execution of orders."""
        # Connect to IB first
        self.trading_system.connect_to_ib(host='localhost', port=7497, client_id=1)
        
        # Create a test order
        order = {
            'order_id': 'test_order_1',
            'symbol': 'AAPL',
            'direction': 'BUY',
            'quantity': 10,
            'order_type': 'MARKET',
            'status': 'PENDING',
            'execution_algorithm': 'market',
            'strategy': 'momentum',
            'timestamp': datetime.now().isoformat()
        }
        
        # Execute order
        result = self.trading_system.execute_order(order)
        
        # Check that order was executed
        self.assertTrue(result)
        self.assertEqual(order['status'], 'FILLED')
    
    def test_cancel_order(self):
        """Test cancellation of orders."""
        # Connect to IB first
        self.trading_system.connect_to_ib(host='localhost', port=7497, client_id=1)
        
        # Create a test order
        order = {
            'order_id': 'test_order_2',
            'symbol': 'AAPL',
            'direction': 'BUY',
            'quantity': 10,
            'order_type': 'LIMIT',
            'price': 150.0,
            'status': 'PENDING',
            'execution_algorithm': 'limit',
            'strategy': 'momentum',
            'timestamp': datetime.now().isoformat()
        }
        
        # Add to active orders
        self.trading_system.active_orders[order['order_id']] = order
        
        # Cancel order
        result = self.trading_system.cancel_order(order['order_id'])
        
        # Check that order was cancelled
        self.assertTrue(result)
        self.assertEqual(order['status'], 'CANCELLED')
        self.assertNotIn(order['order_id'], self.trading_system.active_orders)
    
    def test_get_order_status(self):
        """Test retrieving order status."""
        # Connect to IB first
        self.trading_system.connect_to_ib(host='localhost', port=7497, client_id=1)
        
        # Create a test order
        order = {
            'order_id': 'test_order_3',
            'symbol': 'AAPL',
            'direction': 'BUY',
            'quantity': 10,
            'order_type': 'MARKET',
            'status': 'PENDING',
            'execution_algorithm': 'market',
            'strategy': 'momentum',
            'timestamp': datetime.now().isoformat()
        }
        
        # Add to active orders
        self.trading_system.active_orders[order['order_id']] = order
        
        # Get order status
        status = self.trading_system.get_order_status(order['order_id'])
        
        # Check that status was retrieved
        self.assertEqual(status['order_id'], order['order_id'])
        self.assertEqual(status['status'], order['status'])
    
    def test_get_portfolio_status(self):
        """Test retrieving portfolio status."""
        # Connect to IB first
        self.trading_system.connect_to_ib(host='localhost', port=7497, client_id=1)
        
        # Get portfolio status
        status = self.trading_system.get_portfolio_status()
        
        # Check that status was retrieved
        self.assertIn('portfolio_value', status)
        self.assertIn('buying_power', status)
        self.assertIn('cash_balance', status)
        self.assertIn('positions', status)
    
    def test_risk_controls(self):
        """Test risk controls."""
        # Create risk controls
        risk_controls = RiskControls(
            max_position_size_pct=0.05,
            max_daily_drawdown_pct=0.02,
            max_sector_exposure_pct=0.30,
            max_leverage=1.5
        )
        
        # Create test positions
        positions = {
            'AAPL': {
                'symbol': 'AAPL',
                'position': 100,
                'average_cost': 150.0,
                'market_value': 15000.0,
                'unrealized_pnl': 0.0,
                'sector': 'Technology'
            }
        }
        
        # Create test account info
        account_info = {
            'net_liquidation_value': 100000.0
        }
        
        # Create a valid signal
        valid_signal = {
            'symbol': 'MSFT',
            'direction': 'BUY',
            'strategy': 'momentum',
            'strength': 0.5,
            'timeframe': 'daily'
        }
        
        # Check valid signal
        result = risk_controls.check_signal(valid_signal, positions, account_info)
        self.assertTrue(result)
        
        # Create a signal that would exceed position size
        large_signal = {
            'symbol': 'AAPL',
            'direction': 'BUY',
            'strategy': 'momentum',
            'strength': 1.0,  # 100% of max position size
            'timeframe': 'daily'
        }
        
        # Check large signal
        result = risk_controls.check_signal(large_signal, positions, account_info)
        self.assertFalse(result)
    
    def tearDown(self):
        """Clean up after tests."""
        # Disconnect from IB if connected
        if self.trading_system.ib_connection and self.trading_system.ib_connection.get('connected', False):
            self.trading_system.disconnect_from_ib()
        
        # Shutdown trading system
        self.trading_system.shutdown()


class TestExecutionAlgorithms(unittest.TestCase):
    """Test cases for the execution algorithms."""
    
    def setUp(self):
        """Set up test environment."""
        # Initialize Algorithmic Trading System
        self.trading_system = AlgorithmicTradingSystem(paper_trading=True, max_positions=10)
        
        # Connect to IB
        self.trading_system.connect_to_ib(host='localhost', port=7497, client_id=1)
    
    def test_market_order_algorithm(self):
        """Test market order algorithm."""
        # Create a test order
        order = {
            'order_id': 'test_market_1',
            'symbol': 'AAPL',
            'direction': 'BUY',
            'quantity': 10,
            'order_type': 'MARKET',
            'status': 'PENDING',
            'execution_algorithm': 'market',
            'strategy': 'momentum',
            'timestamp': datetime.now().isoformat()
        }
        
        # Get algorithm
        algorithm = self.trading_system.execution_algorithms['market']
        
        # Execute order
        result = algorithm.execute_paper(order)
        
        # Check that order was executed
        self.assertTrue(result['success'])
        self.assertEqual(result['order_id'], order['order_id'])
        self.assertIn('average_price', result)
        self.assertIn('filled_quantity', result)
    
    def test_limit_order_algorithm(self):
        """Test limit order algorithm."""
        # Create a test order
        order = {
            'order_id': 'test_limit_1',
            'symbol': 'AAPL',
            'direction': 'BUY',
            'quantity': 10,
            'order_type': 'LIMIT',
            'price': 150.0,
            'status': 'PENDING',
            'execution_algorithm': 'limit',
            'strategy': 'momentum',
            'timestamp': datetime.now().isoformat()
        }
        
        # Get algorithm
        algorithm = self.trading_system.execution_algorithms['limit']
        
        # Execute order
        result = algorithm.execute_paper(order)
        
        # Check that order was executed
        self.assertTrue(result['success'])
        self.assertEqual(result['order_id'], order['order_id'])
        self.assertIn('average_price', result)
        self.assertIn('filled_quantity', result)
    
    def test_twap_algorithm(self):
        """Test TWAP algorithm."""
        # Create a test order
        order = {
            'order_id': 'test_twap_1',
            'symbol': 'AAPL',
            'direction': 'BUY',
            'quantity': 100,
            'order_type': 'ALGO',
            'status': 'PENDING',
            'execution_algorithm': 'twap',
            'strategy': 'momentum',
            'timestamp': datetime.now().isoformat()
        }
        
        # Get algorithm
        algorithm = self.trading_system.execution_algorithms['twap']
        
        # Execute order
        result = algorithm.execute_paper(order)
        
        # Check that order was executed
        self.assertTrue(result['success'])
        self.assertEqual(result['order_id'], order['order_id'])
        self.assertIn('average_price', result)
        self.assertIn('filled_quantity', result)
        self.assertIn('execution_details', result)
        self.assertEqual(result['execution_details']['algorithm'], 'TWAP')
    
    def test_vwap_algorithm(self):
        """Test VWAP algorithm."""
        # Create a test order
        order = {
            'order_id': 'test_vwap_1',
            'symbol': 'AAPL',
            'direction': 'BUY',
            'quantity': 100,
            'order_type': 'ALGO',
            'status': 'PENDING',
            'execution_algorithm': 'vwap',
            'strategy': 'momentum',
            'timestamp': datetime.now().isoformat()
        }
        
        # Get algorithm
        algorithm = self.trading_system.execution_algorithms['vwap']
        
        # Execute order
        result = algorithm.execute_paper(order)
        
        # Check that order was executed
        self.assertTrue(result['success'])
        self.assertEqual(result['order_id'], order['order_id'])
        self.assertIn('average_price', result)
        self.assertIn('filled_quantity', result)
        self.assertIn('execution_details', result)
        self.assertEqual(result['execution_details']['algorithm'], 'VWAP')
    
    def test_iceberg_algorithm(self):
        """Test Iceberg algorithm."""
        # Create a test order
        order = {
            'order_id': 'test_iceberg_1',
            'symbol': 'AAPL',
            'direction': 'BUY',
            'quantity': 1000,
            'order_type': 'ALGO',
            'status': 'PENDING',
            'execution_algorithm': 'iceberg',
            'strategy': 'momentum',
            'timestamp': datetime.now().isoformat()
        }
        
        # Get algorithm
        algorithm = self.trading_system.execution_algorithms['iceberg']
        
        # Execute order
        result = algorithm.execute_paper(order)
        
        # Check that order was executed
        self.assertTrue(result['success'])
        self.assertEqual(result['order_id'], order['order_id'])
        self.assertIn('average_price', result)
        self.assertIn('filled_quantity', result)
        self.assertIn('execution_details', result)
        self.assertEqual(result['execution_details']['algorithm'], 'Iceberg')
    
    def tearDown(self):
        """Clean up after tests."""
        # Disconnect from IB
        self.trading_system.disconnect_from_ib()
        
        # Shutdown trading system
        self.trading_system.shutdown()


class TestModels(unittest.TestCase):
    """Test cases for the data models."""
    
    def test_order_model(self):
        """Test Order model."""
        # Create an order
        order = Order(
            order_id='test_order_1',
            symbol='AAPL',
            direction=OrderDirection.BUY,
            quantity=10,
            order_type=OrderType.MARKET,
            status=OrderStatus.PENDING,
            execution_algorithm='market',
            strategy='momentum'
        )
        
        # Convert to dictionary
        order_dict = order.to_dict()
        
        # Check dictionary values
        self.assertEqual(order_dict['order_id'], 'test_order_1')
        self.assertEqual(order_dict['symbol'], 'AAPL')
        self.assertEqual(order_dict['direction'], 'BUY')
        self.assertEqual(order_dict['quantity'], 10)
        self.assertEqual(order_dict['order_type'], 'MARKET')
        self.assertEqual(order_dict['status'], 'PENDING')
        
        # Create from dictionary
        order2 = Order.from_dict(order_dict)
        
        # Check values
        self.assertEqual(order2.order_id, 'test_order_1')
        self.assertEqual(order2.symbol, 'AAPL')
        self.assertEqual(order2.direction, OrderDirection.BUY)
        self.assertEqual(order2.quantity, 10)
        self.assertEqual(order2.order_type, OrderType.MARKET)
        self.assertEqual(order2.status, OrderStatus.PENDING)
    
    def test_position_model(self):
        """Test Position model."""
        # Create a position
        position = Position(
            symbol='AAPL',
            quantity=100,
            average_cost=150.0,
            market_value=15000.0,
            unrealized_pnl=0.0,
            sector='Technology'
        )
        
        # Convert to dictionary
        position_dict = position.to_dict()
        
        # Check dictionary values
        self.assertEqual(position_dict['symbol'], 'AAPL')
        self.assertEqual(position_dict['quantity'], 100)
        self.assertEqual(position_dict['average_cost'], 150.0)
        self.assertEqual(position_dict['market_value'], 15000.0)
        self.assertEqual(position_dict['unrealized_pnl'], 0.0)
        self.assertEqual(position_dict['sector'], 'Technology')
        
        # Create from dictionary
        position2 = Position.from_dict(position_dict)
        
        # Check values
        self.assertEqual(position2.symbol, 'AAPL')
        self.assertEqual(position2.quantity, 100)
        self.assertEqual(position2.average_cost, 150.0)
        self.assertEqual(position2.market_value, 15000.0)
        self.assertEqual(position2.unrealized_pnl, 0.0)
        self.assertEqual(position2.sector, 'Technology')
    
    def test_account_info_model(self):
        """Test AccountInfo model."""
        # Create account info
        account_info = AccountInfo(
            account_id='DU123456',
            net_liquidation_value=100000.0,
            equity_with_loan_value=100000.0,
            buying_power=200000.0,
            cash_balance=50000.0,
            day_trades_remaining=3,
            leverage=1.0
        )
        
        # Convert to dictionary
        account_dict = account_info.to_dict()
        
        # Check dictionary values
        self.assertEqual(account_dict['account_id'], 'DU123456')
        self.assertEqual(account_dict['net_liquidation_value'], 100000.0)
        self.assertEqual(account_dict['equity_with_loan_value'], 100000.0)
        self.assertEqual(account_dict['buying_power'], 200000.0)
        self.assertEqual(account_dict['cash_balance'], 50000.0)
        self.assertEqual(account_dict['day_trades_remaining'], 3)
        self.assertEqual(account_dict['leverage'], 1.0)
        
        # Create from dictionary
        account_info2 = AccountInfo.from_dict(account_dict)
        
        # Check values
        self.assertEqual(account_info2.account_id, 'DU123456')
        self.assertEqual(account_info2.net_liquidation_value, 100000.0)
        self.assertEqual(account_info2.equity_with_loan_value, 100000.0)
        self.assertEqual(account_info2.buying_power, 200000.0)
        self.assertEqual(account_info2.cash_balance, 50000.0)
        self.assertEqual(account_info2.day_trades_remaining, 3)
        self.assertEqual(account_info2.leverage, 1.0)


if __name__ == '__main__':
    unittest.main()
