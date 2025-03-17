"""
DeepResearch 4.5 Trading System - Main Application

This module integrates all components of the DeepResearch 4.5 Trading System
and provides the main entry point for the application.
"""

import os
import sys
import time
import logging
import argparse
import json
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import threading
import queue
import signal

# Import components
from ..knowledge_base import get_knowledge_base
from ..ai_trading_agent import get_ai_trading_agent
from ..algorithmic_trading import get_algorithmic_trading_system
from ..data_analysis import get_data_analysis_system
from ..continuous_improvement import get_continuous_improvement_system

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_system.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("TradingSystem")

class TradingSystem:
    """
    Main class for the DeepResearch 4.5 Trading System.
    
    This class integrates all components of the trading system and
    provides the main functionality for running the system.
    """
    
    def __init__(self, config_file=None):
        """
        Initialize the trading system.
        
        Args:
            config_file: Path to configuration file
        """
        logger.info("Initializing DeepResearch 4.5 Trading System")
        
        # Load configuration
        self.config = self._load_config(config_file)
        
        # Initialize components
        self.kb = get_knowledge_base()
        self.ai_agent = get_ai_trading_agent()
        self.trading_system = get_algorithmic_trading_system()
        self.data_analysis = get_data_analysis_system()
        self.continuous_improvement = get_continuous_improvement_system()
        
        # Initialize state
        self.running = False
        self.paused = False
        self.main_thread = None
        self.command_queue = queue.Queue()
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("Trading system initialized")
    
    def start(self):
        """Start the trading system."""
        if self.running:
            logger.warning("Trading system is already running")
            return
        
        logger.info("Starting trading system")
        
        # Set running flag
        self.running = True
        self.paused = False
        
        # Start main thread
        self.main_thread = threading.Thread(target=self._main_loop)
        self.main_thread.daemon = True
        self.main_thread.start()
        
        logger.info("Trading system started")
    
    def stop(self):
        """Stop the trading system."""
        if not self.running:
            logger.warning("Trading system is not running")
            return
        
        logger.info("Stopping trading system")
        
        # Clear running flag
        self.running = False
        
        # Wait for main thread to finish
        if self.main_thread and self.main_thread.is_alive():
            self.main_thread.join(timeout=30)
        
        # Shutdown components
        self._shutdown_components()
        
        logger.info("Trading system stopped")
    
    def pause(self):
        """Pause the trading system."""
        if not self.running:
            logger.warning("Trading system is not running")
            return
        
        if self.paused:
            logger.warning("Trading system is already paused")
            return
        
        logger.info("Pausing trading system")
        
        # Set paused flag
        self.paused = True
        
        logger.info("Trading system paused")
    
    def resume(self):
        """Resume the trading system."""
        if not self.running:
            logger.warning("Trading system is not running")
            return
        
        if not self.paused:
            logger.warning("Trading system is not paused")
            return
        
        logger.info("Resuming trading system")
        
        # Clear paused flag
        self.paused = False
        
        logger.info("Trading system resumed")
    
    def execute_command(self, command, **kwargs):
        """
        Execute a command in the trading system.
        
        Args:
            command: Command to execute
            **kwargs: Command arguments
        
        Returns:
            Command result
        """
        if not self.running:
            logger.warning("Trading system is not running")
            return {"status": "error", "message": "Trading system is not running"}
        
        logger.info(f"Executing command: {command}")
        
        # Create command object
        cmd = {
            "command": command,
            "args": kwargs,
            "result_queue": queue.Queue()
        }
        
        # Add command to queue
        self.command_queue.put(cmd)
        
        # Wait for result
        try:
            result = cmd["result_queue"].get(timeout=30)
            return result
        except queue.Empty:
            return {"status": "error", "message": "Command timed out"}
    
    def get_status(self):
        """
        Get the current status of the trading system.
        
        Returns:
            Status information
        """
        # Get component status
        kb_status = self.kb.get_status() if hasattr(self.kb, "get_status") else {}
        ai_agent_status = self.ai_agent.get_status() if hasattr(self.ai_agent, "get_status") else {}
        trading_system_status = self.trading_system.get_status() if hasattr(self.trading_system, "get_status") else {}
        data_analysis_status = self.data_analysis.get_status() if hasattr(self.data_analysis, "get_status") else {}
        ci_status = self.continuous_improvement.get_status() if hasattr(self.continuous_improvement, "get_status") else {}
        
        # Create status object
        status = {
            "running": self.running,
            "paused": self.paused,
            "timestamp": datetime.now().isoformat(),
            "components": {
                "knowledge_base": kb_status,
                "ai_trading_agent": ai_agent_status,
                "algorithmic_trading": trading_system_status,
                "data_analysis": data_analysis_status,
                "continuous_improvement": ci_status
            }
        }
        
        return status
    
    def _main_loop(self):
        """Main processing loop for the trading system."""
        logger.info("Entering main processing loop")
        
        try:
            # Initialize trading session
            self._initialize_session()
            
            # Main loop
            while self.running:
                try:
                    # Check if paused
                    if self.paused:
                        time.sleep(1)
                        continue
                    
                    # Process commands
                    self._process_commands()
                    
                    # Update market data
                    self._update_market_data()
                    
                    # Generate trading signals
                    self._generate_trading_signals()
                    
                    # Execute trading signals
                    self._execute_trading_signals()
                    
                    # Process feedback
                    self._process_feedback()
                    
                    # Sleep to prevent high CPU usage
                    time.sleep(self.config.get("main_loop_interval", 60))
                    
                except Exception as e:
                    logger.error(f"Error in main loop: {str(e)}")
                    time.sleep(10)  # Sleep to prevent rapid error loops
            
        except Exception as e:
            logger.error(f"Fatal error in main loop: {str(e)}")
        
        logger.info("Exiting main processing loop")
    
    def _initialize_session(self):
        """Initialize the trading session."""
        logger.info("Initializing trading session")
        
        try:
            # Connect to Interactive Brokers
            self.trading_system.connect(
                host=self.config.get("ib_host", "127.0.0.1"),
                port=self.config.get("ib_port", 7497),
                client_id=self.config.get("ib_client_id", 1)
            )
            
            # Initialize market data
            self._update_market_data(force_full_update=True)
            
            # Initialize portfolio
            self.trading_system.update_portfolio()
            
            # Initialize account information
            self.trading_system.update_account_info()
            
            # Detect market regime
            self._detect_market_regime()
            
            logger.info("Trading session initialized")
            
        except Exception as e:
            logger.error(f"Error initializing trading session: {str(e)}")
            raise
    
    def _process_commands(self):
        """Process commands from the command queue."""
        # Process up to 10 commands per cycle
        for _ in range(10):
            try:
                # Get command from queue with timeout
                try:
                    cmd = self.command_queue.get(block=False)
                except queue.Empty:
                    break
                
                # Process command
                try:
                    result = self._execute_command_internal(cmd["command"], **cmd["args"])
                    cmd["result_queue"].put(result)
                except Exception as e:
                    logger.error(f"Error executing command {cmd['command']}: {str(e)}")
                    cmd["result_queue"].put({"status": "error", "message": str(e)})
                
            except Exception as e:
                logger.error(f"Error processing commands: {str(e)}")
    
    def _execute_command_internal(self, command, **kwargs):
        """
        Execute a command internally.
        
        Args:
            command: Command to execute
            **kwargs: Command arguments
        
        Returns:
            Command result
        """
        if command == "get_status":
            return {"status": "success", "data": self.get_status()}
        
        elif command == "get_portfolio":
            portfolio = self.trading_system.get_portfolio()
            return {"status": "success", "data": portfolio}
        
        elif command == "get_account_info":
            account_info = self.trading_system.get_account_info()
            return {"status": "success", "data": account_info}
        
        elif command == "get_market_data":
            symbol = kwargs.get("symbol")
            if not symbol:
                return {"status": "error", "message": "Symbol is required"}
            
            start_date = kwargs.get("start_date")
            end_date = kwargs.get("end_date")
            
            market_data = self.kb.get_market_data(symbol, start_date, end_date)
            return {"status": "success", "data": market_data.to_dict(orient="records")}
        
        elif command == "place_order":
            symbol = kwargs.get("symbol")
            if not symbol:
                return {"status": "error", "message": "Symbol is required"}
            
            order_type = kwargs.get("order_type", "MARKET")
            direction = kwargs.get("direction", "BUY")
            quantity = kwargs.get("quantity")
            if not quantity:
                return {"status": "error", "message": "Quantity is required"}
            
            price = kwargs.get("price")
            if order_type != "MARKET" and not price:
                return {"status": "error", "message": "Price is required for non-market orders"}
            
            order_id = self.trading_system.place_order(
                symbol=symbol,
                order_type=order_type,
                direction=direction,
                quantity=quantity,
                price=price
            )
            
            return {"status": "success", "data": {"order_id": order_id}}
        
        elif command == "cancel_order":
            order_id = kwargs.get("order_id")
            if not order_id:
                return {"status": "error", "message": "Order ID is required"}
            
            success = self.trading_system.cancel_order(order_id)
            
            if success:
                return {"status": "success", "message": f"Order {order_id} cancelled"}
            else:
                return {"status": "error", "message": f"Failed to cancel order {order_id}"}
        
        elif command == "get_orders":
            orders = self.kb.get_orders()
            return {"status": "success", "data": orders}
        
        elif command == "get_performance_metrics":
            start_date = kwargs.get("start_date")
            end_date = kwargs.get("end_date")
            
            metrics = self.continuous_improvement.calculate_performance_metrics(start_date, end_date)
            return {"status": "success", "data": metrics}
        
        else:
            return {"status": "error", "message": f"Unknown command: {command}"}
    
    def _update_market_data(self, force_full_update=False):
        """
        Update market data.
        
        Args:
            force_full_update: Whether to force a full update
        """
        try:
            # Determine update type
            if force_full_update:
                update_type = "full"
            else:
                # Check last update time
                last_update = self.kb.get_last_market_data_update()
                now = datetime.now()
                
                if not last_update or (now - last_update).total_seconds() > self.config.get("full_update_interval", 86400):
                    update_type = "full"
                else:
                    update_type = "incremental"
            
            logger.info(f"Updating market data ({update_type} update)")
            
            # Get symbols to update
            if update_type == "full":
                # Get all symbols from universe
                symbols = self.config.get("symbol_universe", [])
                
                # Add symbols from current positions
                portfolio = self.trading_system.get_portfolio()
                position_symbols = [position["symbol"] for position in portfolio]
                
                # Combine and deduplicate
                symbols = list(set(symbols + position_symbols))
            else:
                # Get symbols from current positions and watchlist
                portfolio = self.trading_system.get_portfolio()
                position_symbols = [position["symbol"] for position in portfolio]
                watchlist = self.config.get("watchlist", [])
                
                # Combine and deduplicate
                symbols = list(set(position_symbols + watchlist))
            
            # Update market data
            self.data_analysis.update_market_data(symbols)
            
            # Update last update time
            self.kb.set_last_market_data_update(datetime.now())
            
            logger.info(f"Market data updated for {len(symbols)} symbols")
            
        except Exception as e:
            logger.error(f"Error updating market data: {str(e)}")
    
    def _generate_trading_signals(self):
        """Generate trading signals."""
        try:
            logger.info("Generating trading signals")
            
            # Get current market regime
            market_regime = self._detect_market_regime()
            
            # Get symbols to analyze
            portfolio = self.trading_system.get_portfolio()
            position_symbols = [position["symbol"] for position in portfolio]
            watchlist = self.config.get("watchlist", [])
            
            # Combine and deduplicate
            symbols = list(set(position_symbols + watchlist))
            
            # Generate signals for each symbol
            signals = []
            for symbol in symbols:
                # Select optimal strategy for symbol
                strategy = self.continuous_improvement.select_optimal_strategy(symbol, market_regime)
                
                # Generate signal
                signal = self.ai_agent.generate_signal(symbol, strategy)
                
                if signal:
                    # Evaluate signal quality
                    quality = self.continuous_improvement.evaluate_signal_quality(signal)
                    signal["quality"] = quality
                    
                    # Add to signals list if quality is above threshold
                    if quality >= self.config.get("signal_quality_threshold", 0.6):
                        signals.append(signal)
            
            # Store signals in Knowledge Base
            for signal in signals:
                self.kb.store_signal(signal)
            
            logger.info(f"Generated {len(signals)} trading signals")
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating trading signals: {str(e)}")
            return []
    
    def _execute_trading_signals(self):
        """Execute trading signals."""
        try:
            logger.info("Executing trading signals")
            
            # Get recent signals
            signals = self.kb.get_recent_signals(
                max_age=self.config.get("signal_max_age", 3600)
            )
            
            # Filter signals that haven't been executed
            signals = [signal for signal in signals if not signal.get("executed", False)]
            
            if not signals:
                logger.info("No signals to execute")
                return
            
            # Get account information
            account_info = self.trading_system.get_account_info()
            
            # Execute each signal
            executed_count = 0
            for signal in signals:
                try:
                    # Optimize position size
                    position_size = self.continuous_improvement.optimize_position_size(signal, account_info)
                    
                    # Execute signal
                    order_id = self.trading_system.execute_signal(signal, position_size)
                    
                    if order_id:
                        # Mark signal as executed
                        signal["executed"] = True
                        signal["order_id"] = order_id
                        signal["execution_time"] = datetime.now().isoformat()
                        
                        # Update signal in Knowledge Base
                        self.kb.update_signal(signal)
                        
                        executed_count += 1
                        
                except Exception as e:
                    logger.error(f"Error executing signal {signal.get('id', '')}: {str(e)}")
            
            logger.info(f"Executed {executed_count} trading signals")
            
        except Exception as e:
            logger.error(f"Error executing trading signals: {str(e)}")
    
    def _process_feedback(self):
        """Process feedback from completed trades."""
        try:
            logger.info("Processing trade feedback")
            
            # Get completed trades
            completed_trades = self.kb.get_completed_trades(
                processed=False,
                max_age=self.config.get("feedback_max_age", 86400 * 7)
            )
            
            if not completed_trades:
                logger.info("No completed trades to process")
                return
            
            # Process each trade
            processed_count = 0
            for trade in completed_trades:
                try:
                    # Process feedback
                    self.continuous_improvement.process_trade_feedback(trade)
                    
                    # Mark trade as processed
                    trade["feedback_processed"] = True
                    trade["feedback_time"] = datetime.now().isoformat()
                    
                    # Update trade in Knowledge Base
                    self.kb.update_trade(trade)
                    
                    processed_count += 1
                    
                except Exception as e:
                    logger.error(f"Error processing feedback for trade {trade.get('id', '')}: {str(e)}")
            
            logger.info(f"Processed feedback for {processed_count} trades")
            
        except Exception as e:
            logger.error(f"Error processing trade feedback: {str(e)}")
    
    def _detect_market_regime(self):
        """
        Detect the current market regime.
        
        Returns:
            Market regime information
        """
        try:
            logger.info("Detecting market regime")
            
            # Get market data for index ETFs
            market_data = {}
            for symbol in ["SPY", "QQQ", "IWM", "DIA"]:
                data = self.kb.get_market_data(
                    symbol=symbol,
                    start_date=datetime.now() - timedelta(days=30),
                    end_date=datetime.now()
                )
                
                if not data.empty:
                    market_data[symbol] = data
            
            # Detect market regime
            regime_info = self.continuous_improvement.detect_market_regime(market_data)
            
            # Store regime in Knowledge Base
            self.kb.store_market_regime(regime_info)
            
            logger.info(f"Detected market regime: {regime_info.get('regime', 'unknown')}")
            
            return regime_info
            
        except Exception as e:
            logger.error(f"Error detecting market regime: {str(e)}")
            return {"regime": "unknown", "confidence": 0.0, "timestamp": datetime.now().isoformat()}
    
    def _shutdown_components(self):
        """Shutdown all components."""
        logger.info("Shutting down components")
        
        try:
            # Shutdown Continuous Improvement
            if hasattr(self.continuous_improvement, "shutdown"):
                self.continuous_improvement.shutdown()
            
            # Disconnect from Interactive Brokers
            if hasattr(self.trading_system, "disconnect"):
                self.trading_system.disconnect()
            
            # Shutdown other components if needed
            
            logger.info("Components shutdown complete")
            
        except Exception as e:
            logger.error(f"Error shutting down components: {str(e)}")
    
    def _load_config(self, config_file):
        """
        Load configuration from file.
        
        Args:
            config_file: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        # Default configuration
        default_config = {
            "ib_host": "127.0.0.1",
            "ib_port": 7497,
            "ib_client_id": 1,
            "main_loop_interval": 60,
            "full_update_interval": 86400,
            "signal_quality_threshold": 0.6,
            "signal_max_age": 3600,
            "feedback_max_age": 604800,
            "symbol_universe": [
                "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "NVDA", "AMD",
                "JPM", "BAC", "WFC", "GS", "MS", "C",
                "JNJ", "PFE", "MRK", "ABBV", "LLY", "BMY",
                "XOM", "CVX", "COP", "EOG", "SLB",
                "HD", "WMT", "COST", "TGT", "AMZN",
                "SPY", "QQQ", "IWM", "DIA"
            ],
            "watchlist": [
                "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "NVDA", "AMD",
                "SPY", "QQQ", "IWM", "DIA"
            ]
        }
        
        # Load configuration from file if provided
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    file_config = json.load(f)
                
                # Merge configurations
                config = {**default_config, **file_config}
                
                logger.info(f"Loaded configuration from {config_file}")
                
                return config
                
            except Exception as e:
                logger.error(f"Error loading configuration from {config_file}: {str(e)}")
        
        logger.info("Using default configuration")
        return default_config
    
    def _signal_handler(self, sig, frame):
        """
        Handle signals.
        
        Args:
            sig: Signal number
            frame: Current stack frame
        """
        logger.info(f"Received signal {sig}")
        
        if sig in (signal.SIGINT, signal.SIGTERM):
            logger.info("Stopping trading system")
            self.stop()


def main():
    """Main entry point for the application."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="DeepResearch 4.5 Trading System")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--mode", choices=["live", "paper"], default="paper", help="Trading mode")
    args = parser.parse_args()
    
    # Create trading system
    trading_system = TradingSystem(config_file=args.config)
    
    # Set trading mode
    if args.mode == "live":
        trading_system.trading_system.set_mode("live")
    else:
        trading_system.trading_system.set_mode("paper")
    
    try:
        # Start trading system
        trading_system.start()
        
        # Keep main thread alive
        while trading_system.running:
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    
    finally:
        # Stop trading system
        trading_system.stop()


if __name__ == "__main__":
    main()
