#!/usr/bin/env python3
"""
DeepResearch 4.5 Trading System - Startup Script

This script provides a command-line interface for starting and managing
the DeepResearch 4.5 Trading System.
"""

import os
import sys
import argparse
import logging
import time
import json
import signal
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import trading system
from src.main import get_trading_system

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_system.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Startup")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="DeepResearch 4.5 Trading System")
    
    # Main arguments
    parser.add_argument("--config", default="config.json", help="Path to configuration file")
    parser.add_argument("--mode", choices=["live", "paper"], default="paper", help="Trading mode")
    
    # Commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Start command
    start_parser = subparsers.add_parser("start", help="Start the trading system")
    
    # Stop command
    stop_parser = subparsers.add_parser("stop", help="Stop the trading system")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Get trading system status")
    
    # Portfolio command
    portfolio_parser = subparsers.add_parser("portfolio", help="Get current portfolio")
    
    # Performance command
    performance_parser = subparsers.add_parser("performance", help="Get performance metrics")
    performance_parser.add_argument("--days", type=int, default=30, help="Number of days to analyze")
    
    # Order command
    order_parser = subparsers.add_parser("order", help="Place an order")
    order_parser.add_argument("--symbol", required=True, help="Symbol to trade")
    order_parser.add_argument("--direction", choices=["BUY", "SELL"], required=True, help="Order direction")
    order_parser.add_argument("--quantity", type=int, required=True, help="Order quantity")
    order_parser.add_argument("--type", choices=["MARKET", "LIMIT"], default="MARKET", help="Order type")
    order_parser.add_argument("--price", type=float, help="Order price (required for LIMIT orders)")
    
    return parser.parse_args()

def start_trading_system(args):
    """Start the trading system."""
    logger.info("Starting DeepResearch 4.5 Trading System")
    
    # Get trading system instance
    trading_system = get_trading_system(config_file=args.config)
    
    # Set trading mode
    if args.mode == "live":
        logger.info("Setting LIVE trading mode")
        trading_system.trading_system.set_mode("live")
    else:
        logger.info("Setting PAPER trading mode")
        trading_system.trading_system.set_mode("paper")
    
    # Start trading system
    trading_system.start()
    
    logger.info("Trading system started")
    
    # Keep running until interrupted
    try:
        while trading_system.running:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
        trading_system.stop()
    
    logger.info("Trading system stopped")

def stop_trading_system(args):
    """Stop the trading system."""
    logger.info("Stopping DeepResearch 4.5 Trading System")
    
    # Get trading system instance
    trading_system = get_trading_system(config_file=args.config)
    
    # Stop trading system
    trading_system.stop()
    
    logger.info("Trading system stopped")

def get_system_status(args):
    """Get trading system status."""
    logger.info("Getting trading system status")
    
    # Get trading system instance
    trading_system = get_trading_system(config_file=args.config)
    
    # Get status
    status = trading_system.get_status()
    
    # Print status
    print(json.dumps(status, indent=2))

def get_portfolio(args):
    """Get current portfolio."""
    logger.info("Getting current portfolio")
    
    # Get trading system instance
    trading_system = get_trading_system(config_file=args.config)
    
    # Execute command
    result = trading_system.execute_command("get_portfolio")
    
    # Print result
    if result["status"] == "success":
        print(json.dumps(result["data"], indent=2))
    else:
        print(f"Error: {result.get('message', 'Unknown error')}")

def get_performance(args):
    """Get performance metrics."""
    logger.info(f"Getting performance metrics for the last {args.days} days")
    
    # Get trading system instance
    trading_system = get_trading_system(config_file=args.config)
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - datetime.timedelta(days=args.days)
    
    # Execute command
    result = trading_system.execute_command(
        "get_performance_metrics",
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat()
    )
    
    # Print result
    if result["status"] == "success":
        print(json.dumps(result["data"], indent=2))
    else:
        print(f"Error: {result.get('message', 'Unknown error')}")

def place_order(args):
    """Place an order."""
    logger.info(f"Placing {args.direction} order for {args.quantity} {args.symbol}")
    
    # Get trading system instance
    trading_system = get_trading_system(config_file=args.config)
    
    # Check if price is provided for LIMIT orders
    if args.type == "LIMIT" and args.price is None:
        print("Error: Price is required for LIMIT orders")
        return
    
    # Execute command
    result = trading_system.execute_command(
        "place_order",
        symbol=args.symbol,
        order_type=args.type,
        direction=args.direction,
        quantity=args.quantity,
        price=args.price
    )
    
    # Print result
    if result["status"] == "success":
        print(f"Order placed successfully: {result['data']['order_id']}")
    else:
        print(f"Error: {result.get('message', 'Unknown error')}")

def main():
    """Main entry point."""
    # Parse arguments
    args = parse_arguments()
    
    # Execute command
    if args.command == "start":
        start_trading_system(args)
    elif args.command == "stop":
        stop_trading_system(args)
    elif args.command == "status":
        get_system_status(args)
    elif args.command == "portfolio":
        get_portfolio(args)
    elif args.command == "performance":
        get_performance(args)
    elif args.command == "order":
        place_order(args)
    else:
        # Default to start if no command is provided
        start_trading_system(args)

if __name__ == "__main__":
    main()
