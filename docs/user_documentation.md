# DeepResearch 4.5 Trading System - User Documentation

## Overview

The DeepResearch 4.5 Trading System is a comprehensive algorithmic trading platform designed to achieve significant returns through momentum investing, sector rotation, swing trading, and other quantitative strategies. The system integrates with Interactive Brokers for trade execution and implements a continuous improvement mechanism for adaptive learning.

This documentation provides instructions for installation, configuration, and operation of the trading system.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Getting Started](#getting-started)
5. [Command Line Interface](#command-line-interface)
6. [System Architecture](#system-architecture)
7. [Trading Strategies](#trading-strategies)
8. [Risk Management](#risk-management)
9. [Performance Monitoring](#performance-monitoring)
10. [Troubleshooting](#troubleshooting)
11. [FAQ](#faq)

## System Requirements

- Python 3.8 or higher
- Interactive Brokers account with API access
- Interactive Brokers Trader Workstation (TWS) or IB Gateway
- Minimum 8GB RAM
- 50GB available disk space
- Internet connection

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/deepresearch-trading-system.git
cd deepresearch-trading-system
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Set Up Interactive Brokers

1. Install Interactive Brokers Trader Workstation (TWS) or IB Gateway
2. Configure API settings in TWS/IB Gateway:
   - Enable API connections
   - Set API port (default: 7497 for paper trading, 7496 for live trading)
   - Add your IP address to the trusted IPs list

## Configuration

The system is configured through the `config.json` file. The main configuration parameters are:

### Connection Settings

```json
{
  "ib_host": "127.0.0.1",
  "ib_port": 7497,
  "ib_client_id": 1
}
```

- `ib_host`: IP address of the TWS/IB Gateway (usually localhost)
- `ib_port`: Port number for API connections (7497 for paper trading, 7496 for live trading)
- `ib_client_id`: Client ID for API connection (must be unique)

### Trading Parameters

```json
{
  "symbol_universe": ["AAPL", "MSFT", "AMZN", "GOOGL", "..."],
  "watchlist": ["AAPL", "MSFT", "AMZN", "GOOGL", "..."],
  "signal_quality_threshold": 0.6,
  "main_loop_interval": 60
}
```

- `symbol_universe`: List of symbols to include in the trading universe
- `watchlist`: List of symbols to actively monitor
- `signal_quality_threshold`: Minimum quality score for trading signals (0.0-1.0)
- `main_loop_interval`: Interval in seconds for the main processing loop

### Risk Parameters

```json
{
  "risk_parameters": {
    "max_position_size_pct": 0.05,
    "max_sector_exposure_pct": 0.3,
    "max_leverage": 1.0,
    "stop_loss_pct": 0.05,
    "profit_target_pct": 0.15
  }
}
```

- `max_position_size_pct`: Maximum position size as percentage of portfolio
- `max_sector_exposure_pct`: Maximum exposure to a single sector
- `max_leverage`: Maximum leverage to use
- `stop_loss_pct`: Default stop loss percentage
- `profit_target_pct`: Default profit target percentage

### Strategy Parameters

```json
{
  "strategy_parameters": {
    "momentum": {
      "lookback_period": 20,
      "momentum_threshold": 0.05
    },
    "sector_rotation": {
      "top_sectors_count": 3,
      "rotation_period_days": 30
    },
    "swing_trading": {
      "overbought_threshold": 70,
      "oversold_threshold": 30
    },
    "mean_reversion": {
      "deviation_threshold": 2.0,
      "lookback_period": 20
    }
  }
}
```

Each strategy has its own set of parameters that can be customized.

## Getting Started

### Starting the System

To start the trading system in paper trading mode:

```bash
python start.py --mode paper start
```

To start in live trading mode:

```bash
python start.py --mode live start
```

### Stopping the System

To stop the trading system:

```bash
python start.py stop
```

### Checking System Status

To check the status of the trading system:

```bash
python start.py status
```

## Command Line Interface

The system provides a command-line interface for various operations:

### Get Portfolio

```bash
python start.py portfolio
```

### Get Performance Metrics

```bash
python start.py performance --days 30
```

### Place an Order

```bash
python start.py order --symbol AAPL --direction BUY --quantity 10 --type MARKET
```

For limit orders:

```bash
python start.py order --symbol AAPL --direction BUY --quantity 10 --type LIMIT --price 150.00
```

## System Architecture

The DeepResearch 4.5 Trading System consists of five main components:

1. **Trading Knowledge Base**: Central repository for market data, analysis, and trading knowledge
2. **AI Trading Agent**: Decision-making core that generates trading signals
3. **Algorithmic Trading System**: Execution engine that interfaces with Interactive Brokers
4. **Data Analysis and Web Monitoring**: System for gathering and analyzing market data
5. **Continuous Improvement Mechanism**: Framework for system optimization and learning

## Trading Strategies

The system implements four main trading strategies:

### Momentum Investing

Identifies securities with strong price momentum and trends. The strategy buys securities that have been rising and sells those that have been falling, based on the premise that securities that have performed well (poorly) in the past will continue to perform well (poorly) in the near future.

Key parameters:
- `lookback_period`: Period for calculating momentum
- `momentum_threshold`: Minimum momentum score for signal generation

### Sector Rotation

Capitalizes on the cyclical nature of economic sectors. The strategy moves investments from sectors that are declining in performance to sectors that are expected to outperform in the current or upcoming economic environment.

Key parameters:
- `top_sectors_count`: Number of top-performing sectors to invest in
- `rotation_period_days`: Period for sector rotation

### Swing Trading

Identifies short-term trading opportunities based on overbought/oversold conditions. The strategy aims to capture "swings" in price movements over a period of days or weeks.

Key parameters:
- `overbought_threshold`: RSI threshold for overbought condition
- `oversold_threshold`: RSI threshold for oversold condition

### Mean Reversion

Identifies securities that have deviated significantly from their mean and are expected to revert back. The strategy buys securities that have fallen significantly below their historical average and sells those that have risen significantly above.

Key parameters:
- `deviation_threshold`: Standard deviation threshold for signal generation
- `lookback_period`: Period for calculating mean and standard deviation

## Risk Management

The system implements a comprehensive risk management framework:

### Position Sizing

- Dynamic position sizing based on signal quality and market conditions
- Maximum position size limit as percentage of portfolio

### Sector Exposure Limits

- Maximum exposure to any single sector
- Diversification across multiple sectors

### Stop Loss and Take Profit

- Automatic stop loss orders for all positions
- Take profit targets based on risk-reward ratio
- Trailing stops for capturing extended moves

### Leverage Control

- Maximum leverage limit
- Gradual increase/decrease based on performance

### Drawdown Protection

- Automatic reduction of position sizes during drawdowns
- Trading pause during extreme market conditions

## Performance Monitoring

The system provides comprehensive performance monitoring:

### Key Performance Indicators

- Total Return
- Win Rate
- Profit Factor
- Sharpe Ratio
- Maximum Drawdown
- Average Win/Loss
- Win/Loss Ratio

### Performance Reports

To generate a performance report:

```bash
python start.py performance --days 30
```

## Troubleshooting

### Connection Issues

If you experience connection issues with Interactive Brokers:

1. Verify that TWS/IB Gateway is running
2. Check that API connections are enabled in TWS/IB Gateway
3. Verify that the port number in `config.json` matches the port in TWS/IB Gateway
4. Ensure your IP address is in the trusted IPs list in TWS/IB Gateway

### System Not Starting

If the system fails to start:

1. Check the log file (`trading_system.log`) for error messages
2. Verify that all dependencies are installed
3. Ensure that the configuration file is valid JSON
4. Check that Interactive Brokers is properly configured

### Trading Issues

If the system is not trading as expected:

1. Check the signal quality threshold in the configuration
2. Verify that the symbols in your watchlist are valid
3. Ensure that your Interactive Brokers account has sufficient funds
4. Check the risk parameters in the configuration

## FAQ

### Can I use the system with a different broker?

Currently, the system is designed to work with Interactive Brokers. Support for other brokers may be added in future versions.

### How much capital do I need to use the system?

The system is designed to work with any account size, but a minimum of $25,000 USD is recommended for US-based traders to avoid pattern day trader restrictions.

### Can I customize the trading strategies?

Yes, all trading strategies can be customized through the `config.json` file. You can adjust parameters such as lookback periods, thresholds, and more.

### How does the continuous improvement mechanism work?

The continuous improvement mechanism uses machine learning to analyze past trades and optimize strategy parameters, position sizing, and risk management. It adapts to changing market conditions and improves performance over time.

### Is the system suitable for beginners?

The system is designed for experienced traders with a good understanding of algorithmic trading, risk management, and financial markets. Beginners should start with paper trading and gradually transition to live trading as they gain experience.

### How often does the system trade?

The trading frequency depends on the configured strategies and market conditions. The system can execute trades from multiple times per day to several times per week.

### Can I run the system on a cloud server?

Yes, the system can be run on a cloud server with a stable internet connection. Ensure that the server can connect to Interactive Brokers API.

### How do I update the system?

To update the system, pull the latest changes from the repository and restart the system:

```bash
git pull
pip install -r requirements.txt
python start.py start
```
