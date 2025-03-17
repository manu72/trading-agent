# DeepResearch 4.5 Trading System - API Documentation

This document provides detailed information about the APIs and interfaces available in the DeepResearch 4.5 Trading System.

## Table of Contents

1. [Knowledge Base API](#knowledge-base-api)
2. [AI Trading Agent API](#ai-trading-agent-api)
3. [Algorithmic Trading System API](#algorithmic-trading-system-api)
4. [Data Analysis API](#data-analysis-api)
5. [Continuous Improvement API](#continuous-improvement-api)
6. [Main Trading System API](#main-trading-system-api)

## Knowledge Base API

The Knowledge Base component provides methods for storing and retrieving market data, trading signals, orders, and performance metrics.

### Initialization

```python
from trading_system.knowledge_base import get_knowledge_base

# Get singleton instance
kb = get_knowledge_base()
```

### Market Data Methods

```python
# Store market data
kb.store_market_data(symbol, data_frame)

# Get market data
data = kb.get_market_data(symbol, start_date=None, end_date=None)

# Get last market data update time
last_update = kb.get_last_market_data_update()

# Set last market data update time
kb.set_last_market_data_update(datetime.now())
```

### Trading Signal Methods

```python
# Store trading signal
signal_id = kb.store_signal(signal_dict)

# Get signal by ID
signal = kb.get_signal_by_id(signal_id)

# Get recent signals
signals = kb.get_recent_signals(max_age=3600)

# Update signal
kb.update_signal(signal_dict)
```

### Order Methods

```python
# Store order
order_id = kb.store_order(order_dict)

# Get order by ID
order = kb.get_order_by_id(order_id)

# Get orders
orders = kb.get_orders(symbol=None, start_date=None, end_date=None)

# Update order
kb.update_order(order_dict)
```

### Trade Methods

```python
# Store trade
trade_id = kb.store_trade(trade_dict)

# Get trade by ID
trade = kb.get_trade_by_id(trade_id)

# Get completed trades
trades = kb.get_completed_trades(processed=False, max_age=604800)

# Update trade
kb.update_trade(trade_dict)
```

### Performance Methods

```python
# Store performance metrics
kb.store_performance_metrics(metrics_dict)

# Get performance metrics
metrics = kb.get_performance_metrics(start_date=None, end_date=None)
```

### Market Regime Methods

```python
# Store market regime
kb.store_market_regime(regime_dict)

# Get current market regime
regime = kb.get_current_market_regime()

# Get market regime history
regimes = kb.get_market_regime_history(start_date=None, end_date=None)
```

## AI Trading Agent API

The AI Trading Agent component provides methods for analyzing market data and generating trading signals.

### Initialization

```python
from trading_system.ai_trading_agent import get_ai_trading_agent

# Get singleton instance
agent = get_ai_trading_agent()
```

### Signal Generation Methods

```python
# Generate trading signal
signal = agent.generate_signal(symbol, strategy=None)

# Generate signals for multiple symbols
signals = agent.generate_signals(symbols, strategy=None)
```

### Strategy Methods

```python
# Execute momentum strategy
signal = agent.execute_momentum_strategy(symbol, lookback_period=20)

# Execute sector rotation strategy
signals = agent.execute_sector_rotation_strategy(sectors, top_count=3)

# Execute swing trading strategy
signal = agent.execute_swing_trading_strategy(symbol)

# Execute mean reversion strategy
signal = agent.execute_mean_reversion_strategy(symbol, lookback_period=20)
```

### Analysis Methods

```python
# Analyze trend
trend_info = agent.analyze_trend(symbol, timeframe='daily')

# Analyze volatility
volatility_info = agent.analyze_volatility(symbol, lookback_period=20)

# Analyze support/resistance levels
levels = agent.analyze_support_resistance(symbol)

# Analyze correlation
correlation = agent.analyze_correlation(symbol1, symbol2)
```

## Algorithmic Trading System API

The Algorithmic Trading System component provides methods for executing trades through Interactive Brokers.

### Initialization

```python
from trading_system.algorithmic_trading import get_algorithmic_trading_system

# Get singleton instance
trading_system = get_algorithmic_trading_system()
```

### Connection Methods

```python
# Connect to Interactive Brokers
trading_system.connect(host='127.0.0.1', port=7497, client_id=1)

# Disconnect from Interactive Brokers
trading_system.disconnect()

# Check connection status
is_connected = trading_system.is_connected()
```

### Trading Mode Methods

```python
# Set trading mode
trading_system.set_mode('paper')  # or 'live'

# Get current trading mode
mode = trading_system.get_mode()
```

### Order Methods

```python
# Place order
order_id = trading_system.place_order(
    symbol='AAPL',
    order_type='MARKET',  # or 'LIMIT', 'STOP', etc.
    direction='BUY',      # or 'SELL'
    quantity=10,
    price=None            # Required for LIMIT orders
)

# Execute trading signal
order_id = trading_system.execute_signal(signal, position_size=0.05)

# Cancel order
success = trading_system.cancel_order(order_id)

# Get order status
status = trading_system.get_order_status(order_id)
```

### Portfolio Methods

```python
# Get portfolio
portfolio = trading_system.get_portfolio()

# Get position
position = trading_system.get_position(symbol)

# Close position
order_id = trading_system.close_position(symbol)

# Close all positions
order_ids = trading_system.close_all_positions()
```

### Account Methods

```python
# Get account info
account_info = trading_system.get_account_info()

# Update account info
trading_system.update_account_info()

# Update portfolio
trading_system.update_portfolio()
```

## Data Analysis API

The Data Analysis component provides methods for gathering and analyzing market data.

### Initialization

```python
from trading_system.data_analysis import get_data_analysis_system

# Get singleton instance
data_analysis = get_data_analysis_system()
```

### Data Update Methods

```python
# Update market data
data_analysis.update_market_data(symbols)

# Update news data
data_analysis.update_news_data(symbols)

# Update social sentiment data
data_analysis.update_social_sentiment_data(symbols)

# Update economic indicators
data_analysis.update_economic_indicators()

# Update SEC filings
data_analysis.update_sec_filings(symbols)
```

### Analysis Methods

```python
# Analyze technical indicators
indicators = data_analysis.analyze_technical_indicators(symbol)

# Analyze fundamentals
fundamentals = data_analysis.analyze_fundamentals(symbol)

# Analyze sentiment
sentiment = data_analysis.analyze_sentiment(symbol)

# Analyze correlations
correlations = data_analysis.analyze_correlations(symbols)

# Analyze volatility
volatility = data_analysis.analyze_volatility(symbol)
```

### Scheduling Methods

```python
# Schedule data update
data_analysis.schedule_update(data_type, interval, symbols=None)

# Cancel scheduled update
data_analysis.cancel_scheduled_update(update_id)
```

## Continuous Improvement API

The Continuous Improvement component provides methods for learning from performance and adapting strategies.

### Initialization

```python
from trading_system.continuous_improvement import get_continuous_improvement_system

# Get singleton instance
ci_system = get_continuous_improvement_system()
```

### Signal Evaluation Methods

```python
# Evaluate signal quality
quality_score = ci_system.evaluate_signal_quality(signal)
```

### Position Sizing Methods

```python
# Optimize position size
position_size = ci_system.optimize_position_size(signal, account_info)
```

### Exit Timing Methods

```python
# Recommend exit timing
recommendation = ci_system.recommend_exit_timing(position, market_data)
```

### Market Regime Methods

```python
# Detect market regime
regime_info = ci_system.detect_market_regime(market_data)
```

### Strategy Selection Methods

```python
# Select optimal strategy
strategy = ci_system.select_optimal_strategy(symbol, market_regime)
```

### Performance Methods

```python
# Calculate performance metrics
metrics = ci_system.calculate_performance_metrics(start_date, end_date)
```

### Feedback Methods

```python
# Process trade feedback
ci_system.process_trade_feedback(trade_result)
```

### System Methods

```python
# Shutdown
ci_system.shutdown()
```

## Main Trading System API

The Main Trading System component provides methods for managing the overall trading system.

### Initialization

```python
from trading_system.main import get_trading_system

# Get singleton instance
trading_system = get_trading_system(config_file='config.json')
```

### System Control Methods

```python
# Start trading system
trading_system.start()

# Stop trading system
trading_system.stop()

# Pause trading system
trading_system.pause()

# Resume trading system
trading_system.resume()
```

### Status Methods

```python
# Get system status
status = trading_system.get_status()
```

### Command Methods

```python
# Execute command
result = trading_system.execute_command(command, **kwargs)
```

### Available Commands

```python
# Get status
result = trading_system.execute_command('get_status')

# Get portfolio
result = trading_system.execute_command('get_portfolio')

# Get account info
result = trading_system.execute_command('get_account_info')

# Get market data
result = trading_system.execute_command(
    'get_market_data',
    symbol='AAPL',
    start_date='2023-01-01',
    end_date='2023-01-31'
)

# Place order
result = trading_system.execute_command(
    'place_order',
    symbol='AAPL',
    order_type='MARKET',
    direction='BUY',
    quantity=10,
    price=None
)

# Cancel order
result = trading_system.execute_command(
    'cancel_order',
    order_id='12345'
)

# Get orders
result = trading_system.execute_command('get_orders')

# Get performance metrics
result = trading_system.execute_command(
    'get_performance_metrics',
    start_date='2023-01-01',
    end_date='2023-01-31'
)
```

## Command Line Interface

The system also provides a command-line interface through the `start.py` script:

```bash
# Start trading system
python start.py --mode paper start

# Stop trading system
python start.py stop

# Get status
python start.py status

# Get portfolio
python start.py portfolio

# Get performance metrics
python start.py performance --days 30

# Place order
python start.py order --symbol AAPL --direction BUY --quantity 10 --type MARKET
```
