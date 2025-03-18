# DeepResearch 4.5 Trading System - Installation Guide

This guide provides step-by-step instructions for installing and setting up the DeepResearch 4.5 Trading System.

## Prerequisites

Before installing the trading system, ensure you have the following:

- Python 3.8 or higher
- Interactive Brokers account with API access
- Interactive Brokers Trader Workstation (TWS) or IB Gateway
- Git (for cloning the repository)
- Minimum 8GB RAM
- 50GB available disk space
- Internet connection

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/manu72/trading-agent.git
cd trading-agent
```

### 2. Create a Virtual Environment (Recommended)

```bash
# Create a virtual environment [note must be Python 3.11 or below for TensorFlow support]
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

The requirements.txt file includes the following dependencies:

- pandas
- numpy
- matplotlib
- seaborn
- plotly
- scikit-learn
- tensorflow
- ibapi (Interactive Brokers API)
- requests
- beautifulsoup4
- sqlalchemy
- pytest
- jupyter

### 4. Set Up Interactive Brokers

1. Install Interactive Brokers Trader Workstation (TWS) or IB Gateway from the [Interactive Brokers website](https://www.interactivebrokers.com/en/index.php?f=14099)

2. Configure API settings in TWS/IB Gateway:

   - Go to File > Global Configuration > API > Settings
   - Enable Active X and Socket Clients
   - Set the Socket Port (default: 7497 for paper trading, 7496 for live trading)
   - Add your IP address to the trusted IPs list
   - Check "Allow connections from localhost"
   - Uncheck "Read-Only API"

3. Restart TWS/IB Gateway for the changes to take effect

### 5. Configure the Trading System

1. Copy the example configuration file:

   ```bash
   cp config.example.json config.json
   ```

2. Edit the configuration file to match your preferences:

   ```bash
   # Use your favorite text editor
   nano config.json
   ```

3. Update the following settings:
   - `ib_host`: IP address of the TWS/IB Gateway (usually "127.0.0.1")
   - `ib_port`: Port number for API connections (7497 for paper trading, 7496 for live trading)
   - `ib_client_id`: Client ID for API connection (must be unique)
   - `symbol_universe`: List of symbols to include in the trading universe
   - `watchlist`: List of symbols to actively monitor
   - Risk parameters according to your risk tolerance
   - Strategy parameters according to your trading preferences

### 6. Verify Installation

Run the system status check to verify that everything is installed correctly:

```bash
python start.py status
```

If the installation is successful, you should see the system status information.

## Setting Up for Development (Optional)

If you plan to modify or extend the trading system, follow these additional steps:

### 1. Install Development Dependencies

```bash
pip install -r requirements-dev.txt
```

### 2. Set Up Pre-commit Hooks

```bash
pre-commit install
```

### 3. Run Tests

```bash
pytest
```

## Troubleshooting

### Common Installation Issues

#### Package Installation Failures

If you encounter issues installing packages, try:

```bash
pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir
```

#### Interactive Brokers API Connection Issues

If you cannot connect to Interactive Brokers:

1. Verify that TWS/IB Gateway is running
2. Check that API connections are enabled
3. Ensure the port numbers match
4. Check firewall settings
5. Verify that your IP address is in the trusted IPs list

#### Python Version Issues

If you encounter Python version compatibility issues:

1. Verify your Python version:

   ```bash
   python --version
   ```

2. If your Python version is below 3.8, install a newer version from the [Python website](https://www.python.org/downloads/)

## Next Steps

After successful installation, refer to the [User Documentation](user_documentation.md) for instructions on how to use the trading system.
