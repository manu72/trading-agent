# Interactive Brokers API Research Notes

## Overview
The TWS API (Trader Workstation API) is an interface that allows clients to automate trading strategies, request market data, and monitor account balances and portfolios in real-time. This API connects to either the TWS (Trader Workstation) or IB Gateway applications.

## Key Information
- **Current Version**: 9.72+
- **Requirements**: 
  - Network connectivity to a running instance of TWS or IB Gateway
  - TWS build 952.x or higher
  - Working knowledge of programming languages (Java, VB, C#, C++, Python)

## Limitations
- The API is designed to automate operations normally performed manually in TWS
- Limited to fifty messages per second from client side (all connected clients combined)
- No limits on messages from TWS to client application
- If a feature is not available in TWS, it will not be available via the API

## Available Programming Languages
- Java
- VB
- C#
- C++
- Python

## Key Components to Research Further
1. **Connectivity** - How to establish and maintain connection to TWS/IB Gateway
2. **Financial Instruments (Contracts)** - How to define and work with different financial instruments
3. **Orders** - Order types, placement, modification, and cancellation
4. **Streaming Market Data** - Real-time data feeds
5. **Historical Market Data** - Historical price and volume data
6. **Account & Portfolio Data** - Account balances, positions, and P&L
7. **Cryptocurrency** - Specific functionality for crypto trading
8. **Error Handling** - How to handle and respond to errors
9. **Market Scanners** - Tools for scanning markets based on criteria

## Sample Code
The TWS API installation includes sample projects called "Testbed" in the samples folder, which demonstrate API functionality. These samples are available for all supported programming languages.

## Next Steps for Research
1. Explore the connectivity section to understand how to establish a connection
2. Research order types and execution capabilities
3. Investigate market data retrieval methods
4. Understand account and portfolio management functions
5. Look into error handling and risk management capabilities
6. Research Python-specific implementation details and best practices
