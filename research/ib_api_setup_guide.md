# Interactive Brokers Python API Setup Guide

## Basic Steps for Setting Up IB API Connection

1. **Open an account with Interactive Brokers**
   - IB offers demo accounts for testing
   - Live accounts have a read-only API option for initial testing

2. **Download the IB Python Native API**
   - Available from Interactive Brokers website (Technology → Trading APIs → Get API Software)
   - Requires API version 9.73 or higher
   - Requires Python 3.1 or higher
   - Installation process:
     - Run the downloaded MSI file
     - Navigate to /TWS API/source/pythonclient
     - Run `python3 setup.py install` to install as a package
     - Verify installation with `import ibapi` in Python terminal

3. **Download IB Client Software**
   - Two options:
     - **Trader Workstation (TWS)**: Full trading platform, good for visual confirmation during testing
     - **IB Gateway**: Minimal solution for establishing connection, uses fewer resources
   - Configuration for TWS:
     - Enable ActiveX and Socket Clients
     - Consider enabling Read-Only API for initial testing
     - Note the Socket port (default is usually 7496 for live, 7497 for paper trading)
     - Allow connections from localhost only for security

4. **Choose an IDE**
   - Connection requires open and constant connection (uses threading)
   - Some interactive environments (Jupyter, Spyder) may have issues with EWrapper functions
   - Recommended IDEs: PyCharm, VS Code, Sublime Text

5. **Test Connectivity**
   - Requires implementing both EClient (outgoing calls) and EWrapper (incoming data) classes
   - Uses threading to maintain open connection

## Connection Architecture

- IB API uses a unique connection method:
  - Python script connects to IB client software (TWS or Gateway)
  - Client software acts as intermediary to IB servers
  - Requires constant open connection (threading)
  - EClient functions handle outgoing calls
  - EWrapper functions handle incoming data

## Alternative Libraries

- **IB-insync**: Third-party library that utilizes asyncio for asynchronous single thread interaction
  - May be useful for interactive environments like Jupyter notebooks
  - Simplifies the API interaction

## Next Research Topics

1. Order types and execution
2. Market data retrieval (real-time and historical)
3. Account and portfolio management
4. Position tracking and risk management
5. Error handling
6. Implementation of trading strategies
7. Integration with other frameworks for analysis
