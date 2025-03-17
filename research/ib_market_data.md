# Interactive Brokers Market Data Retrieval Research

## Market Data Subscription Requirements
- Funded IBKR account with at least $500 USD in most instances
- This threshold must be maintained in addition to the cost of any subscriptions
- Market data subscriptions are on a per-user basis
- API functionality is considered "off-platform" by exchanges and typically has costs associated
- Subscriptions can be managed through the Client Portal

## Types of Market Data Available
1. **Streaming Market Data** - Real-time price and size data (TWS Watchlist data)
2. **Delayed Data** - 15-minute delayed data (for clients without market data subscriptions)
3. **Historical Data** - Past price and volume data
4. **Other Data Types**:
   - Tick data
   - Histogram data
   - Market depth data

## Market Data Type Options
- Type 1: Live data
- Type 2: Frozen data (market data from most recent close)
- Type 3: Delayed data (15-minute delay)
- Type 4: Delayed frozen data (yesterday's closing values)

## Requesting Streaming Market Data
- Use `EClient.reqMktData` method
- Key parameters:
  - reqId: Unique identifier for the request
  - contract: Contract object defining the instrument
  - genericTickList: String of comma-separated values for additional data (e.g., "232" for mark price)
  - snapshot: Boolean for single return instance (aggregates last 11 seconds of trading)
  - regulatorySnapshot: For determining price before executing a trade (costs ~$0.01 per request)
  - mktDataOptions: For internal purposes

## Handling Incoming Market Data
- Implement `tickPrice` function to handle price-related data
  - Parameters: reqId, tickType, price, attrib
  - tickType is an integer value correlating to specific data (bid price, last size, etc.)
  - Can use `TickTypeEnum.toStr()` to convert tick type integers to readable values

- Implement `tickSize` function to handle quantity-related data
  - Parameters: reqId, tickType, size
  - Sizes returned relate to prices from tickPrice function

## Requesting Historical Data
- Similar pattern to live market data requests
- Requires valid market data subscription
- Can use `headTimeStamp` function to determine how far back data is available

## Implementation Considerations
1. Need to implement both EClient (outgoing calls) and EWrapper (incoming data) functions
2. Threading is required for maintaining open connections
3. Interactive Python environments (Jupyter, Spyder) may have issues with EWrapper functions
4. IB-insync is a third-party library that uses asyncio for asynchronous single thread interaction

## Code Structure Requirements
- Import necessary modules: `from ibapi.ticktype import TickTypeEnum`
- Create contract objects with symbol, security type, currency, and exchange
- Implement callback functions for different data types
- Handle threading for maintaining connections

## Next Research Steps
1. Detailed examination of historical data retrieval parameters
2. Research on account and portfolio data retrieval
3. Investigation of market scanners for sector rotation strategy
4. Study of error handling for market data requests
