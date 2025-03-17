# Interactive Brokers Order Types Research

## Introduction
This document contains research on the order types available through the Interactive Brokers API interfaces. The values documented showcase the minimum values needed to create orders, with additional parameters that can be used or combined in some cases.

## Available Interfaces for Order Information
- TWS API's Order Object
- CPAPI's /iserver/account/{{accountId}}/orders endpoint

## Important Notes & Limitations
1. IB support team can help with API parameters but cannot provide financial or trading advice
2. Test order types manually in TWS before implementing in API
3. Some execution ideas may not be supported in TWS and therefore not supported by API
4. Paper Trading Accounts use a simulated environment - order execution behavior is not indicative of real-world trading conditions
5. IB APIs do not support fractional or cash quantity trading (except for Cryptocurrencies and Forex)

## Order Types Categories
1. Basic Orders
   - Market Orders
   - Limit Orders
   - Stop Orders
   - Stop-Limit Orders
   - Trailing Stop Orders
   - Market-to-Limit Orders
   - Limit-if-Touched Orders
   - Market-with-Protection Orders

2. Auction Orders
   - Market-on-Close
   - Limit-on-Close
   - Market-on-Open
   - Limit-on-Open
   - Auction Orders

3. Complex Orders
   - Relative Orders
   - Pegged-to-Market Orders
   - Volatility Orders
   - VWAP Orders
   - TWAP Orders
   - Arrival Price Orders

4. Order Conditioning
   - Time Conditions
   - Price Conditions
   - Volume Conditions
   - Margin Conditions
   - Percentage Conditions

5. IBKRATS Order Types
   - Special order types specific to Interactive Brokers

6. Algorithmic Orders
   - Adaptive Algorithms
   - Arrival Price
   - Dark Ice
   - TWAP
   - VWAP

## Implementation Considerations
1. Each order type requires specific parameters
2. Order types have different behaviors across different asset classes
3. Some order types are only available for specific exchanges or products
4. Testing in paper trading environment is essential before live implementation
5. Error handling is critical for order management

## Next Research Steps
1. Detailed examination of order parameters for momentum and sector rotation strategies
2. Research on stop-loss and take-profit implementation
3. Investigation of position sizing and risk management parameters
4. Study of order status monitoring and modification capabilities
