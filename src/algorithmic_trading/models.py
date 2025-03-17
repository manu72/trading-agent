"""
Algorithmic Trading System - Models

This module defines the data models used by the Algorithmic Trading System component.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from enum import Enum

class OrderStatus(str, Enum):
    """Enum for order status values."""
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    FAILED = "FAILED"
    UNKNOWN = "UNKNOWN"

class OrderType(str, Enum):
    """Enum for order type values."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"
    TRAIL = "TRAIL"
    TRAIL_LIMIT = "TRAIL_LIMIT"
    MOC = "MOC"  # Market on Close
    LOC = "LOC"  # Limit on Close
    ALGO = "ALGO"  # Algorithmic order

class OrderDirection(str, Enum):
    """Enum for order direction values."""
    BUY = "BUY"
    SELL = "SELL"

class TimeInForce(str, Enum):
    """Enum for time in force values."""
    DAY = "DAY"
    GTC = "GTC"  # Good Till Cancelled
    IOC = "IOC"  # Immediate or Cancel
    FOK = "FOK"  # Fill or Kill
    GTD = "GTD"  # Good Till Date

@dataclass
class Order:
    """Model for orders."""
    order_id: str
    symbol: str
    direction: OrderDirection
    quantity: float
    order_type: OrderType
    status: OrderStatus = OrderStatus.PENDING
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.DAY
    execution_algorithm: str = "market"
    strategy: str = ""
    signal_id: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    execution_details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'direction': self.direction if isinstance(self.direction, str) else self.direction.value,
            'quantity': self.quantity,
            'order_type': self.order_type if isinstance(self.order_type, str) else self.order_type.value,
            'status': self.status if isinstance(self.status, str) else self.status.value,
            'price': self.price,
            'stop_price': self.stop_price,
            'time_in_force': self.time_in_force if isinstance(self.time_in_force, str) else self.time_in_force.value,
            'execution_algorithm': self.execution_algorithm,
            'strategy': self.strategy,
            'signal_id': self.signal_id,
            'timestamp': self.timestamp,
            'execution_details': self.execution_details
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Order':
        """Create from dictionary representation."""
        # Convert string enums to enum values
        direction = OrderDirection(data['direction']) if isinstance(data['direction'], str) else data['direction']
        order_type = OrderType(data['order_type']) if isinstance(data['order_type'], str) else data['order_type']
        status = OrderStatus(data['status']) if isinstance(data['status'], str) else data['status']
        time_in_force = TimeInForce(data['time_in_force']) if isinstance(data['time_in_force'], str) and 'time_in_force' in data else TimeInForce.DAY
        
        return cls(
            order_id=data['order_id'],
            symbol=data['symbol'],
            direction=direction,
            quantity=data['quantity'],
            order_type=order_type,
            status=status,
            price=data.get('price'),
            stop_price=data.get('stop_price'),
            time_in_force=time_in_force,
            execution_algorithm=data.get('execution_algorithm', 'market'),
            strategy=data.get('strategy', ''),
            signal_id=data.get('signal_id', ''),
            timestamp=data.get('timestamp', datetime.now().isoformat()),
            execution_details=data.get('execution_details', {})
        )

@dataclass
class Position:
    """Model for positions."""
    symbol: str
    quantity: float
    average_cost: float
    market_value: float
    unrealized_pnl: float
    sector: str = "Unknown"
    currency: str = "USD"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'symbol': self.symbol,
            'quantity': self.quantity,
            'average_cost': self.average_cost,
            'market_value': self.market_value,
            'unrealized_pnl': self.unrealized_pnl,
            'sector': self.sector,
            'currency': self.currency,
            'timestamp': self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Position':
        """Create from dictionary representation."""
        return cls(
            symbol=data['symbol'],
            quantity=data['quantity'],
            average_cost=data['average_cost'],
            market_value=data['market_value'],
            unrealized_pnl=data['unrealized_pnl'],
            sector=data.get('sector', 'Unknown'),
            currency=data.get('currency', 'USD'),
            timestamp=data.get('timestamp', datetime.now().isoformat())
        )

@dataclass
class AccountInfo:
    """Model for account information."""
    account_id: str
    net_liquidation_value: float
    equity_with_loan_value: float
    buying_power: float
    cash_balance: float
    day_trades_remaining: int
    leverage: float
    currency: str = "USD"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'account_id': self.account_id,
            'net_liquidation_value': self.net_liquidation_value,
            'equity_with_loan_value': self.equity_with_loan_value,
            'buying_power': self.buying_power,
            'cash_balance': self.cash_balance,
            'day_trades_remaining': self.day_trades_remaining,
            'leverage': self.leverage,
            'currency': self.currency,
            'timestamp': self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AccountInfo':
        """Create from dictionary representation."""
        return cls(
            account_id=data['account_id'],
            net_liquidation_value=data['net_liquidation_value'],
            equity_with_loan_value=data['equity_with_loan_value'],
            buying_power=data['buying_power'],
            cash_balance=data['cash_balance'],
            day_trades_remaining=data['day_trades_remaining'],
            leverage=data['leverage'],
            currency=data.get('currency', 'USD'),
            timestamp=data.get('timestamp', datetime.now().isoformat())
        )

@dataclass
class ExecutionResult:
    """Model for execution results."""
    success: bool
    order_id: str
    execution_time: str
    average_price: Optional[float] = None
    filled_quantity: Optional[float] = None
    commission: Optional[float] = None
    error: Optional[str] = None
    paper_trading: bool = False
    execution_details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'success': self.success,
            'order_id': self.order_id,
            'execution_time': self.execution_time,
            'average_price': self.average_price,
            'filled_quantity': self.filled_quantity,
            'commission': self.commission,
            'error': self.error,
            'paper_trading': self.paper_trading,
            'execution_details': self.execution_details
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExecutionResult':
        """Create from dictionary representation."""
        return cls(
            success=data['success'],
            order_id=data['order_id'],
            execution_time=data['execution_time'],
            average_price=data.get('average_price'),
            filled_quantity=data.get('filled_quantity'),
            commission=data.get('commission'),
            error=data.get('error'),
            paper_trading=data.get('paper_trading', False),
            execution_details=data.get('execution_details', {})
        )

@dataclass
class PortfolioStatus:
    """Model for portfolio status."""
    timestamp: str
    portfolio_value: float
    buying_power: float
    cash_balance: float
    positions_count: int
    sector_exposure: Dict[str, float]
    gross_exposure: float
    net_exposure: float
    positions: Dict[str, Dict[str, Any]]
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'timestamp': self.timestamp,
            'portfolio_value': self.portfolio_value,
            'buying_power': self.buying_power,
            'cash_balance': self.cash_balance,
            'positions_count': self.positions_count,
            'sector_exposure': self.sector_exposure,
            'gross_exposure': self.gross_exposure,
            'net_exposure': self.net_exposure,
            'positions': self.positions,
            'error': self.error
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PortfolioStatus':
        """Create from dictionary representation."""
        return cls(
            timestamp=data['timestamp'],
            portfolio_value=data['portfolio_value'],
            buying_power=data['buying_power'],
            cash_balance=data['cash_balance'],
            positions_count=data['positions_count'],
            sector_exposure=data['sector_exposure'],
            gross_exposure=data['gross_exposure'],
            net_exposure=data['net_exposure'],
            positions=data['positions'],
            error=data.get('error')
        )

@dataclass
class RiskControlSettings:
    """Model for risk control settings."""
    max_position_size_pct: float
    max_daily_drawdown_pct: float
    max_sector_exposure_pct: float
    max_leverage: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'max_position_size_pct': self.max_position_size_pct,
            'max_daily_drawdown_pct': self.max_daily_drawdown_pct,
            'max_sector_exposure_pct': self.max_sector_exposure_pct,
            'max_leverage': self.max_leverage
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RiskControlSettings':
        """Create from dictionary representation."""
        return cls(
            max_position_size_pct=data['max_position_size_pct'],
            max_daily_drawdown_pct=data['max_daily_drawdown_pct'],
            max_sector_exposure_pct=data['max_sector_exposure_pct'],
            max_leverage=data['max_leverage']
        )
