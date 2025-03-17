"""
Trading Knowledge Base - Data Models

This module defines the data models used by the Knowledge Base component.
These models provide structure and validation for the data stored in the system.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

@dataclass
class MarketData:
    """Model for market data."""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    indicators: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'indicators': self.indicators
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MarketData':
        """Create from dictionary representation."""
        return cls(
            symbol=data['symbol'],
            timestamp=datetime.fromisoformat(data['timestamp']) if isinstance(data['timestamp'], str) else data['timestamp'],
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close'],
            volume=data['volume'],
            indicators=data.get('indicators', {})
        )


@dataclass
class TradingSignal:
    """Model for trading signals."""
    symbol: str
    direction: str  # 'BUY' or 'SELL'
    strategy: str
    strength: float
    timeframe: str
    signal_id: Optional[str] = None
    timestamp: Optional[datetime] = None
    expiration: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'signal_id': self.signal_id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else datetime.now().isoformat(),
            'symbol': self.symbol,
            'direction': self.direction,
            'strategy': self.strategy,
            'strength': self.strength,
            'timeframe': self.timeframe,
            'expiration': self.expiration.isoformat() if self.expiration else None,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TradingSignal':
        """Create from dictionary representation."""
        return cls(
            signal_id=data.get('signal_id'),
            timestamp=datetime.fromisoformat(data['timestamp']) if isinstance(data.get('timestamp'), str) else data.get('timestamp'),
            symbol=data['symbol'],
            direction=data['direction'],
            strategy=data['strategy'],
            strength=data['strength'],
            timeframe=data['timeframe'],
            expiration=datetime.fromisoformat(data['expiration']) if isinstance(data.get('expiration'), str) and data.get('expiration') else data.get('expiration'),
            metadata=data.get('metadata', {})
        )


@dataclass
class Order:
    """Model for orders."""
    symbol: str
    order_type: str  # 'MARKET', 'LIMIT', 'STOP', etc.
    direction: str  # 'BUY' or 'SELL'
    quantity: float
    price: Optional[float] = None
    order_id: Optional[str] = None
    signal_id: Optional[str] = None
    status: str = 'PENDING'  # 'PENDING', 'FILLED', 'CANCELLED', etc.
    timestamp: Optional[datetime] = None
    execution_details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'order_id': self.order_id,
            'signal_id': self.signal_id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else datetime.now().isoformat(),
            'symbol': self.symbol,
            'order_type': self.order_type,
            'direction': self.direction,
            'quantity': self.quantity,
            'price': self.price,
            'status': self.status,
            'execution_details': self.execution_details
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Order':
        """Create from dictionary representation."""
        return cls(
            order_id=data.get('order_id'),
            signal_id=data.get('signal_id'),
            timestamp=datetime.fromisoformat(data['timestamp']) if isinstance(data.get('timestamp'), str) else data.get('timestamp'),
            symbol=data['symbol'],
            order_type=data['order_type'],
            direction=data['direction'],
            quantity=data['quantity'],
            price=data.get('price'),
            status=data.get('status', 'PENDING'),
            execution_details=data.get('execution_details', {})
        )


@dataclass
class PerformanceMetrics:
    """Model for performance metrics."""
    strategy: str
    timestamp: datetime
    return_pct: float
    win_rate: float
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    profit_factor: Optional[float] = None
    trades_count: Optional[int] = None
    additional_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'strategy': self.strategy,
            'timestamp': self.timestamp.isoformat(),
            'return_pct': self.return_pct,
            'win_rate': self.win_rate,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'profit_factor': self.profit_factor,
            'trades_count': self.trades_count,
            **self.additional_metrics
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PerformanceMetrics':
        """Create from dictionary representation."""
        # Extract known fields
        known_fields = {
            'strategy': data['strategy'],
            'timestamp': datetime.fromisoformat(data['timestamp']) if isinstance(data['timestamp'], str) else data['timestamp'],
            'return_pct': data['return_pct'],
            'win_rate': data['win_rate'],
            'sharpe_ratio': data.get('sharpe_ratio'),
            'max_drawdown': data.get('max_drawdown'),
            'profit_factor': data.get('profit_factor'),
            'trades_count': data.get('trades_count')
        }
        
        # Extract additional metrics
        additional_metrics = {k: v for k, v in data.items() if k not in known_fields}
        
        return cls(
            **known_fields,
            additional_metrics=additional_metrics
        )
