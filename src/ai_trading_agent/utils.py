"""
AI Trading Agent - Models

This module defines the data models used by the AI Trading Agent component.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

@dataclass
class MarketAnalysis:
    """Model for market analysis results."""
    symbol: str
    timestamp: datetime
    trend: str  # 'bullish', 'bearish', or 'neutral'
    price: float
    volume: int
    indicators: Dict[str, float] = field(default_factory=dict)
    strategy_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'trend': self.trend,
            'price': self.price,
            'volume': self.volume,
            'indicators': self.indicators,
            'strategy_results': self.strategy_results
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MarketAnalysis':
        """Create from dictionary representation."""
        return cls(
            symbol=data['symbol'],
            timestamp=datetime.fromisoformat(data['timestamp']) if isinstance(data['timestamp'], str) else data['timestamp'],
            trend=data['trend'],
            price=data['price'],
            volume=data['volume'],
            indicators=data.get('indicators', {}),
            strategy_results=data.get('strategy_results', {})
        )


@dataclass
class SignalStrength:
    """Model for signal strength evaluation."""
    value: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    factors: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'value': self.value,
            'confidence': self.confidence,
            'factors': self.factors
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SignalStrength':
        """Create from dictionary representation."""
        return cls(
            value=data['value'],
            confidence=data['confidence'],
            factors=data.get('factors', {})
        )


@dataclass
class MarketRegime:
    """Model for market regime classification."""
    regime_type: str  # 'trending_bullish', 'trending_bearish', 'range_bound', 'high_volatility'
    confidence: float  # 0.0 to 1.0
    volatility: float
    trend_strength: float
    correlation: float
    additional_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'regime_type': self.regime_type,
            'confidence': self.confidence,
            'volatility': self.volatility,
            'trend_strength': self.trend_strength,
            'correlation': self.correlation,
            **self.additional_metrics
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MarketRegime':
        """Create from dictionary representation."""
        # Extract known fields
        known_fields = {
            'regime_type': data['regime_type'],
            'confidence': data['confidence'],
            'volatility': data['volatility'],
            'trend_strength': data['trend_strength'],
            'correlation': data['correlation']
        }
        
        # Extract additional metrics
        additional_metrics = {k: v for k, v in data.items() if k not in known_fields}
        
        return cls(
            **known_fields,
            additional_metrics=additional_metrics
        )


@dataclass
class RiskAssessment:
    """Model for risk assessment."""
    risk_level: str  # 'low', 'moderate', 'elevated', 'high'
    volatility: float
    correlation: float
    liquidity: float
    market_regime: str
    risk_factors: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'risk_level': self.risk_level,
            'volatility': self.volatility,
            'correlation': self.correlation,
            'liquidity': self.liquidity,
            'market_regime': self.market_regime,
            'risk_factors': self.risk_factors
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RiskAssessment':
        """Create from dictionary representation."""
        return cls(
            risk_level=data['risk_level'],
            volatility=data['volatility'],
            correlation=data['correlation'],
            liquidity=data['liquidity'],
            market_regime=data['market_regime'],
            risk_factors=data.get('risk_factors', {})
        )


@dataclass
class StrategyPerformance:
    """Model for strategy performance metrics."""
    strategy_name: str
    timestamp: datetime
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    return_pct: float
    trades_count: int
    additional_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'strategy_name': self.strategy_name,
            'timestamp': self.timestamp.isoformat(),
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'return_pct': self.return_pct,
            'trades_count': self.trades_count,
            **self.additional_metrics
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StrategyPerformance':
        """Create from dictionary representation."""
        # Extract known fields
        known_fields = {
            'strategy_name': data['strategy_name'],
            'timestamp': datetime.fromisoformat(data['timestamp']) if isinstance(data['timestamp'], str) else data['timestamp'],
            'win_rate': data['win_rate'],
            'profit_factor': data['profit_factor'],
            'sharpe_ratio': data['sharpe_ratio'],
            'max_drawdown': data['max_drawdown'],
            'return_pct': data['return_pct'],
            'trades_count': data['trades_count']
        }
        
        # Extract additional metrics
        additional_metrics = {k: v for k, v in data.items() if k not in known_fields}
        
        return cls(
            **known_fields,
            additional_metrics=additional_metrics
        )
"""

AI Trading Agent - Utilities

This module provides utility functions for the AI Trading Agent component.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta

def calculate_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate technical indicators for market data.
    
    Args:
        data: DataFrame containing market data (OHLCV)
        
    Returns:
        DataFrame with additional indicator columns
    """
    # Make a copy to avoid modifying the original
    df = data.copy()
    
    # Simple Moving Averages
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['sma_200'] = df['close'].rolling(window=200).mean()
    
    # Exponential Moving Averages
    df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
    
    # MACD
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # Relative Strength Index (RSI)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    df['bb_std'] = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
    
    # Average True Range (ATR)
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = true_range.rolling(window=14).mean()
    
    # Volume indicators
    df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma_20']
    
    # Momentum
    df['momentum_14'] = df['close'] / df['close'].shift(14) - 1
    
    # Rate of Change
    df['roc_10'] = (df['close'] / df['close'].shift(10) - 1) * 100
    
    # Stochastic Oscillator
    low_14 = df['low'].rolling(window=14).min()
    high_14 = df['high'].rolling(window=14).max()
    df['stoch_k'] = 100 * ((df['close'] - low_14) / (high_14 - low_14))
    df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
    
    # On-Balance Volume (OBV)
    obv = pd.Series(index=df.index)
    obv.iloc[0] = 0
    
    for i in range(1, len(df)):
        if df['close'].iloc[i] > df['close'].iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] + df['volume'].iloc[i]
        elif df['close'].iloc[i] < df['close'].iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] - df['volume'].iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i-1]
    
    df['obv'] = obv
    
    return df

def detect_trend(data: pd.DataFrame) -> str:
    """
    Detect the trend direction in market data.
    
    Args:
        data: DataFrame containing market data with indicators
        
    Returns:
        Trend direction ('bullish', 'bearish', or 'neutral')
    """
    if len(data) < 50:
        return 'neutral'
    
    latest = data.iloc[-1]
    
    # Check moving average relationships
    if latest['sma_20'] > latest['sma_50'] > latest['sma_200'] and latest['close'] > latest['sma_20']:
        return 'bullish'
    elif latest['sma_20'] < latest['sma_50'] < latest['sma_200'] and latest['close'] < latest['sma_20']:
        return 'bearish'
    else:
        return 'neutral'

def calculate_volatility(data: pd.DataFrame, window: int = 20) -> float:
    """
    Calculate the volatility of market data.
    
    Args:
        data: DataFrame containing market data
        window: Window size for volatility calculation
        
    Returns:
        Volatility as a percentage
    """
    if len(data) < window:
        return 0.0
    
    # Calculate daily returns
    returns = data['close'].pct_change().dropna()
    
    # Calculate volatility (standard deviation of returns)
    volatility = returns.rolling(window=window).std().iloc[-1]
    
    # Annualize volatility (assuming daily data)
    annualized_volatility = volatility * np.sqrt(252)
    
    return annualized_volatility * 100

def calculate_z_score(data: pd.Series, window: int = 20) -> float:
    """
    Calculate the Z-score of the latest value in a series.
    
    Args:
        data: Series of values
        window: Window size for Z-score calculation
        
    Returns:
        Z-score of the latest value
    """
    if len(data) < window:
        return 0.0
    
    # Get the window of data
    window_data = data.iloc[-window:]
    
    # Calculate mean and standard deviation
    mean = window_data.mean()
    std = window_data.std()
    
    # Calculate Z-score
    if std == 0:
        return 0.0
    else:
        return (window_data.iloc[-1] - mean) / std

def calculate_relative_strength(data: pd.DataFrame, benchmark_data: pd.DataFrame, window: int = 90) -> float:
    """
    Calculate the relative strength of a security compared to a benchmark.
    
    Args:
        data: DataFrame containing market data for the security
        benchmark_data: DataFrame containing market data for the benchmark
        window: Window size for relative strength calculation
        
    Returns:
        Relative strength as a percentage
    """
    if len(data) < window or len(benchmark_data) < window:
        return 0.0
    
    # Calculate performance over the window
    security_start = data['close'].iloc[-window]
    security_end = data['close'].iloc[-1]
    security_performance = (security_end / security_start - 1) * 100
    
    benchmark_start = benchmark_data['close'].iloc[-window]
    benchmark_end = benchmark_data['close'].iloc[-1]
    benchmark_performance = (benchmark_end / benchmark_start - 1) * 100
    
    # Calculate relative strength
    relative_strength = security_performance - benchmark_performance
    
    return relative_strength

def detect_support_resistance(data: pd.DataFrame, window: int = 20, threshold: float = 0.03) -> Dict[str, List[float]]:
    """
    Detect support and resistance levels in market data.
    
    Args:
        data: DataFrame containing market data
        window: Window size for peak detection
        threshold: Minimum distance between levels (as percentage)
        
    Returns:
        Dictionary with 'support' and 'resistance' lists
    """
    if len(data) < window * 2:
        return {'support': [], 'resistance': []}
    
    # Find local minima (support) and maxima (resistance)
    support = []
    resistance = []
    
    for i in range(window, len(data) - window):
        # Check if this point is a local minimum (support)
        if all(data['low'].iloc[i] <= data['low'].iloc[i-j] for j in range(1, window+1)) and \
           all(data['low'].iloc[i] <= data['low'].iloc[i+j] for j in range(1, window+1)):
            support.append(data['low'].iloc[i])
        
        # Check if this point is a local maximum (resistance)
        if all(data['high'].iloc[i] >= data['high'].iloc[i-j] for j in range(1, window+1)) and \
           all(data['high'].iloc[i] >= data['high'].iloc[i+j] for j in range(1, window+1)):
            resistance.append(data['high'].iloc[i])
    
    # Filter out levels that are too close to each other
    filtered_support = []
    filtered_resistance = []
    
    # Sort levels
    support.sort()
    resistance.sort()
    
    # Filter support levels
    for level in support:
        if not filtered_support or all(abs(level / s - 1) > threshold for s in filtered_support):
            filtered_support.append(level)
    
    # Filter resistance levels
    for level in resistance:
        if not filtered_resistance or all(abs(level / r - 1) > threshold for r in filtered_resistance):
            filtered_resistance.append(level)
    
    return {'support': filtered_support, 'resistance': filtered_resistance}

def calculate_risk_reward_ratio(entry_price: float, target_price: float, stop_loss: float) -> float:
    """
    Calculate the risk-reward ratio for a trade.
    
    Args:
        entry_price: Entry price
        target_price: Target price
        stop_loss: Stop-loss price
        
    Returns:
        Risk-reward ratio
    """
    # Calculate potential reward
    if entry_price < target_price:  # Long position
        reward = target_price - entry_price
        risk = entry_price - stop_loss
    else:  # Short position
        reward = entry_price - target_price
        risk = stop_loss - entry_price
    
    # Calculate ratio
    if risk <= 0:
        return 0.0  # Invalid stop-loss
    
    return reward / risk

def calculate_position_size(account_value: float, risk_per_trade: float, entry_price: float, stop_loss: float) -> float:
    """
    Calculate the position size based on risk parameters.
    
    Args:
        account_value: Total account value
        risk_per_trade: Percentage of account to risk per trade (0-1)
        entry_price: Entry price
        stop_loss: Stop-loss price
        
    Returns:
        Number of shares/contracts to trade
    """
    # Calculate dollar risk
    dollar_risk = account_value * risk_per_trade
    
    # Calculate risk per share
    if entry_price < stop_loss:  # Short position
        risk_per_share = stop_loss - entry_price
    else:  # Long position
        risk_per_share = entry_price - stop_loss
    
    # Calculate position size
    if risk_per_share <= 0:
        return 0.0  # Invalid stop-loss
    
    position_size = dollar_risk / risk_per_share
    
    return position_size
