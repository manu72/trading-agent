"""
AI Trading Agent - Core Module

This module implements the AI Trading Agent component of the DeepResearch 4.5 Trading System.
It analyzes market data, generates trading signals, and optimizes execution.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

# Import from Knowledge Base
from ..knowledge_base import get_knowledge_base
from ..knowledge_base.models import TradingSignal

class AITradingAgent:
    """
    Main class for the AI Trading Agent component.
    
    The AI Trading Agent analyzes market data, generates trading signals,
    and optimizes execution based on various strategies.
    """
    
    def __init__(self):
        """Initialize the AI Trading Agent."""
        self.kb = get_knowledge_base()
        self.strategies = {}
        self.market_state = {}
        self.active_signals = []
        
        # Register strategies
        self._register_strategies()
    
    def _register_strategies(self):
        """Register trading strategies."""
        self.strategies = {
            'momentum': MomentumStrategy(),
            'sector_rotation': SectorRotationStrategy(),
            'swing_trading': SwingTradingStrategy(),
            'mean_reversion': MeanReversionStrategy()
        }
    
    def analyze_market(self, symbols: List[str], timeframe: str = 'daily', 
                      lookback_periods: int = 100) -> Dict[str, Any]:
        """
        Analyze market data and update internal state.
        
        Args:
            symbols: List of symbols to analyze
            timeframe: Data timeframe (e.g., 'daily', 'hourly')
            lookback_periods: Number of periods to look back
            
        Returns:
            Dictionary containing market analysis results
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(days=lookback_periods * 2)  # Extra buffer for calculations
        
        analysis_results = {}
        
        for symbol in symbols:
            # Get market data from Knowledge Base
            data = self.kb.get_market_data(symbol, start_time, end_time, timeframe)
            
            if data.empty:
                continue
            
            # Calculate indicators
            data = self._calculate_indicators(data)
            
            # Store in market state
            self.market_state[symbol] = data
            
            # Analyze with each strategy
            symbol_results = {}
            for strategy_name, strategy in self.strategies.items():
                strategy_result = strategy.analyze(data)
                symbol_results[strategy_name] = strategy_result
            
            analysis_results[symbol] = symbol_results
        
        return analysis_results
    
    def _calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for market data.
        
        Args:
            data: DataFrame containing market data
            
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
        
        return df
    
    def generate_signals(self) -> List[Dict[str, Any]]:
        """
        Generate trading signals based on current market analysis.
        
        Returns:
            List of trading signal dictionaries
        """
        signals = []
        
        for symbol, data in self.market_state.items():
            if data.empty or len(data) < 50:  # Need sufficient data
                continue
            
            # Get the latest data point
            latest = data.iloc[-1]
            
            # Generate signals from each strategy
            for strategy_name, strategy in self.strategies.items():
                strategy_signals = strategy.generate_signals(symbol, data)
                signals.extend(strategy_signals)
        
        # Store signals in Knowledge Base
        for signal in signals:
            signal_model = TradingSignal.from_dict(signal)
            signal_dict = signal_model.to_dict()
            self.kb.store_trading_signal(signal_dict)
            self.active_signals.append(signal_dict)
        
        return signals
    
    def get_active_signals(self) -> List[Dict[str, Any]]:
        """
        Retrieve currently active trading signals.
        
        Returns:
            List of active trading signal dictionaries
        """
        # Filter out expired signals
        current_time = datetime.now()
        active_signals = []
        
        for signal in self.active_signals:
            if 'expiration' in signal and signal['expiration']:
                expiration = datetime.fromisoformat(signal['expiration']) if isinstance(signal['expiration'], str) else signal['expiration']
                if expiration > current_time:
                    active_signals.append(signal)
            else:
                active_signals.append(signal)
        
        self.active_signals = active_signals
        return active_signals
    
    def update_model(self, performance_data: Dict[str, Any]) -> bool:
        """
        Update internal models based on performance feedback.
        
        Args:
            performance_data: Dictionary of performance metrics
            
        Returns:
            True if update was successful, False otherwise
        """
        # Update each strategy with performance data
        for strategy_name, strategy in self.strategies.items():
            if strategy_name in performance_data:
                strategy.update(performance_data[strategy_name])
        
        return True
    
    def get_market_view(self) -> Dict[str, Any]:
        """
        Retrieve current market view and analysis.
        
        Returns:
            Dictionary containing market view and analysis
        """
        market_view = {
            'timestamp': datetime.now().isoformat(),
            'symbols': {},
            'sector_analysis': {},
            'market_regime': self._detect_market_regime(),
            'risk_assessment': self._assess_risk()
        }
        
        # Add symbol-specific analysis
        for symbol, data in self.market_state.items():
            if data.empty:
                continue
            
            latest = data.iloc[-1]
            market_view['symbols'][symbol] = {
                'price': latest['close'],
                'trend': self._determine_trend(data),
                'volatility': latest['atr'] / latest['close'] * 100,
                'rsi': latest['rsi'],
                'volume_ratio': latest['volume_ratio'],
                'momentum': latest['momentum_14']
            }
        
        return market_view
    
    def _determine_trend(self, data: pd.DataFrame) -> str:
        """
        Determine the trend direction based on moving averages.
        
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
    
    def _detect_market_regime(self) -> str:
        """
        Detect the current market regime.
        
        Returns:
            Market regime ('trending_bullish', 'trending_bearish', 'range_bound', 'high_volatility')
        """
        # This is a simplified implementation
        # A more sophisticated approach would use machine learning for regime detection
        
        # Count trends across symbols
        bullish_count = 0
        bearish_count = 0
        neutral_count = 0
        
        for symbol, data in self.market_state.items():
            if data.empty:
                continue
            
            trend = self._determine_trend(data)
            if trend == 'bullish':
                bullish_count += 1
            elif trend == 'bearish':
                bearish_count += 1
            else:
                neutral_count += 1
        
        total_count = bullish_count + bearish_count + neutral_count
        if total_count == 0:
            return 'unknown'
        
        # Calculate average volatility
        volatility_sum = 0
        volatility_count = 0
        
        for symbol, data in self.market_state.items():
            if data.empty or len(data) < 14:
                continue
            
            latest = data.iloc[-1]
            volatility_sum += latest['atr'] / latest['close'] * 100
            volatility_count += 1
        
        avg_volatility = volatility_sum / volatility_count if volatility_count > 0 else 0
        
        # Determine regime
        if avg_volatility > 3.0:  # High volatility threshold
            return 'high_volatility'
        elif bullish_count / total_count > 0.6:
            return 'trending_bullish'
        elif bearish_count / total_count > 0.6:
            return 'trending_bearish'
        else:
            return 'range_bound'
    
    def _assess_risk(self) -> Dict[str, Any]:
        """
        Assess current market risk levels.
        
        Returns:
            Dictionary containing risk assessment
        """
        # This is a simplified implementation
        # A more sophisticated approach would incorporate more risk factors
        
        # Calculate average volatility
        volatility_sum = 0
        volatility_count = 0
        
        for symbol, data in self.market_state.items():
            if data.empty or len(data) < 14:
                continue
            
            latest = data.iloc[-1]
            volatility_sum += latest['atr'] / latest['close'] * 100
            volatility_count += 1
        
        avg_volatility = volatility_sum / volatility_count if volatility_count > 0 else 0
        
        # Determine risk level
        if avg_volatility < 1.0:
            risk_level = 'low'
        elif avg_volatility < 2.0:
            risk_level = 'moderate'
        elif avg_volatility < 3.0:
            risk_level = 'elevated'
        else:
            risk_level = 'high'
        
        return {
            'risk_level': risk_level,
            'avg_volatility': avg_volatility,
            'market_regime': self._detect_market_regime()
        }


class TradingStrategy:
    """Base class for trading strategies."""
    
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze market data for the strategy.
        
        Args:
            data: DataFrame containing market data with indicators
            
        Returns:
            Dictionary containing analysis results
        """
        raise NotImplementedError("Subclasses must implement analyze()")
    
    def generate_signals(self, symbol: str, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Generate trading signals based on market data.
        
        Args:
            symbol: Symbol to generate signals for
            data: DataFrame containing market data with indicators
            
        Returns:
            List of trading signal dictionaries
        """
        raise NotImplementedError("Subclasses must implement generate_signals()")
    
    def update(self, performance_data: Dict[str, Any]) -> bool:
        """
        Update strategy parameters based on performance feedback.
        
        Args:
            performance_data: Dictionary of performance metrics
            
        Returns:
            True if update was successful, False otherwise
        """
        # Default implementation does nothing
        return True


class MomentumStrategy(TradingStrategy):
    """Implementation of the momentum trading strategy."""
    
    def __init__(self):
        """Initialize the momentum strategy."""
        self.lookback_periods = 14
        self.momentum_threshold = 0.05  # 5% momentum
        self.rsi_overbought = 70
        self.rsi_oversold = 30
    
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze market data for momentum signals.
        
        Args:
            data: DataFrame containing market data with indicators
            
        Returns:
            Dictionary containing momentum analysis results
        """
        if len(data) < self.lookback_periods:
            return {'momentum_score': 0, 'signal': 'neutral'}
        
        latest = data.iloc[-1]
        
        # Calculate momentum score
        momentum = latest['momentum_14']
        rsi = latest['rsi']
        
        # Determine signal direction
        if momentum > self.momentum_threshold and rsi < self.rsi_overbought:
            signal = 'buy'
            strength = min(1.0, momentum / 0.2)  # Scale strength, max at 20% momentum
        elif momentum < -self.momentum_threshold and rsi > self.rsi_oversold:
            signal = 'sell'
            strength = min(1.0, abs(momentum) / 0.2)
        else:
            signal = 'neutral'
            strength = 0.0
        
        return {
            'momentum_score': momentum,
            'rsi': rsi,
            'signal': signal,
            'strength': strength
        }
    
    def generate_signals(self, symbol: str, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Generate momentum trading signals.
        
        Args:
            symbol: Symbol to generate signals for
            data: DataFrame containing market data with indicators
            
        Returns:
            List of trading signal dictionaries
        """
        signals = []
        
        if len(data) < self.lookback_periods:
            return signals
        
        analysis = self.analyze(data)
        
        if analysis['signal'] == 'buy' and analysis['strength'] > 0.3:
            signals.append({
                'symbol': symbol,
                'direction': 'BUY',
                'strategy': 'momentum',
                'strength': analysis['strength'],
                'timeframe': 'daily',
                'expiration': (datetime.now() + timedelta(days=5)).isoformat(),
                'metadata': {
                    'momentum_score': analysis['momentum_score'],
                    'rsi': analysis['rsi']
                }
            })
        elif analysis['signal'] == 'sell' and analysis['strength'] > 0.3:
            signals.append({
                'symbol': symbol,
                'direction': 'SELL',
                'strategy': 'momentum',
                'strength': analysis['strength'],
                'timeframe': 'daily',
                'expiration': (datetime.now() + timedelta(days=5)).isoformat(),
                'metadata': {
                    'momentum_score': analysis['momentum_score'],
                    'rsi': analysis['rsi']
                }
            })
        
        return signals
    
    def update(self, performance_data: Dict[str, Any]) -> bool:
        """
        Update momentum strategy parameters based on performance.
        
        Args:
            performance_data: Dictionary of performance metrics
            
        Returns:
            True if update was successful, False otherwise
        """
        # Adjust momentum threshold based on performance
        if 'win_rate' in performance_data:
            win_rate = performance_data['win_rate']
            
            # If win rate is low, increase threshold to be more selective
            if win_rate < 0.4:
                self.momentum_threshold = min(0.1, self.momentum_threshold + 0.01)
            # If win rate is high, decrease threshold to be more aggressive
            elif win_rate > 0.7:
                self.momentum_threshold = max(0.03, self.momentum_threshold - 0.01)
        
        return True


class SectorRotationStrategy(TradingStrategy):
    """Implementation of the sector rotation trading strategy."""
    
    def __init__(self):
        """Initialize the sector rotation strategy."""
        self.sector_lookback = 90  # 90 days for sector performance
        self.relative_strength_threshold = 0.05  # 5% outperformance
        self.sectors = {
            'XLK': 'Technology',
            'XLV': 'Healthcare',
            'XLF': 'Financials',
            'XLY': 'Consumer Discretionary',
            'XLP': 'Consumer Staples',
            'XLI': 'Industrials',
            'XLE': 'Energy',
            'XLB': 'Materials',
            'XLU': 'Utilities',
            'XLRE': 'Real Estate',
            'XLC': 'Communication Services'
        }
        self.sector_performance = {}
    
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze market data for sector rotation signals.
        
        Args:
            data: DataFrame containing market data with indicators
            
        Returns:
            Dictionary containing sector analysis results
        """
        if len(data) < self.sector_lookback:
            return {'relative_strength': 0, 'signal': 'neutral'}
        
        # Calculate sector performance
        start_price = data.iloc[-self.sector_lookback]['close']
        end_price = data.iloc[-1]['close']
        performance = (end_price / start_price - 1) * 100
        
        # Store in sector performance dictionary
        symbol = data.index.name if data.index.name else 'unknown'
        self.sector_performance[symbol] = performance
        
        # Calculate average sector performance
        avg_performance = sum(self.sector_performance.values()) / len(self.sector_performance) if self.sector_performance else 0
        
        # Calculate relative strength
        relative_strength = performance - avg_performance
        
        # Determine signal
        if relative_strength > self.relative_strength_threshold * 100:
            signal = 'buy'
            strength = min(1.0, relative_strength / 10)  # Scale strength, max at 10% outperformance
        elif relative_strength < -self.relative_strength_threshold * 100:
            signal = 'sell'
            strength = min(1.0, abs(relative_strength) / 10)
        else:
            signal = 'neutral'
            strength = 0.0
        
        return {
            'performance': performance,
            'relative_strength': relative_strength,
            'signal': signal,
            'strength': strength
        }
    
    def generate_signals(self, symbol: str, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Generate sector rotation trading signals.
        
        Args:
            symbol: Symbol to generate signals for
            data: DataFrame containing market data with indicators
            
        Returns:
            List of trading signal dictionaries
        """
        signals = []
        
        if len(data) < self.sector_lookback:
            return signals
        
        # Only generate signals for sector ETFs
        if symbol not in self.sectors:
            return signals
        
        analysis = self.analyze(data)
        
        if analysis['signal'] == 'buy' and analysis['strength'] > 0.3:
            signals.append({
                'symbol': symbol,
                'direction': 'BUY',
                'strategy': 'sector_rotation',
                'strength': analysis['strength'],
                'timeframe': 'daily',
                'expiration': (datetime.now() + timedelta(days=30)).isoformat(),
                'metadata': {
                    'sector': self.sectors[symbol],
                    'performance': analysis['performance'],
                    'relative_strength': analysis['relative_strength']
                }
            })
        elif analysis['signal'] == 'sell' and analysis['strength'] > 0.3:
            signals.append({
                'symbol': symbol,
                'direction': 'SELL',
                'strategy': 'sector_rotation',
                'strength': analysis['strength'],
                'timeframe': 'daily',
                'expiration': (datetime.now() + timedelta(days=30)).isoformat(),
                'metadata': {
                    'sector': self.sectors[symbol],
                    'performance': analysis['performance'],
                    'relative_strength': analysis['relative_strength']
                }
            })
        
        return signals


class SwingTradingStrategy(TradingStrategy):
    """Implementation of the swing trading strategy."""
    
    def __init__(self):
        """Initialize the swing trading strategy."""
        self.rsi_overbought = 70
        self.rsi_oversold = 30
        self.bb_threshold = 0.05  # 5% from Bollinger Band
    
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze market data for swing trading signals.
        
        Args:
            data: DataFrame containing market data with indicators
            
        Returns:
            Dictionary containing swing trading analysis results
        """
        if len(data) < 20:  # Need at least 20 periods for Bollinger Bands
            return {'signal': 'neutral', 'strength': 0.0}
        
        latest = data.iloc[-1]
        
        # Check for oversold conditions (potential buy)
        if latest['rsi'] < self.rsi_oversold and latest['close'] < latest['bb_lower']:
            bb_distance = (latest['bb_lower'] - latest['close']) / latest['close']
            signal = 'buy'
            strength = min(1.0, bb_distance / self.bb_threshold)
        
        # Check for overbought conditions (potential sell)
        elif latest['rsi'] > self.rsi_overbought and latest['close'] > latest['bb_upper']:
            bb_distance = (latest['close'] - latest['bb_upper']) / latest['close']
            signal = 'sell'
            strength = min(1.0, bb_distance / self.bb_threshold)
        
        else:
            signal = 'neutral'
            strength = 0.0
        
        return {
            'rsi': latest['rsi'],
            'bb_upper': latest['bb_upper'],
            'bb_lower': latest['bb_lower'],
            'signal': signal,
            'strength': strength
        }
    
    def generate_signals(self, symbol: str, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Generate swing trading signals.
        
        Args:
            symbol: Symbol to generate signals for
            data: DataFrame containing market data with indicators
            
        Returns:
            List of trading signal dictionaries
        """
        signals = []
        
        if len(data) < 20:
            return signals
        
        analysis = self.analyze(data)
        
        if analysis['signal'] == 'buy' and analysis['strength'] > 0.5:
            signals.append({
                'symbol': symbol,
                'direction': 'BUY',
                'strategy': 'swing_trading',
                'strength': analysis['strength'],
                'timeframe': 'daily',
                'expiration': (datetime.now() + timedelta(days=3)).isoformat(),
                'metadata': {
                    'rsi': analysis['rsi'],
                    'bb_lower': analysis['bb_lower']
                }
            })
        elif analysis['signal'] == 'sell' and analysis['strength'] > 0.5:
            signals.append({
                'symbol': symbol,
                'direction': 'SELL',
                'strategy': 'swing_trading',
                'strength': analysis['strength'],
                'timeframe': 'daily',
                'expiration': (datetime.now() + timedelta(days=3)).isoformat(),
                'metadata': {
                    'rsi': analysis['rsi'],
                    'bb_upper': analysis['bb_upper']
                }
            })
        
        return signals


class MeanReversionStrategy(TradingStrategy):
    """Implementation of the mean reversion trading strategy."""
    
    def __init__(self):
        """Initialize the mean reversion strategy."""
        self.z_score_threshold = 2.0
        self.lookback = 20
    
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze market data for mean reversion signals.
        
        Args:
            data: DataFrame containing market data with indicators
            
        Returns:
            Dictionary containing mean reversion analysis results
        """
        if len(data) < self.lookback:
            return {'signal': 'neutral', 'strength': 0.0}
        
        # Calculate Z-score
        price_series = data['close'].iloc[-self.lookback:]
        mean = price_series.mean()
        std = price_series.std()
        latest_price = price_series.iloc[-1]
        
        if std == 0:
            z_score = 0
        else:
            z_score = (latest_price - mean) / std
        
        # Determine signal
        if z_score > self.z_score_threshold:
            signal = 'sell'  # Price is too high, expect reversion down
            strength = min(1.0, (z_score - self.z_score_threshold) / 2)
        elif z_score < -self.z_score_threshold:
            signal = 'buy'  # Price is too low, expect reversion up
            strength = min(1.0, (-z_score - self.z_score_threshold) / 2)
        else:
            signal = 'neutral'
            strength = 0.0
        
        return {
            'z_score': z_score,
            'mean': mean,
            'std': std,
            'signal': signal,
            'strength': strength
        }
    
    def generate_signals(self, symbol: str, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Generate mean reversion trading signals.
        
        Args:
            symbol: Symbol to generate signals for
            data: DataFrame containing market data with indicators
            
        Returns:
            List of trading signal dictionaries
        """
        signals = []
        
        if len(data) < self.lookback:
            return signals
        
        analysis = self.analyze(data)
        
        if analysis['signal'] == 'buy' and analysis['strength'] > 0.3:
            signals.append({
                'symbol': symbol,
                'direction': 'BUY',
                'strategy': 'mean_reversion',
                'strength': analysis['strength'],
                'timeframe': 'daily',
                'expiration': (datetime.now() + timedelta(days=5)).isoformat(),
                'metadata': {
                    'z_score': analysis['z_score'],
                    'mean': analysis['mean'],
                    'std': analysis['std']
                }
            })
        elif analysis['signal'] == 'sell' and analysis['strength'] > 0.3:
            signals.append({
                'symbol': symbol,
                'direction': 'SELL',
                'strategy': 'mean_reversion',
                'strength': analysis['strength'],
                'timeframe': 'daily',
                'expiration': (datetime.now() + timedelta(days=5)).isoformat(),
                'metadata': {
                    'z_score': analysis['z_score'],
                    'mean': analysis['mean'],
                    'std': analysis['std']
                }
            })
        
        return signals
