"""
Trading Knowledge Base - Utilities

This module provides utility functions for the Knowledge Base component.
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from pathlib import Path

def convert_timeframe_to_timedelta(timeframe: str) -> timedelta:
    """
    Convert a timeframe string to a timedelta object.
    
    Args:
        timeframe: String representation of timeframe (e.g., 'daily', 'hourly')
        
    Returns:
        timedelta object representing the timeframe
    """
    timeframe_map = {
        'minute': timedelta(minutes=1),
        'minute5': timedelta(minutes=5),
        'minute15': timedelta(minutes=15),
        'minute30': timedelta(minutes=30),
        'hourly': timedelta(hours=1),
        'hourly4': timedelta(hours=4),
        'daily': timedelta(days=1),
        'weekly': timedelta(days=7),
        'monthly': timedelta(days=30)
    }
    
    return timeframe_map.get(timeframe, timedelta(days=1))

def resample_market_data(data: pd.DataFrame, target_timeframe: str) -> pd.DataFrame:
    """
    Resample market data to a different timeframe.
    
    Args:
        data: DataFrame containing market data with DatetimeIndex
        target_timeframe: Target timeframe for resampling
        
    Returns:
        Resampled DataFrame
    """
    # Map timeframe strings to pandas resample rule
    timeframe_map = {
        'minute': '1T',
        'minute5': '5T',
        'minute15': '15T',
        'minute30': '30T',
        'hourly': '1H',
        'hourly4': '4H',
        'daily': '1D',
        'weekly': '1W',
        'monthly': '1M'
    }
    
    rule = timeframe_map.get(target_timeframe)
    if not rule:
        raise ValueError(f"Unsupported timeframe: {target_timeframe}")
    
    # Resample the data
    resampled = data.resample(rule).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })
    
    # Handle indicators if present
    indicator_columns = [col for col in data.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
    if indicator_columns:
        # For indicators, use the last value in the period
        indicators = data[indicator_columns].resample(rule).last()
        resampled = pd.concat([resampled, indicators], axis=1)
    
    return resampled

def calculate_performance_metrics(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate performance metrics from a list of trades.
    
    Args:
        trades: List of trade dictionaries
        
    Returns:
        Dictionary of performance metrics
    """
    if not trades:
        return {
            'return_pct': 0.0,
            'win_rate': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'profit_factor': 0.0,
            'trades_count': 0
        }
    
    # Extract profits/losses from trades
    pnl = [trade.get('profit_loss', 0) for trade in trades]
    
    # Calculate basic metrics
    total_trades = len(trades)
    winning_trades = sum(1 for p in pnl if p > 0)
    losing_trades = sum(1 for p in pnl if p < 0)
    
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    
    # Calculate returns
    initial_capital = 100000  # Assuming $100,000 initial capital
    final_capital = initial_capital + sum(pnl)
    return_pct = (final_capital / initial_capital - 1) * 100
    
    # Calculate Sharpe ratio (simplified)
    if len(pnl) > 1:
        returns = [p / initial_capital for p in pnl]
        sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
    else:
        sharpe_ratio = 0
    
    # Calculate maximum drawdown
    cumulative = np.cumsum(pnl)
    max_drawdown = 0
    peak = cumulative[0]
    
    for value in cumulative:
        if value > peak:
            peak = value
        drawdown = (peak - value) / initial_capital * 100 if peak > 0 else 0
        max_drawdown = max(max_drawdown, drawdown)
    
    # Calculate profit factor
    gross_profit = sum(p for p in pnl if p > 0)
    gross_loss = abs(sum(p for p in pnl if p < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    return {
        'return_pct': return_pct,
        'win_rate': win_rate,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'profit_factor': profit_factor,
        'trades_count': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'gross_profit': gross_profit,
        'gross_loss': gross_loss
    }

def filter_by_date_range(data: List[Dict[str, Any]], start_date: Optional[datetime] = None, 
                        end_date: Optional[datetime] = None, date_field: str = 'timestamp') -> List[Dict[str, Any]]:
    """
    Filter a list of dictionaries by date range.
    
    Args:
        data: List of dictionaries
        start_date: Start date for filtering
        end_date: End date for filtering
        date_field: Field name containing the date
        
    Returns:
        Filtered list of dictionaries
    """
    if not start_date and not end_date:
        return data
    
    filtered_data = []
    
    for item in data:
        # Get the date value
        date_value = item.get(date_field)
        
        # Convert string to datetime if needed
        if isinstance(date_value, str):
            try:
                date_value = datetime.fromisoformat(date_value)
            except ValueError:
                continue
        
        # Skip if no valid date
        if not date_value:
            continue
        
        # Apply filters
        if start_date and date_value < start_date:
            continue
        
        if end_date and date_value > end_date:
            continue
        
        filtered_data.append(item)
    
    return filtered_data

def merge_market_data(data_list: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Merge multiple market data DataFrames into one.
    
    Args:
        data_list: List of DataFrames to merge
        
    Returns:
        Merged DataFrame
    """
    if not data_list:
        return pd.DataFrame()
    
    if len(data_list) == 1:
        return data_list[0]
    
    # Concatenate all DataFrames
    merged = pd.concat(data_list)
    
    # Sort by index (timestamp)
    merged = merged.sort_index()
    
    # Remove duplicates
    merged = merged[~merged.index.duplicated(keep='last')]
    
    return merged
