"""
Trading Knowledge Base - Core Module

This module implements the central data repository for the DeepResearch 4.5 Trading System.
It provides storage and retrieval capabilities for market data, trading signals, orders,
and performance metrics.
"""

import os
import json
import datetime
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

class KnowledgeBase:
    """
    Main class for the Trading Knowledge Base component.
    
    The Knowledge Base serves as the central repository for all data in the trading system,
    including market data, trading signals, orders, and performance metrics.
    """
    
    def __init__(self, base_path: str = None):
        """
        Initialize the Knowledge Base with the specified base path.
        
        Args:
            base_path: Base directory for storing Knowledge Base data
        """
        if base_path is None:
            base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data')
        
        self.base_path = Path(base_path)
        
        # Create directory structure
        self._create_directory_structure()
        
        # Initialize data stores
        self.market_data = MarketDataStore(self.base_path / 'market_data')
        self.signals = SignalStore(self.base_path / 'signals')
        self.orders = OrderStore(self.base_path / 'orders')
        self.performance = PerformanceStore(self.base_path / 'performance')
        self.metadata = MetadataStore(self.base_path / 'metadata')
        
    def _create_directory_structure(self):
        """Create the necessary directory structure for the Knowledge Base."""
        directories = [
            self.base_path,
            self.base_path / 'market_data',
            self.base_path / 'signals',
            self.base_path / 'orders',
            self.base_path / 'performance',
            self.base_path / 'metadata'
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def store_market_data(self, symbol: str, data: pd.DataFrame):
        """
        Store market data for a symbol.
        
        Args:
            symbol: The ticker symbol
            data: DataFrame containing market data
        """
        return self.market_data.store(symbol, data)
    
    def get_market_data(self, symbol: str, start_time: Optional[datetime.datetime] = None, 
                       end_time: Optional[datetime.datetime] = None, 
                       timeframe: str = 'daily') -> pd.DataFrame:
        """
        Retrieve market data for a symbol within a time range.
        
        Args:
            symbol: The ticker symbol
            start_time: Start of the time range
            end_time: End of the time range
            timeframe: Data timeframe (e.g., 'daily', 'hourly')
            
        Returns:
            DataFrame containing the requested market data
        """
        return self.market_data.get(symbol, start_time, end_time, timeframe)
    
    def store_trading_signal(self, signal: Dict[str, Any]):
        """
        Store a trading signal.
        
        Args:
            signal: Dictionary containing signal information
        """
        return self.signals.store(signal)
    
    def get_trading_signals(self, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Retrieve trading signals based on filters.
        
        Args:
            filters: Dictionary of filter criteria
            
        Returns:
            List of signal dictionaries matching the filters
        """
        return self.signals.get(filters)
    
    def store_order(self, order: Dict[str, Any]):
        """
        Store order information.
        
        Args:
            order: Dictionary containing order information
        """
        return self.orders.store(order)
    
    def get_orders(self, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Retrieve orders based on filters.
        
        Args:
            filters: Dictionary of filter criteria
            
        Returns:
            List of order dictionaries matching the filters
        """
        return self.orders.get(filters)
    
    def store_strategy_performance(self, strategy: str, metrics: Dict[str, Any]):
        """
        Store performance metrics for a strategy.
        
        Args:
            strategy: Name of the strategy
            metrics: Dictionary of performance metrics
        """
        return self.performance.store_strategy_performance(strategy, metrics)
    
    def get_strategy_performance(self, strategy: str, timeframe: str = 'all') -> Dict[str, Any]:
        """
        Retrieve performance metrics for a strategy.
        
        Args:
            strategy: Name of the strategy
            timeframe: Time period for the metrics
            
        Returns:
            Dictionary of performance metrics
        """
        return self.performance.get_strategy_performance(strategy, timeframe)
    
    def store_metadata(self, key: str, value: Any):
        """
        Store metadata information.
        
        Args:
            key: Metadata key
            value: Metadata value
        """
        return self.metadata.store(key, value)
    
    def get_metadata(self, key: str) -> Any:
        """
        Retrieve metadata information.
        
        Args:
            key: Metadata key
            
        Returns:
            Metadata value
        """
        return self.metadata.get(key)


class MarketDataStore:
    """Store for market data (prices, volumes, indicators)."""
    
    def __init__(self, base_path: Path):
        """
        Initialize the MarketDataStore.
        
        Args:
            base_path: Directory for storing market data
        """
        self.base_path = base_path
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def store(self, symbol: str, data: pd.DataFrame):
        """
        Store market data for a symbol.
        
        Args:
            symbol: The ticker symbol
            data: DataFrame containing market data
        """
        # Ensure the data has a datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Market data must have a DatetimeIndex")
        
        # Create symbol directory if it doesn't exist
        symbol_dir = self.base_path / symbol
        symbol_dir.mkdir(exist_ok=True)
        
        # Determine the timeframe from the data
        timeframe = self._determine_timeframe(data)
        
        # Save the data to a parquet file
        file_path = symbol_dir / f"{timeframe}.parquet"
        data.to_parquet(file_path)
        
        return True
    
    def get(self, symbol: str, start_time: Optional[datetime.datetime] = None, 
           end_time: Optional[datetime.datetime] = None, 
           timeframe: str = 'daily') -> pd.DataFrame:
        """
        Retrieve market data for a symbol within a time range.
        
        Args:
            symbol: The ticker symbol
            start_time: Start of the time range
            end_time: End of the time range
            timeframe: Data timeframe (e.g., 'daily', 'hourly')
            
        Returns:
            DataFrame containing the requested market data
        """
        # Check if the data file exists
        symbol_dir = self.base_path / symbol
        file_path = symbol_dir / f"{timeframe}.parquet"
        
        if not file_path.exists():
            return pd.DataFrame()
        
        # Load the data
        data = pd.read_parquet(file_path)
        
        # Filter by time range if specified
        if start_time is not None:
            data = data[data.index >= pd.Timestamp(start_time)]
        
        if end_time is not None:
            data = data[data.index <= pd.Timestamp(end_time)]
        
        return data
    
    def _determine_timeframe(self, data: pd.DataFrame) -> str:
        """
        Determine the timeframe of the data based on the index.
        
        Args:
            data: DataFrame with DatetimeIndex
            
        Returns:
            String representing the timeframe
        """
        if len(data) <= 1:
            return 'daily'  # Default if not enough data points
        
        # Calculate the median time difference
        diff = pd.Series(data.index).diff().median()
        
        if diff < pd.Timedelta(minutes=5):
            return 'minute'
        elif diff < pd.Timedelta(hours=1):
            return 'minute5'
        elif diff < pd.Timedelta(hours=2):
            return 'hourly'
        elif diff < pd.Timedelta(days=1):
            return 'hourly4'
        elif diff < pd.Timedelta(days=7):
            return 'daily'
        elif diff < pd.Timedelta(days=30):
            return 'weekly'
        else:
            return 'monthly'


class SignalStore:
    """Store for trading signals."""
    
    def __init__(self, base_path: Path):
        """
        Initialize the SignalStore.
        
        Args:
            base_path: Directory for storing signals
        """
        self.base_path = base_path
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Create index file if it doesn't exist
        self.index_file = self.base_path / 'index.json'
        if not self.index_file.exists():
            with open(self.index_file, 'w') as f:
                json.dump([], f)
    
    def store(self, signal: Dict[str, Any]):
        """
        Store a trading signal.
        
        Args:
            signal: Dictionary containing signal information
        """
        # Ensure the signal has an ID
        if 'signal_id' not in signal:
            signal['signal_id'] = self._generate_signal_id()
        
        # Ensure the signal has a timestamp
        if 'timestamp' not in signal:
            signal['timestamp'] = datetime.datetime.now().isoformat()
        
        # Save the signal to a JSON file
        signal_file = self.base_path / f"{signal['signal_id']}.json"
        with open(signal_file, 'w') as f:
            json.dump(signal, f, indent=2)
        
        # Update the index
        self._update_index(signal)
        
        return signal['signal_id']
    
    def get(self, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Retrieve trading signals based on filters.
        
        Args:
            filters: Dictionary of filter criteria
            
        Returns:
            List of signal dictionaries matching the filters
        """
        # Load the index
        with open(self.index_file, 'r') as f:
            index = json.load(f)
        
        # Apply filters if specified
        if filters:
            filtered_index = []
            for entry in index:
                match = True
                for key, value in filters.items():
                    if key not in entry or entry[key] != value:
                        match = False
                        break
                if match:
                    filtered_index.append(entry)
        else:
            filtered_index = index
        
        # Load the full signal data for each matching index entry
        signals = []
        for entry in filtered_index:
            signal_file = self.base_path / f"{entry['signal_id']}.json"
            if signal_file.exists():
                with open(signal_file, 'r') as f:
                    signals.append(json.load(f))
        
        return signals
    
    def _generate_signal_id(self) -> str:
        """
        Generate a unique signal ID.
        
        Returns:
            Unique signal ID string
        """
        return f"sig_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}_{np.random.randint(1000, 9999)}"
    
    def _update_index(self, signal: Dict[str, Any]):
        """
        Update the signal index with a new signal.
        
        Args:
            signal: Signal dictionary
        """
        # Create index entry with key fields
        index_entry = {
            'signal_id': signal['signal_id'],
            'timestamp': signal['timestamp'],
            'symbol': signal.get('symbol', ''),
            'direction': signal.get('direction', ''),
            'strategy': signal.get('strategy', '')
        }
        
        # Load the current index
        with open(self.index_file, 'r') as f:
            index = json.load(f)
        
        # Add the new entry
        index.append(index_entry)
        
        # Save the updated index
        with open(self.index_file, 'w') as f:
            json.dump(index, f, indent=2)


class OrderStore:
    """Store for order information."""
    
    def __init__(self, base_path: Path):
        """
        Initialize the OrderStore.
        
        Args:
            base_path: Directory for storing orders
        """
        self.base_path = base_path
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Create index file if it doesn't exist
        self.index_file = self.base_path / 'index.json'
        if not self.index_file.exists():
            with open(self.index_file, 'w') as f:
                json.dump([], f)
    
    def store(self, order: Dict[str, Any]):
        """
        Store order information.
        
        Args:
            order: Dictionary containing order information
        """
        # Ensure the order has an ID
        if 'order_id' not in order:
            order['order_id'] = self._generate_order_id()
        
        # Ensure the order has a timestamp
        if 'timestamp' not in order:
            order['timestamp'] = datetime.datetime.now().isoformat()
        
        # Save the order to a JSON file
        order_file = self.base_path / f"{order['order_id']}.json"
        with open(order_file, 'w') as f:
            json.dump(order, f, indent=2)
        
        # Update the index
        self._update_index(order)
        
        return order['order_id']
    
    def get(self, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Retrieve orders based on filters.
        
        Args:
            filters: Dictionary of filter criteria
            
        Returns:
            List of order dictionaries matching the filters
        """
        # Load the index
        with open(self.index_file, 'r') as f:
            index = json.load(f)
        
        # Apply filters if specified
        if filters:
            filtered_index = []
            for entry in index:
                match = True
                for key, value in filters.items():
                    if key not in entry or entry[key] != value:
                        match = False
                        break
                if match:
                    filtered_index.append(entry)
        else:
            filtered_index = index
        
        # Load the full order data for each matching index entry
        orders = []
        for entry in filtered_index:
            order_file = self.base_path / f"{entry['order_id']}.json"
            if order_file.exists():
                with open(order_file, 'r') as f:
                    orders.append(json.load(f))
        
        return orders
    
    def _generate_order_id(self) -> str:
        """
        Generate a unique order ID.
        
        Returns:
            Unique order ID string
        """
        return f"ord_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}_{np.random.randint(1000, 9999)}"
    
    def _update_index(self, order: Dict[str, Any]):
        """
        Update the order index with a new order.
        
        Args:
            order: Order dictionary
        """
        # Create index entry with key fields
        index_entry = {
            'order_id': order['order_id'],
            'timestamp': order['timestamp'],
            'symbol': order.get('symbol', ''),
            'direction': order.get('direction', ''),
            'status': order.get('status', '')
        }
        
        # Load the current index
        with open(self.index_file, 'r') as f:
            index = json.load(f)
        
        # Add the new entry
        index.append(index_entry)
        
        # Save the updated index
        with open(self.index_file, 'w') as f:
            json.dump(index, f, indent=2)


class PerformanceStore:
    """Store for performance metrics."""
    
    def __init__(self, base_path: Path):
        """
        Initialize the PerformanceStore.
        
        Args:
            base_path: Directory for storing performance metrics
        """
        self.base_path = base_path
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def store_strategy_performance(self, strategy: str, metrics: Dict[str, Any]):
        """
        Store performance metrics for a strategy.
        
        Args:
            strategy: Name of the strategy
            metrics: Dictionary of performance metrics
        """
        # Ensure the metrics have a timestamp
        if 'timestamp' not in metrics:
            metrics['timestamp'] = datetime.datetime.now().isoformat()
        
        # Create strategy directory if it doesn't exist
        strategy_dir = self.base_path / strategy
        strategy_dir.mkdir(exist_ok=True)
        
        # Save the metrics to a JSON file
        file_name = f"{metrics['timestamp'].split('T')[0]}.json"
        metrics_file = strategy_dir / file_name
        
        # If the file exists, load and update it
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                existing_metrics = json.load(f)
            
            # Update with new metrics
            existing_metrics.update(metrics)
            metrics = existing_metrics
        
        # Save the metrics
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return True
    
    def get_strategy_performance(self, strategy: str, timeframe: str = 'all') -> Dict[str, Any]:
        """
        Retrieve performance metrics for a strategy.
        
        Args:
            strategy: Name of the strategy
            timeframe: Time period for the metrics
            
        Returns:
            Dictionary of performance metrics
        """
        strategy_dir = self.base_path / strategy
        
        if not strategy_dir.exists():
            return {}
        
        # Get all metric files
        metric_files = list(strategy_dir.glob('*.json'))
        
        if not metric_files:
            return {}
        
        # Filter by timeframe if specified
        if timeframe != 'all':
            today = datetime.date.today()
            
            if timeframe == 'today':
                start_date = today
            elif timeframe == 'week':
                start_date = today - datetime.timedelta(days=7)
            elif timeframe == 'month':
                start_date = today - datetime.timedelta(days=30)
            elif timeframe == 'year':
                start_date = today - datetime.timedelta(days=365)
            else:
                start_date = None
            
            if start_date:
                metric_files = [f for f in metric_files if self._file_date(f) >= start_date]
        
        # Load and combine all metrics
        combined_metrics = {}
        for file in metric_files:
            with open(file, 'r') as f:
                metrics = json.load(f)
            combined_metrics.update(metrics)
        
        return combined_metrics
    
    def _file_date(self, file_path: Path) -> datetime.date:
        """
        Extract the date from a file name.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Date object
        """
        date_str = file_path.stem
        return datetime.date.fromisoformat(date_str)


class MetadataStore:
    """Store for system metadata."""
    
    def __init__(self, base_path: Path):
        """
        Initialize the MetadataStore.
        
        Args:
            base_path: Directory for storing metadata
        """
        self.base_path = base_path
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Create metadata file if it doesn't exist
        self.metadata_file = self.base_path / 'metadata.json'
        if not self.metadata_file.exists():
            with open(self.metadata_file, 'w') as f:
                json.dump({}, f)
    
    def store(self, key: str, value: Any):
        """
        Store metadata information.
        
        Args:
            key: Metadata key
            value: Metadata value
        """
        # Load current metadata
        with open(self.metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Update with new value
        metadata[key] = value
        
        # Save updated metadata
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return True
    
    def get(self, key: str) -> Any:
        """
        Retrieve metadata information.
        
        Args:
            key: Metadata key
            
        Returns:
            Metadata value
        """
        # Load metadata
        with open(self.metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Return the requested value
        return metadata.get(key)
