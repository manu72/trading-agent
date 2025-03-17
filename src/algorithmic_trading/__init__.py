"""
Algorithmic Trading System - Initialization

This module initializes the Algorithmic Trading System component and provides a simple interface
for other components to interact with it.
"""

from .trading_system import AlgorithmicTradingSystem

# Singleton instance of the Algorithmic Trading System
_algorithmic_trading_system = None

def get_algorithmic_trading_system(paper_trading: bool = True, max_positions: int = 10):
    """
    Get the singleton instance of the Algorithmic Trading System.
    
    Args:
        paper_trading: Whether to use paper trading mode
        max_positions: Maximum number of positions to hold simultaneously
        
    Returns:
        AlgorithmicTradingSystem instance
    """
    global _algorithmic_trading_system
    
    if _algorithmic_trading_system is None:
        _algorithmic_trading_system = AlgorithmicTradingSystem(
            paper_trading=paper_trading,
            max_positions=max_positions
        )
    
    return _algorithmic_trading_system
