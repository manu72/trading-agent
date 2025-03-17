"""
DeepResearch 4.5 Trading System - Main Application Initialization

This module provides a singleton instance of the Trading System.
"""

import os
from pathlib import Path
from .trading_system import TradingSystem

# Singleton instance
_trading_system = None

def get_trading_system(config_file=None):
    """
    Get the singleton instance of the Trading System.
    
    Args:
        config_file: Path to configuration file
    
    Returns:
        TradingSystem: The singleton instance
    """
    global _trading_system
    
    if _trading_system is None:
        # Create instance
        _trading_system = TradingSystem(config_file=config_file)
    
    return _trading_system
