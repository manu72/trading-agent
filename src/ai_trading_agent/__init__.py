"""
AI Trading Agent - Initialization

This module initializes the AI Trading Agent component and provides a simple interface
for other components to interact with it.
"""

from .agent import AITradingAgent

# Singleton instance of the AI Trading Agent
_ai_trading_agent = None

def get_ai_trading_agent():
    """
    Get the singleton instance of the AI Trading Agent.
    
    Returns:
        AITradingAgent instance
    """
    global _ai_trading_agent
    
    if _ai_trading_agent is None:
        _ai_trading_agent = AITradingAgent()
    
    return _ai_trading_agent
