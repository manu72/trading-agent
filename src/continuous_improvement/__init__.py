"""
Continuous Improvement Mechanism - Initialization Module

This module provides a singleton instance of the Continuous Improvement System.
"""

import os
from pathlib import Path
from .continuous_improvement import ContinuousImprovementSystem

# Singleton instance
_continuous_improvement_system = None

def get_continuous_improvement_system():
    """
    Get the singleton instance of the Continuous Improvement System.
    
    Returns:
        ContinuousImprovementSystem: The singleton instance
    """
    global _continuous_improvement_system
    
    if _continuous_improvement_system is None:
        # Set models directory
        models_dir = Path(os.path.dirname(os.path.abspath(__file__))) / 'models'
        
        # Create models directory if it doesn't exist
        models_dir.mkdir(exist_ok=True)
        
        # Create instance
        _continuous_improvement_system = ContinuousImprovementSystem(
            models_dir=str(models_dir),
            performance_review_interval=86400  # Daily reviews
        )
    
    return _continuous_improvement_system
