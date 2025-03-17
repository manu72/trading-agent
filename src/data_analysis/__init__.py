"""
Data Analysis and Web Monitoring - Initialization

This module initializes the Data Analysis and Web Monitoring component and provides a simple interface
for other components to interact with it.
"""

from .data_analysis import DataAnalysisSystem

# Singleton instance of the Data Analysis System
_data_analysis_system = None

def get_data_analysis_system(data_update_interval: int = 3600):
    """
    Get the singleton instance of the Data Analysis System.
    
    Args:
        data_update_interval: Interval in seconds for data updates
        
    Returns:
        DataAnalysisSystem instance
    """
    global _data_analysis_system
    
    if _data_analysis_system is None:
        _data_analysis_system = DataAnalysisSystem(
            data_update_interval=data_update_interval
        )
    
    return _data_analysis_system
