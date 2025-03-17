"""
Trading Knowledge Base - Initialization

This module initializes the Knowledge Base component and provides a simple interface
for other components to interact with it.
"""

from pathlib import Path
import os
from .knowledge_base import KnowledgeBase

# Singleton instance of the Knowledge Base
_knowledge_base = None

def get_knowledge_base(base_path=None):
    """
    Get the singleton instance of the Knowledge Base.
    
    Args:
        base_path: Optional base path for the Knowledge Base data
        
    Returns:
        KnowledgeBase instance
    """
    global _knowledge_base
    
    if _knowledge_base is None:
        if base_path is None:
            # Default to a data directory in the project root
            base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data')
        
        _knowledge_base = KnowledgeBase(base_path)
    
    return _knowledge_base
