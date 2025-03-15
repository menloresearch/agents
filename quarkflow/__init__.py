"""
Quarkflow - Circuit-Based Agent Library

A lightweight library for creating and managing AI agents with circuit-based workflows.
"""

from .agent import Agent, AgentProtocol
from .tools import BaseTool, ToolProtocol
from .controller import ControllerAgent
from .circuits import (
    CircuitComponent, 
    AgentComponent, 
    ToolComponent, 
    FunctionComponent,
    SeriesCircuit, 
    ParallelCircuit, 
    CircuitBuilder, 
    CircuitVisualizer
)

# Package metadata
__version__ = "0.1.0"
__author__ = "Menlo Deep Labs"
__email__ = "info@menlodeep.com"
__license__ = "MIT"
__description__ = "A lightweight library for creating and managing AI agent circuits"

# Bhumi integration is used by default for LLM inference
# Import this only to check if bhumi is installed
try:
    import bhumi
except ImportError:
    import warnings
    warnings.warn(
        "The bhumi package is not installed. LLM functionality will not work. "
        "Install it with 'pip install bhumi'."
    )
