"""
Pions Tools Module

Defines the protocols and base implementations for tools that can be used by agents.
"""

from typing import Dict, Any, Optional, Protocol, TypeVar, Generic

class ToolProtocol(Protocol):
    """Protocol for tools that can be registered with agents."""
    
    @property
    def name(self) -> str:
        """Return the name of the tool."""
        ...
    
    async def execute(self, *args, **kwargs) -> Dict[str, Any]:
        """Execute the tool and return the result."""
        ...

T = TypeVar('T')

class BaseTool(Generic[T]):
    """Base tool class that implements the ToolProtocol."""
    
    def __init__(self, name: str, config: Optional[T] = None):
        """Initialize the tool with a name and optional configuration."""
        self._name = name
        self.config = config
    
    @property
    def name(self) -> str:
        """Return the name of the tool."""
        return self._name
    
    async def execute(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Execute the tool and return the result.
        
        This method should be overridden by subclasses to implement specific tool behavior.
        
        Args:
            *args: Positional arguments for the tool
            **kwargs: Keyword arguments for the tool
            
        Returns:
            A dictionary containing the execution results
        """
        raise NotImplementedError("Subclasses must implement execute method")
