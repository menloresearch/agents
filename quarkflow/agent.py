"""
Pions Agent Module

Defines the protocols and base implementations for agents in the pions library.
"""

import asyncio
import os
from typing import Dict, Any, List, Optional, Protocol, TypeVar, Generic

from bhumi.base_client import BaseLLMClient, LLMConfig

class AgentProtocol(Protocol):
    """Protocol for specialized agents that can be registered with a controller agent."""
    
    @property
    def name(self) -> str:
        """Return the name of the agent."""
        ...
    
    async def process(self, query: str, *args, **kwargs) -> Dict[str, Any]:
        """Process a query using this specialized agent."""
        ...

T = TypeVar('T')

class Agent(Generic[T]):
    """Base agent class that implements the AgentProtocol."""
    
    def __init__(self, name: str, config: Optional[T] = None, llm_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the agent with a name and optional configurations.
        
        Args:
            name: The name of the agent
            config: Optional agent-specific configuration
            llm_config: Optional LLM configuration for bhumi
        """
        self._name = name
        self.config = config
        
        # Initialize LLM client if configuration is provided
        self.llm_client = None
        if llm_config:
            api_key = llm_config.get("api_key") or os.getenv(llm_config.get("api_key_env", ""))
            model = llm_config.get("model", "gemini/gemini-1.5-pro-latest")
            debug = llm_config.get("debug", False)
            
            if api_key:
                llm_config = LLMConfig(
                    api_key=api_key,
                    model=model,
                    debug=debug
                )
                self.llm_client = BaseLLMClient(llm_config, debug=debug)
    
    @property
    def name(self) -> str:
        """Return the name of the agent."""
        return self._name
    
    async def generate(self, prompt: str, system_prompt: Optional[str] = None, 
                      temperature: float = 0.7) -> str:
        """
        Generate text using the LLM client.
        
        Args:
            prompt: The prompt to send to the LLM
            system_prompt: Optional system prompt
            temperature: Temperature for generation
            
        Returns:
            Generated text from the LLM
            
        Raises:
            ValueError: If the LLM client is not initialized
        """
        if not self.llm_client:
            raise ValueError("LLM client not initialized. Please provide llm_config during initialization.")
        
        response = await self.llm_client.chat_completion(
            messages=[
                {"role": "system", "content": system_prompt or "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature
        )
        
        return response.content
    
    async def process(self, query: str, *args, **kwargs) -> Dict[str, Any]:
        """
        Process a query using this agent.
        
        This method should be overridden by subclasses to implement specific agent behavior.
        
        Args:
            query: The user query or input to process
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
            
        Returns:
            A dictionary containing the processing results
        """
        raise NotImplementedError("Subclasses must implement process method")
