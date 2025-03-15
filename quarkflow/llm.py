"""
QuarkFlow LLM Interface Module

Defines the protocols and base implementations for LLM providers.
"""

from typing import Dict, Any, List, Optional, Protocol, TypeVar, Generic, Callable, Union, AsyncIterator
from dataclasses import dataclass, field
import os
import json
import uuid
import asyncio
from abc import ABC, abstractmethod


@dataclass
class LLMConfig:
    """Configuration for LLM providers"""
    api_key: str
    model: str
    base_url: Optional[str] = None
    provider: Optional[str] = None
    api_version: Optional[str] = None
    organization: Optional[str] = None
    max_retries: int = 3
    timeout: float = 30.0
    headers: Optional[Dict[str, str]] = None
    debug: bool = False
    max_tokens: Optional[int] = None
    extra_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    """Standard response format for LLM completions"""
    content: str
    model: str
    raw: Dict[str, Any] = field(default_factory=dict)
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, int]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


@dataclass
class ToolCall:
    """Representation of a tool call from an LLM"""
    id: str
    type: str = "function"
    function: Dict[str, Any] = field(default_factory=dict)


class ToolRegistry:
    """Registry for tools that can be called by the LLM"""
    
    def __init__(self):
        """Initialize an empty tool registry"""
        self.tools = {}
        self.definitions = []
    
    def register(
        self,
        name: str,
        func: Callable,
        description: str,
        parameters: Dict[str, Any]
    ) -> None:
        """Register a new tool that can be called by the model"""
        self.tools[name] = func
        
        # Create tool definition
        tool_def = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters
            }
        }
        
        self.definitions.append(tool_def)
    
    def get_definitions(self) -> List[Dict[str, Any]]:
        """Get all tool definitions for API calls"""
        return self.definitions
    
    async def execute_tool(self, call: ToolCall) -> Any:
        """Execute a tool by name"""
        name = call.function.get("name")
        if name not in self.tools:
            raise ValueError(f"Tool '{name}' not found in registry")
        
        # Parse arguments
        arguments = call.function.get("arguments", "{}")
        if isinstance(arguments, str):
            try:
                args = json.loads(arguments)
            except json.JSONDecodeError:
                args = {}
        else:
            args = arguments
        
        # Execute the tool
        func = self.tools[name]
        
        # Check if the function is async
        if asyncio.iscoroutinefunction(func):
            return await func(**args)
        else:
            return func(**args)


class LLMClientProtocol(Protocol):
    """Protocol for LLM clients that can be used with QuarkFlow agents"""
    
    @property
    def config(self) -> LLMConfig:
        """Return the client configuration"""
        ...
    
    async def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs
    ) -> LLMResponse:
        """Send a chat completion request to the LLM provider"""
        ...
    
    def register_tool(
        self,
        name: str,
        func: Callable,
        description: str,
        parameters: Dict[str, Any]
    ) -> None:
        """Register a tool that can be called by the LLM"""
        ...


class BaseLLMClient(ABC):
    """Base class for LLM clients that can be used with QuarkFlow agents"""
    
    def __init__(self, config: LLMConfig, debug: bool = False):
        """Initialize the client with configuration"""
        self.config = config
        self.debug = debug
        self.tool_registry = ToolRegistry()
    
    def register_tool(
        self,
        name: str,
        func: Callable,
        description: str,
        parameters: Dict[str, Any]
    ) -> None:
        """Register a tool that can be called by the LLM"""
        self.tool_registry.register(name, func, description, parameters)
    
    async def handle_tool_calls(
        self,
        messages: List[Dict[str, Any]],
        tool_calls: List[Dict[str, Any]],
        debug: bool = False
    ) -> List[Dict[str, Any]]:
        """Handle tool calls and append results to messages"""
        if debug:
            print("\nHandling tool calls...")
        
        # First add the assistant's message with tool calls
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": tool_calls
        })
        
        # Then handle each tool call
        for tool_call in tool_calls:
            if debug:
                print(f"\nProcessing tool call: {json.dumps(tool_call, indent=2)}")
            
            # Create ToolCall object
            call = ToolCall(
                id=tool_call.get("id", str(uuid.uuid4())),
                type=tool_call["type"],
                function=tool_call["function"]
            )
            
            try:
                # Execute the tool
                if debug:
                    print(f"\nExecuting tool: {call.function['name']}")
                    print(f"Arguments: {call.function['arguments']}")
                
                result = await self.tool_registry.execute_tool(call)
                
                if debug:
                    print(f"Tool execution result: {result}")
                
                # Add tool result to messages
                tool_message = {
                    "role": "tool",
                    "content": str(result),
                    "tool_call_id": call.id
                }
                
                messages.append(tool_message)
                
                if debug:
                    print(f"Added tool message: {json.dumps(tool_message, indent=2)}")
                    
            except Exception as e:
                if debug:
                    print(f"Error executing tool {call.function['name']}: {e}")
                messages.append({
                    "role": "tool",
                    "content": f"Error: {str(e)}",
                    "tool_call_id": call.id
                })
        
        return messages
    
    @abstractmethod
    async def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs
    ) -> LLMResponse:
        """Send a chat completion request to the LLM provider"""
        pass


# Optional Bhumi adapter if Bhumi is installed
try:
    import bhumi
    
    class BhumiAdapter(BaseLLMClient):
        """Adapter for Bhumi LLM client"""
        
        def __init__(self, config: LLMConfig, debug: bool = False):
            """Initialize the Bhumi client"""
            super().__init__(config, debug)
            
            # Import Bhumi components
            from bhumi.base_client import BaseLLMClient as BhumiBaseLLMClient
            from bhumi.base_client import LLMConfig as BhumiLLMConfig
            
            # Create Bhumi config
            bhumi_config = BhumiLLMConfig(
                api_key=config.api_key,
                model=config.model,
                base_url=config.base_url,
                provider=config.provider,
                api_version=config.api_version,
                organization=config.organization,
                max_retries=config.max_retries,
                timeout=config.timeout,
                headers=config.headers,
                debug=config.debug,
                max_tokens=config.max_tokens,
                extra_config=config.extra_config
            )
            
            # Create Bhumi client
            self.bhumi_client = BhumiBaseLLMClient(bhumi_config, debug=debug)
        
        def register_tool(
            self,
            name: str,
            func: Callable,
            description: str,
            parameters: Dict[str, Any]
        ) -> None:
            """Register a tool with both the adapter and Bhumi client"""
            super().register_tool(name, func, description, parameters)
            self.bhumi_client.register_tool(name, func, description, parameters)
        
        async def chat_completion(
            self,
            messages: List[Dict[str, Any]],
            temperature: float = 0.7,
            max_tokens: Optional[int] = None,
            stream: bool = False,
            **kwargs
        ) -> LLMResponse:
            """Send a chat completion request using Bhumi"""
            response = await self.bhumi_client.chat_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
                **kwargs
            )
            
            # Convert Bhumi response to our standard format
            return LLMResponse(
                content=response.content,
                model=response.model if hasattr(response, 'model') else self.config.model,
                raw=response.raw if hasattr(response, 'raw') else {},
                finish_reason=response.finish_reason if hasattr(response, 'finish_reason') else None,
                usage=response.usage if hasattr(response, 'usage') else None,
                tool_calls=response.tool_calls if hasattr(response, 'tool_calls') else None
            )

except ImportError:
    # Bhumi is not installed
    pass


class MockLLMClient(BaseLLMClient):
    """Mock LLM client for testing"""
    
    def __init__(self, config: LLMConfig, debug: bool = False, responses: Dict[str, str] = None):
        """Initialize with optional predefined responses"""
        super().__init__(config, debug)
        self.responses = responses or {}
    
    async def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs
    ) -> LLMResponse:
        """Return a mock response"""
        # Get the last user message
        last_message = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        
        # Look for a predefined response or use a default
        content = self.responses.get(last_message, "This is a mock response from the LLM.")
        
        return LLMResponse(
            content=content,
            model="mock-model",
            raw={},
            finish_reason="stop"
        )
