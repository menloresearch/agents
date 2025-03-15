"""
Pions Controller Module

Defines the controller agent that orchestrates multiple specialized agents.
"""

import asyncio
import uuid
import datetime
import json
import os
from typing import Dict, Any, List, Optional, Union, Callable

from .agent import AgentProtocol
from .tools import ToolProtocol

class ControllerAgent:
    """Main controller agent that orchestrates specialized agents and tools."""
    
    def __init__(
        self,
        name: str = "controller",
        conversation_id: Optional[str] = None,
        output_dir: Optional[str] = None
    ):
        """Initialize the controller agent.
        
        Args:
            name: Name of the controller agent
            conversation_id: Optional ID for the conversation
            output_dir: Optional directory to store output files
        """
        self.name = name
        self.conversation_id = conversation_id or str(uuid.uuid4())
        self.output_dir = output_dir
        
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize agent and tool registries
        self.agents: Dict[str, AgentProtocol] = {}
        self.tools: Dict[str, ToolProtocol] = {}
        
        # Initialize conversation history
        self.conversation_history: List[Dict[str, Any]] = []
    
    def register_agent(self, agent: AgentProtocol) -> None:
        """Register a specialized agent with the controller.
        
        Args:
            agent: The agent to register
        """
        self.agents[agent.name] = agent
    
    def register_tool(self, tool: ToolProtocol) -> None:
        """Register a tool with the controller.
        
        Args:
            tool: The tool to register
        """
        self.tools[tool.name] = tool
    
    def _add_to_history(self, entry: Dict[str, Any]) -> None:
        """Add an entry to the conversation history.
        
        Args:
            entry: The entry to add
        """
        # Add timestamp
        entry["timestamp"] = datetime.datetime.now().isoformat()
        self.conversation_history.append(entry)
        
        # Save history to file if output directory is specified
        if self.output_dir:
            history_file = os.path.join(
                self.output_dir, f"conversation_{self.conversation_id}.json"
            )
            with open(history_file, "w", encoding="utf-8") as f:
                json.dump(self.conversation_history, f, indent=2)
    
    async def process(self, query: str, agent_name: Optional[str] = None) -> Dict[str, Any]:
        """Process a query using the appropriate agent.
        
        Args:
            query: The query to process
            agent_name: Optional name of the agent to use
            
        Returns:
            Processing results
            
        Raises:
            ValueError: If the specified agent does not exist
        """
        # Add query to history
        self._add_to_history({
            "role": "user",
            "content": query
        })
        
        # Process query
        if agent_name:
            # Use specified agent
            if agent_name not in self.agents:
                raise ValueError(f"Agent '{agent_name}' is not registered")
            agent = self.agents[agent_name]
            result = await agent.process(query)
        else:
            # Simple implementation: just use the first registered agent
            # In a more sophisticated version, this would determine the best agent
            if not self.agents:
                raise ValueError("No agents are registered")
            agent = next(iter(self.agents.values()))
            result = await agent.process(query)
        
        # Add result to history
        self._add_to_history({
            "role": "assistant",
            "content": result
        })
        
        return result
    
    async def execute_tool(self, tool_name: str, *args, **kwargs) -> Dict[str, Any]:
        """Execute a tool.
        
        Args:
            tool_name: Name of the tool to execute
            *args: Positional arguments for the tool
            **kwargs: Keyword arguments for the tool
            
        Returns:
            Tool execution results
            
        Raises:
            ValueError: If the specified tool does not exist
        """
        if tool_name not in self.tools:
            raise ValueError(f"Tool '{tool_name}' is not registered")
        
        tool = self.tools[tool_name]
        result = await tool.execute(*args, **kwargs)
        
        # Add tool execution to history
        self._add_to_history({
            "role": "system",
            "content": f"Executed tool '{tool_name}'",
            "tool_result": result
        })
        
        return result
    
    async def run_pipeline(
        self,
        query: str,
        pipeline: List[Union[str, Callable]]
    ) -> Dict[str, Any]:
        """Run a pipeline of agents and tools.
        
        Args:
            query: The query to process
            pipeline: List of agent/tool names or callable functions to execute in sequence
            
        Returns:
            Pipeline execution results
            
        Raises:
            ValueError: If a pipeline element is invalid
        """
        result = {"query": query}
        
        for element in pipeline:
            if isinstance(element, str):
                # Element is an agent or tool name
                if element in self.agents:
                    # Process using agent
                    # Make a copy of result without 'query' to prevent duplicate args
                    params = {k: v for k, v in result.items() if k != 'query'}
                    agent_result = await self.agents[element].process(query, **params)
                    result.update(agent_result)
                elif element in self.tools:
                    # Execute tool
                    # Make a copy of result without 'query' to prevent duplicate args
                    params = {k: v for k, v in result.items() if k != 'query'}
                    tool_result = await self.tools[element].execute(query, **params)
                    result.update(tool_result)
                else:
                    raise ValueError(f"Pipeline element '{element}' is not a registered agent or tool")
            elif callable(element):
                # Element is a callable function
                # Make a copy of result without 'query' to prevent duplicate args
                params = {k: v for k, v in result.items() if k != 'query'}
                func_result = await element(query, **params)
                result.update(func_result)
            else:
                raise ValueError(f"Pipeline element must be a string or callable, got {type(element)}")
        
        return result
