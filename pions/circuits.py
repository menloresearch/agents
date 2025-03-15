"""
Pions Circuits Module

Defines circuit-like components for connecting agents and tools in series and parallel.
"""

import asyncio
from typing import Dict, Any, List, Union, Callable, Optional, Set, Tuple, Protocol, runtime_checkable
from enum import Enum
import uuid
import time

# Import agent and tool classes directly instead of protocols
from .agent import Agent
from .tools import BaseTool

class ComponentType(Enum):
    """Enum for component types in a circuit."""
    AGENT = "agent"
    TOOL = "tool"
    CIRCUIT = "circuit"

class CircuitComponent:
    """Base class for all circuit components."""
    
    def __init__(self, name: Optional[str] = None):
        """Initialize the circuit component."""
        self.id = str(uuid.uuid4())
        self.name = name or self.id[:8]
        self.type = ComponentType.CIRCUIT
    
    async def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process inputs through this component.
        
        Args:
            inputs: Dictionary of input values
            
        Returns:
            Dictionary of output values
        """
        raise NotImplementedError("Subclasses must implement process method")
    
    def __str__(self) -> str:
        """Return a string representation of the component."""
        return f"{self.name} ({self.type.value})"
    
    def __repr__(self) -> str:
        """Return a string representation of the component."""
        return self.__str__()

class AgentComponent(CircuitComponent):
    """Wrapper for an agent as a circuit component."""
    
    def __init__(self, agent):
        """Initialize the agent component."""
        super().__init__(name=agent.name if hasattr(agent, 'name') else str(agent))
        self.agent = agent
        self.type = ComponentType.AGENT
    
    async def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process inputs through the agent.
        
        Args:
            inputs: Dictionary of input values including 'query'
            
        Returns:
            Dictionary of output values
        """
        # Extract query from inputs
        query = inputs.get("query", "")
        
        # Filter out query to avoid duplication
        params = {k: v for k, v in inputs.items() if k != "query"}
        
        # Process using the agent
        result = await self.agent.process(query, **params)
        
        # Ensure the result is a dictionary
        if not isinstance(result, dict):
            result = {"result": result}
        
        # Add the query to the result for downstream components
        if "query" not in result:
            result["query"] = query
        
        return result

class ToolComponent(CircuitComponent):
    """Wrapper for a tool as a circuit component."""
    
    def __init__(self, tool):
        """Initialize the tool component."""
        super().__init__(name=tool.name if hasattr(tool, 'name') else str(tool))
        self.tool = tool
        self.type = ComponentType.TOOL
    
    async def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process inputs through the tool.
        
        Args:
            inputs: Dictionary of input values including 'query'
            
        Returns:
            Dictionary of output values
        """
        # Extract query from inputs
        query = inputs.get("query", "")
        
        # Filter out query to avoid duplication
        params = {k: v for k, v in inputs.items() if k != "query"}
        
        # Execute the tool
        result = await self.tool.execute(query, **params)
        
        # Ensure the result is a dictionary
        if not isinstance(result, dict):
            result = {"result": result}
        
        # Add the query to the result for downstream components
        if "query" not in result:
            result["query"] = query
        
        return result

class FunctionComponent(CircuitComponent):
    """Wrapper for a function as a circuit component."""
    
    def __init__(self, func: Callable, name: Optional[str] = None):
        """Initialize the function component."""
        super().__init__(name=name or func.__name__)
        self.func = func
    
    async def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process inputs through the function.
        
        Args:
            inputs: Dictionary of input values
            
        Returns:
            Dictionary of output values
        """
        # Call the function
        try:
            # Check if the function is a coroutine
            if asyncio.iscoroutinefunction(self.func):
                result = await self.func(**inputs)
            else:
                result = self.func(**inputs)
            
            # Ensure the result is a dictionary
            if not isinstance(result, dict):
                result = {"result": result}
            
            return result
        except Exception as e:
            # Return error information
            return {
                "error": str(e),
                "inputs": inputs
            }

class SeriesCircuit(CircuitComponent):
    """Circuit component that processes components in series."""
    
    def __init__(self, components: List[CircuitComponent], name: Optional[str] = None):
        """Initialize the series circuit."""
        super().__init__(name=name or "Series")
        self.components = components
    
    async def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process inputs through components in series.
        
        Args:
            inputs: Dictionary of input values
            
        Returns:
            Dictionary of output values from the last component
        """
        result = inputs.copy()
        
        # Process through each component in series
        for component in self.components:
            start_time = time.time()
            result = await component.process(result)
            elapsed = time.time() - start_time
            
            # For debugging
            if "circuit_trace" not in result:
                result["circuit_trace"] = []
            
            result["circuit_trace"].append({
                "component": str(component),
                "elapsed_ms": round(elapsed * 1000, 2)
            })
        
        return result
    
    def __str__(self) -> str:
        """Return a string representation of the circuit."""
        components_str = " → ".join(str(c) for c in self.components)
        return f"{self.name}: {components_str}"

class ParallelCircuit(CircuitComponent):
    """Circuit component that processes components in parallel."""
    
    def __init__(self, components: List[CircuitComponent], name: Optional[str] = None):
        """Initialize the parallel circuit."""
        super().__init__(name=name or "Parallel")
        self.components = components
    
    async def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process inputs through components in parallel.
        
        Args:
            inputs: Dictionary of input values
            
        Returns:
            Dictionary of combined output values from all components
        """
        # Create tasks for all components
        tasks = []
        for component in self.components:
            tasks.append(asyncio.create_task(component.process(inputs.copy())))
        
        # Await all tasks
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        elapsed = time.time() - start_time
        
        # Combine results from all components
        combined_result = {}
        for i, result in enumerate(results):
            component_name = str(self.components[i])
            combined_result[f"result_{i}"] = result
            combined_result[f"{component_name}"] = result
        
        # Add trace information
        if "circuit_trace" not in combined_result:
            combined_result["circuit_trace"] = []
        
        combined_result["circuit_trace"].append({
            "component": str(self),
            "elapsed_ms": round(elapsed * 1000, 2),
            "components": [str(c) for c in self.components]
        })
        
        # Add the original query
        if "query" in inputs:
            combined_result["query"] = inputs["query"]
        
        return combined_result
    
    def __str__(self) -> str:
        """Return a string representation of the circuit."""
        components_str = " || ".join(str(c) for c in self.components)
        return f"{self.name}: {components_str}"

class CircuitBuilder:
    """Helper class for building circuits."""
    
    @staticmethod
    def series(*components: Union[CircuitComponent, Agent, BaseTool, Callable], 
              name: Optional[str] = None) -> SeriesCircuit:
        """
        Create a series circuit from components.
        
        Args:
            *components: Components to include in the circuit
            name: Optional name for the circuit
            
        Returns:
            SeriesCircuit instance
        """
        circuit_components = []
        
        for component in components:
            if isinstance(component, CircuitComponent):
                circuit_components.append(component)
            elif hasattr(component, 'process') and callable(component.process):
                # Assume it's an agent if it has a process method
                circuit_components.append(AgentComponent(component))
            elif hasattr(component, 'execute') and callable(component.execute):
                # Assume it's a tool if it has an execute method
                circuit_components.append(ToolComponent(component))
            elif callable(component):
                circuit_components.append(FunctionComponent(component))
            else:
                raise ValueError(f"Unsupported component type: {type(component)}")
        
        return SeriesCircuit(circuit_components, name=name)
    
    @staticmethod
    def parallel(*components: Union[CircuitComponent, Agent, BaseTool, Callable],
               name: Optional[str] = None) -> ParallelCircuit:
        """
        Create a parallel circuit from components.
        
        Args:
            *components: Components to include in the circuit
            name: Optional name for the circuit
            
        Returns:
            ParallelCircuit instance
        """
        circuit_components = []
        
        for component in components:
            if isinstance(component, CircuitComponent):
                circuit_components.append(component)
            elif hasattr(component, 'process') and callable(component.process):
                # Assume it's an agent if it has a process method
                circuit_components.append(AgentComponent(component))
            elif hasattr(component, 'execute') and callable(component.execute):
                # Assume it's a tool if it has an execute method
                circuit_components.append(ToolComponent(component))
            elif callable(component):
                circuit_components.append(FunctionComponent(component))
            else:
                raise ValueError(f"Unsupported component type: {type(component)}")
        
        return ParallelCircuit(circuit_components, name=name)

class CircuitVisualizer:
    """Helper class for visualizing circuits."""
    
    @staticmethod
    def visualize_circuit(circuit: CircuitComponent, output_format: str = "text") -> str:
        """
        Generate a visualization of the circuit.
        
        Args:
            circuit: The circuit to visualize
            output_format: The format to use ('text', 'mermaid', etc.)
            
        Returns:
            String representation of the circuit
        """
        if output_format == "text":
            return CircuitVisualizer._visualize_text(circuit)
        elif output_format == "mermaid":
            return CircuitVisualizer._visualize_mermaid(circuit)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
    
    @staticmethod
    def _visualize_text(circuit: CircuitComponent, indent: int = 0) -> str:
        """
        Generate a text visualization of the circuit.
        
        Args:
            circuit: The circuit to visualize
            indent: The current indentation level
            
        Returns:
            String representation of the circuit
        """
        indent_str = "  " * indent
        result = f"{indent_str}{circuit}\n"
        
        if isinstance(circuit, SeriesCircuit):
            result += f"{indent_str}Series:\n"
            for component in circuit.components:
                result += CircuitVisualizer._visualize_text(component, indent + 1)
        elif isinstance(circuit, ParallelCircuit):
            result += f"{indent_str}Parallel:\n"
            for component in circuit.components:
                result += CircuitVisualizer._visualize_text(component, indent + 1)
        
        return result
    
    @staticmethod
    def _visualize_mermaid(circuit: CircuitComponent) -> str:
        """
        Generate a Mermaid flowchart visualization of the circuit.
        
        Args:
            circuit: The circuit to visualize
            
        Returns:
            String representation of the circuit in Mermaid format
        """
        # Keep track of nodes and edges
        nodes = {}
        edges = []
        
        # Generate a unique ID for each component
        def get_node_id(component):
            if component.id not in nodes:
                nodes[component.id] = {
                    "name": component.name,
                    "type": component.type.value
                }
            return component.id
        
        # Process the circuit
        def process_circuit(circuit):
            circuit_id = get_node_id(circuit)
            
            if isinstance(circuit, SeriesCircuit):
                for i in range(len(circuit.components) - 1):
                    comp1 = circuit.components[i]
                    comp2 = circuit.components[i + 1]
                    
                    comp1_id = get_node_id(comp1)
                    comp2_id = get_node_id(comp2)
                    
                    # Process subcircuits recursively
                    if isinstance(comp1, (SeriesCircuit, ParallelCircuit)):
                        process_circuit(comp1)
                    if isinstance(comp2, (SeriesCircuit, ParallelCircuit)):
                        process_circuit(comp2)
                    
                    edges.append((comp1_id, comp2_id))
            
            elif isinstance(circuit, ParallelCircuit):
                # Add edges from a virtual start node to all components
                start_id = f"{circuit_id}_start"
                end_id = f"{circuit_id}_end"
                
                nodes[start_id] = {"name": "Start", "type": "virtual"}
                nodes[end_id] = {"name": "End", "type": "virtual"}
                
                for comp in circuit.components:
                    comp_id = get_node_id(comp)
                    
                    # Process subcircuits recursively
                    if isinstance(comp, (SeriesCircuit, ParallelCircuit)):
                        process_circuit(comp)
                    
                    edges.append((start_id, comp_id))
                    edges.append((comp_id, end_id))
        
        # Start processing from the top-level circuit
        process_circuit(circuit)
        
        # Generate Mermaid flowchart
        mermaid = ["```mermaid", "flowchart TD"]
        
        # Add nodes
        for node_id, node_info in nodes.items():
            name = node_info["name"]
            node_type = node_info["type"]
            
            if node_type == "virtual":
                mermaid.append(f"    {node_id}[{name}]")
            elif node_type == "agent":
                mermaid.append(f"    {node_id}[({name})]")
            elif node_type == "tool":
                mermaid.append(f"    {node_id}[{name}]")
            else:
                mermaid.append(f"    {node_id}({name})")
        
        # Add edges
        for src, dst in edges:
            mermaid.append(f"    {src} --> {dst}")
        
        mermaid.append("```")
        
        return "\n".join(mermaid)
