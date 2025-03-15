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
    """Circuit component that processes components in series with task queueing."""
    
    def __init__(self, components: List[CircuitComponent], name: Optional[str] = None, max_queue_size: int = 0):
        """Initialize the series circuit.
        
        Args:
            components: List of components to process in series
            name: Optional name for the circuit
            max_queue_size: Maximum size of the task queue (0 for unlimited)
        """
        super().__init__(name=name or "Series")
        self.components = components
        self.max_queue_size = max_queue_size
        self._queue = asyncio.Queue(maxsize=max_queue_size)
        self._is_processing = False
    
    async def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process inputs through components in series with proper queueing.
        
        Args:
            inputs: Dictionary of input values
            
        Returns:
            Dictionary of output values from the last component
        """
        # Create a future to track the completion of this task
        completion_future = asyncio.Future()
        
        # Package the task with its inputs and completion future
        task_package = {
            "inputs": inputs.copy(),
            "completion_future": completion_future
        }
        
        # Put the task in the queue
        await self._queue.put(task_package)
        
        # Start processing the queue if not already running
        if not self._is_processing:
            # Start the queue processor without awaiting it
            asyncio.create_task(self._process_queue())
        
        # Wait for this specific task to complete
        return await completion_future
    
    async def _process_queue(self):
        """Process tasks from the queue one at a time."""
        self._is_processing = True
        
        try:
            while not self._queue.empty():
                # Get the next task from the queue
                task_package = await self._queue.get()
                
                try:
                    result = task_package["inputs"]
                    trace_data = []
                    
                    # Process sequentially through each component
                    for component in self.components:
                        start_time = time.time()
                        result = await component.process(result)
                        elapsed = time.time() - start_time
                        
                        # Collect trace data
                        trace_data.append({
                            "component": str(component),
                            "elapsed_ms": round(elapsed * 1000, 2)
                        })
                    
                    # Add trace information to the result
                    if "circuit_trace" not in result:
                        result["circuit_trace"] = []
                    
                    result["circuit_trace"].extend(trace_data)
                    
                    # Mark this task as completed with the result
                    task_package["completion_future"].set_result(result)
                    
                except Exception as e:
                    # Handle errors and still mark the task as complete
                    error_result = {
                        "error": str(e),
                        "inputs": task_package["inputs"]
                    }
                    task_package["completion_future"].set_result(error_result)
                
                # Mark the task as done in the queue
                self._queue.task_done()
        
        finally:
            self._is_processing = False
    
    def __str__(self) -> str:
        """Return a string representation of the circuit."""
        components_str = " → ".join(str(c) for c in self.components)
        return f"{self.name}: {components_str}"

class ParallelCircuit(CircuitComponent):
    """Circuit component that processes components in parallel with concurrency control."""
    
    def __init__(self, components: List[CircuitComponent], name: Optional[str] = None, max_concurrency: int = 0):
        """Initialize the parallel circuit.
        
        Args:
            components: List of components to process in parallel
            name: Optional name for the circuit
            max_concurrency: Maximum number of concurrent tasks (0 for unlimited)
        """
        super().__init__(name=name or "Parallel")
        self.components = components
        self.max_concurrency = max_concurrency if max_concurrency > 0 else len(components)
        self._task_queue = asyncio.Queue()
        self._semaphore = asyncio.Semaphore(self.max_concurrency)
        self._is_processing = False
    
    async def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process inputs through components in parallel with controlled concurrency.
        
        Args:
            inputs: Dictionary of input values
            
        Returns:
            Dictionary of combined output values from all components
        """
        # Create a future to track the completion of all components
        completion_future = asyncio.Future()
        
        # Put all components' tasks in the queue
        for component in self.components:
            await self._task_queue.put({
                "component": component,
                "inputs": inputs.copy()
            })
        
        # Start the task processor if not already running
        if not self._is_processing:
            asyncio.create_task(self._process_task_queue(completion_future, len(self.components)))
        
        # Wait for all components to complete
        return await completion_future
    
    async def _process_task_queue(self, completion_future, total_tasks):
        """
        Process tasks from the queue with controlled concurrency.
        
        Args:
            completion_future: Future to complete when all tasks are done
            total_tasks: Total number of tasks to process
        """
        self._is_processing = True
        start_time = time.time()
        results = []
        
        # Create a list to store worker tasks
        workers = []
        
        try:
            # Start worker tasks up to max_concurrency
            for _ in range(min(self.max_concurrency, total_tasks)):
                worker = asyncio.create_task(self._worker())
                workers.append(worker)
            
            # Wait for all workers to complete
            worker_results = await asyncio.gather(*workers)
            
            # Combine all worker results
            for worker_result in worker_results:
                results.extend(worker_result)
            
            # Calculate total elapsed time
            elapsed = time.time() - start_time
            
            # Combine results from all components
            combined_result = {}
            for i, (component, result) in enumerate(results):
                component_name = str(component)
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
            
            # Add the original query if present in any of the inputs
            if results and "query" in results[0][1]:
                combined_result["query"] = results[0][1]["query"]
            
            # Set the result in the completion future
            completion_future.set_result(combined_result)
        
        except Exception as e:
            # Handle errors
            completion_future.set_result({
                "error": str(e),
                "circuit": str(self)
            })
        
        finally:
            self._is_processing = False
    
    async def _worker(self):
        """
        Worker to process tasks from the queue.
        
        Returns:
            List of (component, result) tuples
        """
        results = []
        
        while not self._task_queue.empty():
            # Get a task from the queue
            try:
                task = await self._task_queue.get()
                component = task["component"]
                inputs = task["inputs"]
                
                # Acquire semaphore to control concurrency
                async with self._semaphore:
                    # Process the component
                    result = await component.process(inputs)
                    results.append((component, result))
                
                # Mark task as done
                self._task_queue.task_done()
            
            except Exception as e:
                # Handle errors for individual components
                self._task_queue.task_done()
                if "component" in task:
                    results.append((task["component"], {"error": str(e)}))
        
        return results
    
    def __str__(self) -> str:
        """Return a string representation of the circuit."""
        components_str = " || ".join(str(c) for c in self.components)
        return f"{self.name}: {components_str}"

class CircuitBuilder:
    """Helper class for building circuits."""
    
    @staticmethod
    def series(*components: Union[CircuitComponent, Agent, BaseTool, Callable], 
              name: Optional[str] = None,
              max_queue_size: int = 0) -> SeriesCircuit:
        """
        Create a series circuit from components with task queueing.
        
        Args:
            *components: Components to include in the circuit
            name: Optional name for the circuit
            max_queue_size: Maximum size of the task queue (0 for unlimited)
            
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
        
        return SeriesCircuit(circuit_components, name=name, max_queue_size=max_queue_size)
    
    @staticmethod
    def parallel(*components: Union[CircuitComponent, Agent, BaseTool, Callable],
               name: Optional[str] = None,
               max_concurrency: int = 0) -> ParallelCircuit:
        """
        Create a parallel circuit from components with concurrency control.
        
        Args:
            *components: Components to include in the circuit
            name: Optional name for the circuit
            max_concurrency: Maximum number of concurrent tasks (0 for unlimited)
            
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
        
        return ParallelCircuit(circuit_components, name=name, max_concurrency=max_concurrency)

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
