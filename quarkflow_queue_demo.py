#!/usr/bin/env python3
"""
Quarkflow Queue Demo Script

This script demonstrates how to use the new queueing features in quarkflow circuit components 
for creating more efficient agent workflows with controlled concurrency.
"""

import asyncio
import os
import json
import time
import random
from typing import Dict, Any, List

# Try to load environment variables if dotenv is installed
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("Loaded environment variables from .env file")
except ImportError:
    print("python-dotenv not installed, using environment variables directly")

# Import the quarkflow library components
import quarkflow
from quarkflow import Agent, BaseTool, CircuitBuilder, CircuitVisualizer

# Create simulated components that have variable processing times

class SlowTool(BaseTool):
    """Tool that simulates a slow API call or processing task."""
    
    def __init__(self, name="slow_tool", min_delay=1, max_delay=5):
        """Initialize the slow tool with configurable delay."""
        super().__init__(name=name)
        self.min_delay = min_delay
        self.max_delay = max_delay
    
    async def execute(self, query: str, **kwargs) -> dict:
        """Execute the tool with a random delay."""
        delay = random.uniform(self.min_delay, self.max_delay)
        
        print(f"[{self.name}] Starting task with {delay:.2f}s delay")
        await asyncio.sleep(delay)
        print(f"[{self.name}] Completed task after {delay:.2f}s")
        
        return {
            "tool": self.name,
            "delay": delay,
            "query": query,
            "result": f"Processed '{query}' after {delay:.2f}s"
        }

class SlowAgent(Agent):
    """Agent that simulates a slow LLM or processing task."""
    
    def __init__(self, name="slow_agent", min_delay=2, max_delay=8):
        """Initialize the slow agent with configurable delay."""
        super().__init__(name=name)
        self.min_delay = min_delay
        self.max_delay = max_delay
    
    async def process(self, query: str, **kwargs) -> dict:
        """Process the query with a random delay."""
        delay = random.uniform(self.min_delay, self.max_delay)
        
        print(f"[{self.name}] Starting task with {delay:.2f}s delay")
        await asyncio.sleep(delay)
        print(f"[{self.name}] Completed task after {delay:.2f}s")
        
        return {
            "agent": self.name,
            "delay": delay,
            "query": query,
            "response": f"Processed '{query}' after {delay:.2f}s"
        }

async def transform_data(query: str, **kwargs) -> dict:
    """Function component with a simulated delay."""
    delay = random.uniform(0.5, 2.0)
    
    print(f"[transform_data] Starting task with {delay:.2f}s delay")
    await asyncio.sleep(delay)
    print(f"[transform_data] Completed task after {delay:.2f}s")
    
    # Extract data from kwargs if available
    tool_result = kwargs.get("tool_result", {})
    agent_result = kwargs.get("agent_result", {})
    
    return {
        "function": "transform_data",
        "delay": delay,
        "transformed": f"Transformed data for '{query}' after {delay:.2f}s",
        "original_query": query
    }

async def demo_series_queue():
    """Demonstrate series circuit with task queueing."""
    print("\n===== Series Circuit with Queueing =====")
    
    # Create a series circuit with queueing
    components = [
        SlowTool(name="tool_a", min_delay=1, max_delay=3),
        SlowAgent(name="agent_a", min_delay=2, max_delay=4),
        transform_data
    ]
    
    # Create a series circuit with a queue size of 5
    series_circuit = CircuitBuilder.series(
        *components,
        name="Queued Series Pipeline",
        max_queue_size=5  # Limit queue size to 5 tasks
    )
    
    # Visualize the circuit
    print(CircuitVisualizer.visualize_circuit(series_circuit, "text"))
    
    # Create multiple tasks to process through the series circuit
    tasks = []
    for i in range(8):
        query = f"Task {i+1}"
        print(f"Submitting query: {query}")
        
        # Add a small delay between submissions to show queuing behavior
        await asyncio.sleep(0.5)
        
        # Submit the task to the circuit without awaiting it
        task = asyncio.create_task(series_circuit.process({"query": query}))
        tasks.append(task)
    
    # Wait for all tasks to complete
    results = await asyncio.gather(*tasks)
    
    # Print results
    print("\nResults from series circuit with queueing:")
    for i, result in enumerate(results):
        print(f"Task {i+1}: {json.dumps(result, indent=2)}")

async def demo_parallel_queue():
    """Demonstrate parallel circuit with concurrency control."""
    print("\n===== Parallel Circuit with Concurrency Control =====")
    
    # Create multiple similar components to demonstrate parallel processing
    components = [
        SlowTool(name=f"tool_{i}", min_delay=1, max_delay=5) 
        for i in range(10)  # Create 10 tools
    ]
    
    # Create a parallel circuit with concurrency limit
    parallel_circuit = CircuitBuilder.parallel(
        *components,
        name="Controlled Parallel Pipeline",
        max_concurrency=3  # Only run 3 components at a time
    )
    
    # Visualize the circuit
    print(CircuitVisualizer.visualize_circuit(parallel_circuit, "text"))
    
    # Process a single query through the parallel circuit
    start_time = time.time()
    result = await parallel_circuit.process({"query": "Parallel processing with concurrency control"})
    total_time = time.time() - start_time
    
    # Print results
    print(f"\nParallel circuit completed in {total_time:.2f}s with concurrency limit of 3")
    
    # Extract and sort components by completion time
    component_results = []
    for key, value in result.items():
        if key.startswith("tool_") and isinstance(value, dict) and "delay" in value:
            component_results.append({
                "component": key,
                "delay": value["delay"]
            })
    
    # Sort by delay and print
    component_results.sort(key=lambda x: x["delay"])
    print("\nComponent execution times (shortest to longest):")
    for comp in component_results:
        print(f"{comp['component']}: {comp['delay']:.2f}s")

async def main():
    """Run both demo functions."""
    print("=== Quarkflow Queueing Demo ===")
    
    await demo_series_queue()
    await demo_parallel_queue()

if __name__ == "__main__":
    asyncio.run(main())
