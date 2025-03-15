# Pions

A lightweight, modular agent library for building AI agent systems with circuit-based workflows.

## Overview

Pions provides a simple yet powerful architecture for building agent-based systems using a circuit-based approach. It includes:

- Protocol definitions for agents and tools
- Base classes for extending agent and tool functionality
- A controller agent for orchestrating complex workflows
- Circuit components for visualizing and composing agent workflows
- Example implementations to get started quickly

## Core Components

### Circuits

Circuits are the fundamental building blocks in Pions. They allow for the composition of agents and tools into complex workflows.

```python
from pions.circuits import CircuitBuilder, CircuitVisualizer

# Create a simple weather information circuit
weather_circuit = CircuitBuilder.series(
    name="Weather Information Circuit",
    components=[
        FunctionComponent(entity_extractor),
        ToolComponent(WeatherTool()),
        FunctionComponent(result_formatter)
    ]
)

# Visualize the circuit
print(CircuitVisualizer.visualize_circuit(weather_circuit, "text"))
print(CircuitVisualizer.visualize_circuit(weather_circuit, "mermaid"))

# Process a query through the circuit
result = await weather_circuit.process({"query": "What's the weather in San Francisco?"})
```

### Agents

Agents are the primary actors in the system. They implement the `AgentProtocol` and can process queries to produce results.

```python
from pions import Agent

class MyCustomAgent(Agent):
    def __init__(self):
        super().__init__(name="my_custom_agent")
    
    async def process(self, query: str, **kwargs):
        # Implement your agent's logic here
        return {"result": f"Processed: {query}"}
```

### Tools

Tools are utilities that agents can use to perform specific tasks. They implement the `ToolProtocol`.

```python
from pions import BaseTool

class MyCustomTool(BaseTool):
    def __init__(self):
        super().__init__(name="my_custom_tool")
    
    async def execute(self, *args, **kwargs):
        # Implement your tool's logic here
        return {"result": "Tool executed"}
```

### Controller

The `ControllerAgent` orchestrates multiple agents and tools, facilitating complex workflows. For more advanced workflows, consider using the circuit-based approach.

```python
from pions import ControllerAgent

# Create controller
controller = ControllerAgent(name="my_controller")

# Register agents and tools
controller.register_agent(my_agent)
controller.register_tool(my_tool)

# Process queries
result = await controller.process("My query", agent_name="my_agent")

# Execute tools directly
tool_result = await controller.execute_tool("my_tool", arg1, arg2)

# Run pipelines
pipeline_result = await controller.run_pipeline(
    "My pipeline query",
    ["agent1", "agent2", "tool1"]
)
```

## Examples

The library includes example implementations in the `examples.py` and `circuit_examples.py` modules:

```python
# Traditional agent example
from pions.examples import ResearchAgent, ContentAnalysisAgent

# Create and use example agents
research_agent = ResearchAgent()
result = await research_agent.process("How does AI work?")

# Circuit-based example
from pions.circuit_examples import create_weather_circuit

# Create a weather information circuit
weather_circuit = create_weather_circuit()

# Process a query through the circuit
result = await weather_circuit.process({"query": "What's the weather in San Francisco?"})
print(result.get("formatted_result"))
```

## Installation

```bash
pip install pions
```

Or from source:

```bash
git clone https://github.com/janhq/asimov.git
cd asimov
pip install -e .
```

## Dependencies

Pions requires:
- Python 3.8+
- bhumi (for LLM inference)
- numpy (for certain features)
- typing-extensions

## Using with Bhumi

Pions integrates seamlessly with Bhumi for LLM inference:

```python
from bhumi.base_client import BaseLLMClient, LLMConfig
from pions import Agent

class LLMAgent(Agent):
    def __init__(self, name: str, llm_config: dict):
        super().__init__(name=name)
        self.llm_client = BaseLLMClient(LLMConfig(**llm_config))
        
    async def process(self, query: str, **kwargs):
        response = await self.llm_client.generate(query)
        return {"result": response}

# Usage
llm_config = {
    "api_key": "your_api_key",
    "model": "gemini/gemini-1.5-flash-latest",
    "debug": False
}

agent = LLMAgent("my_llm_agent", llm_config)
result = await agent.process("Tell me about quantum computing")
```

## License

MIT License
