# Quarkflow

<p align="center">
  <img src="https://img.shields.io/badge/version-0.1.0-blue.svg" alt="Version">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/python-3.8+-orange.svg" alt="Python">
</p>

## A Lightweight Library for Creating and Managing AI Agent Circuits

Quarkflow is a flexible, circuit-based AI agent orchestration library that allows you to build, connect, and deploy sophisticated AI agent workflows. Inspired by fundamental particles, Quarkflow provides the building blocks for creating complex agent interactions with minimal overhead.

## Features

- **Circuit-based architecture** - Create reusable components that can be connected in series or parallel
- **Flexible agent composition** - Combine different agent types and tools into unified workflows
- **Built-in visualization** - Visualize your agent circuits for easier debugging and sharing
- **Lightweight implementation** - Focus on essential functionality with minimal dependencies
- **Compatible with major LLM providers** - Works with bhumi for seamless integration with various LLM backends

## Installation

```bash
pip install quarkflow
```

## Quick Start

### Basic Agent Example

```python
from quarkflow import Agent, BaseTool, ControllerAgent
from bhumi.base_client import BaseLLMClient, LLMConfig

# Create a simple agent
class ResearchAgent(Agent):
    def __init__(self, llm_config=None):
        super().__init__(name="research_agent", llm_config=llm_config)
    
    async def process(self, query, **kwargs):
        # Agent processing logic here
        return {"result": f"Research results for: {query}"}

# Create a tool
class SearchTool(BaseTool):
    def __init__(self, api_key=None):
        super().__init__(name="search_tool")
        self.api_key = api_key
    
    async def execute(self, query, **kwargs):
        # Tool execution logic here
        return {"results": [f"Result 1 for {query}", f"Result 2 for {query}"]}

# Use the agent
async def main():
    # Configure LLM
    llm_config = LLMConfig(api_key="YOUR_API_KEY", model="MODEL_NAME")
    
    # Initialize agent
    agent = ResearchAgent(llm_config=llm_config)
    
    # Process a query
    result = await agent.process("quantum computing applications")
    print(result)
```

### Building Circuits

```python
from quarkflow import CircuitBuilder, CircuitVisualizer

# Create circuit components
research_agent = ResearchAgent(llm_config=llm_config)
search_tool = SearchTool(api_key="SEARCH_API_KEY")

# Build a circuit
builder = CircuitBuilder()
circuit = builder.series(
    search_tool,
    research_agent
)

# Execute the circuit
result = await circuit.execute("renewable energy developments")

# Visualize the circuit
visualizer = CircuitVisualizer()
visualizer.visualize(circuit, "my_circuit.png")
```

## Advanced Usage

Quarkflow supports complex circuit compositions with branches, parallel processing, and conditional flows:

```python
# Create a parallel circuit that processes data through multiple agents simultaneously
parallel_circuit = builder.parallel(
    research_agent,
    analysis_agent
)

# Create a more complex workflow with series and parallel components
complex_circuit = builder.series(
    search_tool,
    builder.parallel(
        research_agent,
        analysis_agent
    ),
    summarization_agent
)
```

## Examples

See the `quarkflow_demo.py` and `quarkflow_circuit_demo.py` files for full working examples.

## Deployment

Use the provided `deploy.sh` script to easily deploy your package:

```bash
# Basic deployment (build and install locally)
./deploy.sh

# Deploy to test PyPI
./deploy.sh --pypi

# Deploy to production PyPI
./deploy.sh --pypi --prod

# Create a Git tag and push it
./deploy.sh --tag
```

## Documentation

For more detailed documentation, visit our [GitHub repository](https://github.com/janhq/pions).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

* Developed by Menlo Deep Labs
* Powered by [bhumi](https://github.com/janhq/bhumi) for LLM operations