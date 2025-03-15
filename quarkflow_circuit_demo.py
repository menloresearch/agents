#!/usr/bin/env python3
"""
Quarkflow Circuit Demo Script

This script demonstrates how to use the quarkflow circuit components to create
and visualize agent workflows in series and parallel configurations.
Quarkflow is a standalone library that uses bhumi for LLM operations.
"""

import asyncio
import os
import json

# Try to load environment variables if dotenv is installed
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("Loaded environment variables from .env file")
except ImportError:
    print("python-dotenv not installed, using environment variables directly")

# Import the bhumi client for LLM inference
from bhumi.base_client import BaseLLMClient, LLMConfig

# Import the quarkflow library components
import quarkflow
from quarkflow import Agent, BaseTool, CircuitBuilder, CircuitVisualizer

# Import example agents and tools from quarkflow.examples
try:
    from quarkflow.examples import ResearchAgent, ContentAnalysisAgent, SimpleSearchTool, SummarizationTool
except ImportError as e:
    print(f"Warning: Could not import example components: {e}")
    print("Creating simplified mock components for demonstration")
    
    # Create mock components if the examples are not available
    class ResearchAgent(Agent):
        """Mock research agent."""
        def __init__(self, search_api_key=None, llm_config=None):
            super().__init__(name="research_agent", llm_config=llm_config)
            
        async def process(self, query, **kwargs):
            return {"summary": f"Research results for: {query}", "source": self.name}
    
    class ContentAnalysisAgent(Agent):
        """Mock content analysis agent."""
        def __init__(self, llm_config=None):
            super().__init__(name="content_agent", llm_config=llm_config)
            
        async def process(self, query, **kwargs):
            return {"sentiment": "positive", "summary": f"Analysis of: {query}", "source": self.name}
    
    class SimpleSearchTool(BaseTool):
        """Mock search tool."""
        def __init__(self, api_key=None):
            super().__init__(name="search_tool")
            
        async def execute(self, query, **kwargs):
            return {"search_results": [{"title": f"Result for {query}", "url": "https://example.com"}]}
    
    class SummarizationTool(BaseTool):
        """Mock summarization tool."""
        def __init__(self):
            super().__init__(name="summarize_tool")
            
        async def execute(self, query, **kwargs):
            return {"summary": f"Summary of: {query}"}

# Create a custom transformer function component
async def keyword_extractor(query: str, search_results: list = None, **kwargs):
    """Extract keywords from search results."""
    # Simple implementation - in a real scenario, this would be more sophisticated
    keywords = set()
    
    if search_results:
        for result in search_results:
            # Extract words from title
            title = result.get('title', '')
            words = [w.lower() for w in title.split() if len(w) > 4]
            keywords.update(words)
    
    return {
        "query": query,
        "keywords": list(keywords)[:5],  # Limit to 5 keywords
        "source": "keyword_extractor"
    }

# Create a filter function component
async def content_filter(query: str, analysis: dict = None, **kwargs):
    """Filter content based on sentiment."""
    if not analysis:
        return {"filtered": False, "reason": "No analysis provided"}
    
    sentiment = analysis.get('sentiment', 'neutral')
    
    # Demo filter logic
    if sentiment == 'negative':
        return {
            "filtered": True,
            "reason": "Negative sentiment detected",
            "query": query
        }
    
    return {
        "filtered": False,
        "reason": "Content passed filter",
        "query": query
    }

class MetadataExtractor(BaseTool):
    """Tool for extracting metadata from content."""
    
    def __init__(self):
        """Initialize the metadata extractor tool."""
        super().__init__(name="metadata_extractor")
    
    async def execute(self, query: str, analysis: dict = None, **kwargs) -> dict:
        """Extract metadata from analysis."""
        metadata = {
            "query": query,
            "timestamp": asyncio.get_event_loop().time(),
            "word_count": analysis.get("word_count", 0) if analysis else 0,
            "has_summary": bool(analysis and "summary" in analysis)
        }
        
        return metadata

async def visualize_circuits():
    """Create and visualize different circuit configurations."""
    # Get API keys from environment variables
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    search_api_key = os.getenv("BRAVE_API_KEY")
    
    # Configure LLM if API key is available
    llm_config = None
    if gemini_api_key:
        print("Using Gemini for LLM inference through bhumi")
        llm_config = {
            "api_key": gemini_api_key,
            "model": "gemini/gemini-1.5-flash-latest",
            "debug": False
        }
    else:
        print("No Gemini API key found in environment variables")
        print("Set GEMINI_API_KEY to enable LLM inference with bhumi")
        print("Using mock responses for demonstration purposes")
    
    # Create agents and tools
    research_agent = ResearchAgent(search_api_key=search_api_key, llm_config=llm_config)
    content_agent = ContentAnalysisAgent(llm_config=llm_config)
    search_tool = SimpleSearchTool(api_key=search_api_key)
    summarize_tool = SummarizationTool()
    metadata_tool = MetadataExtractor()
    
    print("\n===== Simple Series Circuit =====")
    # Create a simple series circuit: search → research → content analysis
    simple_series = CircuitBuilder.series(
        search_tool,
        research_agent,
        content_agent,
        name="Research Pipeline"
    )
    
    # Visualize the circuit
    print(CircuitVisualizer.visualize_circuit(simple_series, "text"))
    print(CircuitVisualizer.visualize_circuit(simple_series, "mermaid"))
    
    # Execute the circuit
    result = await simple_series.process({"query": "The impact of quantum computing on cryptography"})
    print(f"Circuit trace: {json.dumps(result.get('circuit_trace', []), indent=2)}")
    print(f"Summary: {result.get('summary', 'No summary available')}")
    
    print("\n===== Parallel Circuit =====")
    # Create a parallel circuit: run research and content analysis in parallel
    parallel_circuit = CircuitBuilder.parallel(
        research_agent,
        content_agent,
        name="Parallel Analysis"
    )
    
    # Visualize the circuit
    print(CircuitVisualizer.visualize_circuit(parallel_circuit, "text"))
    print(CircuitVisualizer.visualize_circuit(parallel_circuit, "mermaid"))
    
    # Execute the circuit
    result = await parallel_circuit.process({"query": "The history of artificial intelligence"})
    print(f"Circuit trace: {json.dumps(result.get('circuit_trace', []), indent=2)}")
    
    print("\n===== Complex Circuit =====")
    # Create a more complex circuit with nested series and parallel components
    # First create a parallel branch for content analysis and metadata extraction
    analysis_branch = CircuitBuilder.parallel(
        content_agent,
        metadata_tool,
        name="Content Processing"
    )
    
    # Create a function component for keyword extraction
    keyword_component = CircuitBuilder.series(
        keyword_extractor,
        name="Keyword Extraction"
    )
    
    # Create the main pipeline that branches into parallel paths
    complex_circuit = CircuitBuilder.series(
        search_tool,
        research_agent,
        analysis_branch,
        keyword_component,
        content_filter,
        name="Advanced Research Pipeline"
    )
    
    # Visualize the complex circuit
    print(CircuitVisualizer.visualize_circuit(complex_circuit, "text"))
    print(CircuitVisualizer.visualize_circuit(complex_circuit, "mermaid"))
    
    # Execute the complex circuit
    result = await complex_circuit.process({"query": "The applications of machine learning in healthcare"})
    print(f"Circuit trace: {json.dumps(result.get('circuit_trace', []), indent=2)}")
    print(f"Keywords: {result.get('keywords', [])}")
    print(f"Filtered: {result.get('filtered', False)}")

async def main():
    """Run the quarkflow circuit demo."""
    print("===== Quarkflow Circuit Demo =====")
    print("Demonstrating quarkflow as a standalone library using bhumi for LLM operations")
    print(f"Quarkflow version: {quarkflow.__version__}")
    
    await visualize_circuits()
    
    print("\n===== Demo Complete =====")
    print("For more information, visit: https://github.com/janhq/pions")

if __name__ == "__main__":
    asyncio.run(main())
