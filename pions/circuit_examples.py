"""
Pions Circuit Examples

This file showcases various examples of using the circuit-based architecture
to build different agent workflows.
"""

import asyncio
import os
import json
from typing import Dict, Any, List
from datetime import datetime

# Import the pions components
from pions import Agent, BaseTool
from pions import CircuitBuilder, CircuitVisualizer
from pions.circuits import FunctionComponent, AgentComponent, ToolComponent, SeriesCircuit, ParallelCircuit
from bhumi.base_client import BaseLLMClient, LLMConfig

# Define some example agents and tools for demonstration

class WeatherTool(BaseTool):
    """Tool for getting weather information."""
    
    def __init__(self):
        """Initialize the weather tool."""
        super().__init__(name="weather_tool")
    
    async def execute(self, query: str, location: str = None, entities: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        """Get weather information for a location."""
        # In a real implementation, this would call a weather API
        # For this example, we'll use mock data
        
        # First try to use provided location
        if not location and entities and entities.get("locations"):
            location = entities["locations"][0]
        
        # If still no location, try to extract from query
        if not location:
            location_markers = ["weather in", "temperature in", "forecast for"]
            for marker in location_markers:
                if marker in query.lower():
                    location = query.lower().split(marker)[1].strip()
                    break
        
        # Default to Unknown if no location found
        if not location:
            location = "Unknown"
        
        print(f"Using location: {location}")
        
        # Generate mock weather data
        weather_data = {
            "location": location,
            "temperature": 72,
            "condition": "Sunny",
            "humidity": 45,
            "wind_speed": 5,
            "timestamp": datetime.now().isoformat()
        }
        
        return {
            "query": query,
            "weather": weather_data,
            "location": location  # Explicitly include location for downstream components
        }

class NewsSearchTool(BaseTool):
    """Tool for searching news articles."""
    
    def __init__(self):
        """Initialize the news search tool."""
        super().__init__(name="news_search")
    
    async def execute(self, query: str, max_results: int = 3, **kwargs) -> Dict[str, Any]:
        """Search for news articles related to the query."""
        # In a real implementation, this would call a news API
        # For this example, we'll use mock data
        
        # Simple keyword extraction
        keywords = [word for word in query.lower().split() if len(word) > 3]
        
        # Make sure we have at least one keyword
        if not keywords and len(query.split()) > 0:
            keywords = [query.split()[0]]
        elif not keywords:
            keywords = ["news"]
        
        print(f"News search keywords: {keywords}")
        
        # Generate mock results based on extracted keywords
        results = []
        for i in range(min(max_results, 3)):
            kw = keywords[i % len(keywords)] if keywords else "news"
            results.append({
                "title": f"Latest developments in {kw}",
                "source": "Example News",
                "date": datetime.now().isoformat(),
                "snippet": f"This is a mock news article about {kw}. "
                          f"It contains information relevant to the search query '{query}'."
            })
        
        return {
            "query": query,
            "news_results": results
        }

class FactCheckerAgent(Agent):
    """Agent for fact-checking information."""
    
    def __init__(self, llm_config: Dict[str, Any] = None):
        """Initialize the fact checker agent."""
        super().__init__(name="fact_checker", llm_config=llm_config)
    
    async def process(self, query: str, news_results: List[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        """Process the query and check facts against news results."""
        # In a production system, this would use the LLM to analyze and fact-check
        # For this example, we'll return mock results
        
        if self.llm_client:
            # If we have an LLM client, we could use it for real analysis
            # For now, we'll just log that we're using it
            print(f"Using LLM for fact checking on: {query}")
        
        if not news_results:
            return {
                "query": query,
                "fact_check": {
                    "verified": False,
                    "reason": "No news results available to verify against"
                }
            }
        
        # Simple mock fact-checking process
        fact_check = {
            "verified": len(news_results) > 1,  # More than one source is a simple heuristic
            "confidence": min(len(news_results) * 0.3, 0.9),  # Simple confidence score
            "sources": [result["source"] for result in news_results],
            "verification_time": datetime.now().isoformat()
        }
        
        return {
            "query": query,
            "fact_check": fact_check,
            "source_count": len(news_results)
        }

class SummarizerAgent(Agent):
    """Agent for summarizing content."""
    
    def __init__(self, llm_config: Dict[str, Any] = None):
        """Initialize the summarizer agent."""
        super().__init__(name="summarizer", llm_config=llm_config)
    
    async def process(self, query: str, weather: Dict[str, Any] = None, 
                     news_results: List[Dict[str, Any]] = None,
                     fact_check: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        """Summarize the information gathered from various sources."""
        # In a production system, this would use the LLM for a sophisticated summary
        # For this example, we'll create a simple template-based summary
        
        if self.llm_client:
            print(f"Using LLM for summarization on: {query}")
        
        summary_parts = []
        
        # Include weather if available
        if weather:
            summary_parts.append(
                f"Weather in {weather.get('location', 'Unknown')}: "
                f"{weather.get('temperature', 'N/A')}°F, {weather.get('condition', 'Unknown')}. "
                f"Humidity: {weather.get('humidity', 'N/A')}%, Wind: {weather.get('wind_speed', 'N/A')} mph."
            )
        
        # Include news summary if available
        if news_results:
            sources = ", ".join(set(r.get("source", "Unknown") for r in news_results))
            summary_parts.append(
                f"Found {len(news_results)} relevant news articles from {sources}. "
                f"Focusing on {', '.join(r.get('title', 'Unknown')[:30] + '...' for r in news_results[:2])}."
            )
        
        # Include fact check results if available
        if fact_check:
            verified = fact_check.get("verified", False)
            confidence = fact_check.get("confidence", 0)
            summary_parts.append(
                f"Fact check: {'Verified' if verified else 'Unverified'} "
                f"with {int(confidence * 100)}% confidence."
            )
        
        # Build the final summary
        if summary_parts:
            summary = " ".join(summary_parts)
        else:
            summary = f"No information found for query: {query}"
        
        return {
            "query": query,
            "summary": summary
        }

# Define transformation functions for the circuit

async def entity_extractor(query: str, **kwargs) -> Dict[str, Any]:
    """Extract entities from the query."""
    # In a production system, this would use NLP techniques
    # For this example, we'll use a simple approach
    
    entities = {
        "locations": [],
        "people": [],
        "organizations": [],
        "dates": [],
        "keywords": []
    }
    
    # Simple extraction of locations
    location_markers = ["in", "at", "near", "around", "for"]
    for marker in location_markers:
        if f" {marker} " in f" {query.lower()} ":
            parts = query.lower().split(f" {marker} ")
            if len(parts) > 1:
                # Take everything until the next preposition or end of sentence
                location_part = parts[1].strip()
                end_markers = [" in ", " at ", " and ", " or ", ".", "?", "!", ","]
                for end in end_markers:
                    if end in location_part:
                        location_part = location_part.split(end)[0].strip()
                
                # Clean up any punctuation at the end
                location_part = location_part.rstrip(",.?!:;")
                
                if location_part:
                    entities["locations"].append(location_part)
    
    # Extract keywords (simple approach: words longer than 5 chars)
    entities["keywords"] = [word for word in query.split() 
                           if len(word) > 5 and word.lower() not in ["weather", "news", "about", "information"]]
    
    print(f"Extracted entities: {entities}")
    return {
        "query": query,
        "entities": entities,
        "location": entities["locations"][0] if entities["locations"] else None
    }

async def query_classifier(query: str, **kwargs) -> Dict[str, Any]:
    """Classify the type of query."""
    query_lower = query.lower()
    
    # Simple classification based on keywords
    categories = []
    
    if any(term in query_lower for term in ["weather", "temperature", "rain", "forecast", "sunny", "cloudy"]):
        categories.append("weather")
    
    if any(term in query_lower for term in ["news", "article", "headline", "report", "story", "latest"]):
        categories.append("news")
    
    if any(term in query_lower for term in ["true", "false", "fact", "check", "verify", "accuracy"]):
        categories.append("fact_check")
    
    if not categories:
        categories.append("general")
    
    # For debugging
    print(f"Classified query: '{query}' as {categories}")
    
    return {
        "query": query,
        "query_type": categories,
        "confidence": 0.8,  # Mock confidence
        **kwargs  # Pass through any existing arguments
    }

async def result_formatter(query: str, summary: str = None, **kwargs) -> Dict[str, Any]:
    """Format the final results in a presentable way."""
    # If no summary but we have other data, create a basic summary
    if not summary:
        if "weather" in kwargs:
            location = kwargs["weather"].get("location", "Unknown")
            condition = kwargs["weather"].get("condition", "Unknown")
            temp = kwargs["weather"].get("temperature", "N/A")
            summary = f"Weather in {location}: {temp}°F, {condition}"
        elif "news_results" in kwargs:
            summary = f"Found {len(kwargs['news_results'])} news articles related to your query."
        else:
            return {
                "query": query,
                "formatted_result": f"No information found for: {query}"
            }
    
    # Create a formatted result with some markdown styling
    formatted_result = f"""
# Results for: "{query}"

## Summary
{summary}

## Details
"""
    
    # Add weather details if available
    if "weather" in kwargs:
        weather = kwargs["weather"]
        formatted_result += f"""
### Weather
- Location: {weather.get('location', 'Unknown')}
- Temperature: {weather.get('temperature', 'N/A')}°F
- Condition: {weather.get('condition', 'Unknown')}
- Humidity: {weather.get('humidity', 'N/A')}%
- Wind Speed: {weather.get('wind_speed', 'N/A')} mph
"""
    
    # Add news details if available
    if "news_results" in kwargs:
        news_results = kwargs["news_results"]
        formatted_result += f"""
### News Articles
Found {len(news_results)} relevant articles:
"""
        for i, article in enumerate(news_results, 1):
            formatted_result += f"""
**{i}. {article.get('title', 'Unknown')}**
Source: {article.get('source', 'Unknown')}
Date: {article.get('date', 'Unknown')}
Snippet: {article.get('snippet', 'No snippet available')}
"""
    
    # Add fact check details if available
    if "fact_check" in kwargs:
        fact_check = kwargs["fact_check"]
        formatted_result += f"""
### Fact Check
- Verified: {'Yes' if fact_check.get('verified', False) else 'No'}
- Confidence: {int(fact_check.get('confidence', 0) * 100)}%
- Sources: {', '.join(fact_check.get('sources', ['None']))}
"""
    
    return {
        "query": query,
        "formatted_result": formatted_result
    }

# Example 1: Simple Weather Circuit
def create_weather_circuit():
    """Create a simple weather information circuit."""
    weather_tool = WeatherTool()
    
    # Create a circuit that extracts entities, gets weather, and formats the result
    weather_circuit = CircuitBuilder.series(
        entity_extractor,
        weather_tool,
        result_formatter,
        name="Weather Information Circuit"
    )
    
    return weather_circuit

# Example 2: News + Fact Checking Circuit
def create_news_circuit(llm_config=None):
    """Create a news search and fact checking circuit."""
    news_tool = NewsSearchTool()
    fact_checker = FactCheckerAgent(llm_config=llm_config)
    
    # Create a circuit that searches for news and fact-checks it
    news_circuit = CircuitBuilder.series(
        news_tool,
        fact_checker,
        result_formatter,
        name="News Verification Circuit"
    )
    
    return news_circuit

# Example 3: Complex Information Pipeline
def create_info_pipeline(llm_config=None):
    """Create a complex information gathering pipeline."""
    # Create tools and agents
    weather_tool = WeatherTool()
    news_tool = NewsSearchTool()
    fact_checker = FactCheckerAgent(llm_config=llm_config)
    summarizer = SummarizerAgent(llm_config=llm_config)
    
    # Create a parallel circuit for data gathering
    data_gathering = CircuitBuilder.parallel(
        weather_tool,
        news_tool,
        name="Data Gathering"
    )
    
    # Create the main information pipeline
    pipeline = CircuitBuilder.series(
        query_classifier,         # First classify the query
        entity_extractor,         # Extract entities from the query
        data_gathering,           # Gather data in parallel
        fact_checker,             # Check facts based on news results
        summarizer,               # Summarize all information
        result_formatter,         # Format the results
        name="Information Pipeline"
    )
    
    return pipeline

# Example 4: Dynamic Routing Circuit
def create_dynamic_routing_circuit(llm_config=None):
    """Create a circuit with dynamic routing based on query type."""
    # Create the specialized circuits
    weather_circuit = create_weather_circuit()
    news_circuit = create_news_circuit(llm_config)
    
    # Create a router function
    async def query_router(query: str, query_type: List[str] = None, **kwargs):
        """Route the query to the appropriate circuit based on its type."""
        # Default to general if no type is specified
        if not query_type:
            query_type = ["general"]
        
        route_to = query_type[0] if query_type else "general"
        print(f"Routing query '{query}' with type: {query_type} to '{route_to}'")
        
        # Return routing information
        return {
            "query": query,
            "route_to": route_to,
            "all_types": query_type,
            **kwargs  # Pass through all other data
        }
    
    # Create branches for different query types
    async def route_executor(query: str, route_to: str = "general", **kwargs):
        """Execute the appropriate circuit based on routing."""
        # Clean the input for the circuit
        circuit_input = {"query": query, **kwargs}
        if "route_to" in circuit_input:
            del circuit_input["route_to"]
        if "all_types" in circuit_input:
            del circuit_input["all_types"]
            
        print(f"Executing circuit for route: {route_to}")
        
        # Choose the circuit based on the route
        if route_to == "weather":
            # Direct execution of the weather circuit
            circuit = create_weather_circuit()
            print("Selected weather circuit")
        elif route_to == "news" or route_to == "fact_check":
            # Direct execution of the news circuit
            circuit = create_news_circuit(llm_config)
            print("Selected news circuit")
        else:
            # Default to the full information pipeline
            circuit = create_info_pipeline(llm_config)
            print("Selected full information pipeline")
        
        # Execute the selected circuit and return its results
        result = await circuit.process(circuit_input)
        result["route_to"] = route_to  # Add the routing information
        return result
    
    # Create the main routing circuit
    routing_circuit = CircuitBuilder.series(
        query_classifier,
        query_router,
        FunctionComponent(route_executor),
        name="Dynamic Routing Circuit"
    )
    
    return routing_circuit

# Example functions to run the circuits
async def run_example_circuits():
    """Run examples of different circuit configurations."""
    # Check for API keys
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    
    # Configure LLM if API key is available
    llm_config = None
    if gemini_api_key:
        print("Using Gemini for LLM inference")
        llm_config = {
            "api_key": gemini_api_key,
            "model": "gemini/gemini-1.5-flash-latest",
            "debug": False
        }
    else:
        print("No LLM API key found, using mock responses")
    
    print("\n---------------------------------------------------")
    print("  Circuit Examples: Visualizing Agent Workflows")
    print("---------------------------------------------------")
    
    # Example 1: Simple Weather Circuit
    print("\n===== Example 1: Simple Weather Circuit =====")
    weather_circuit = create_weather_circuit()
    print(CircuitVisualizer.visualize_circuit(weather_circuit, "text"))
    print(CircuitVisualizer.visualize_circuit(weather_circuit, "mermaid"))
    
    print("\nProcessing: What's the weather in San Francisco?")
    weather_result = await weather_circuit.process({"query": "What's the weather in San Francisco?"})
    print("\nResults:")
    print(weather_result.get("formatted_result", "No results"))
    
    # Example 2: News and Fact Checking
    print("\n===== Example 2: News and Fact Checking Circuit =====")
    news_circuit = create_news_circuit(llm_config)
    print(CircuitVisualizer.visualize_circuit(news_circuit, "text"))
    
    print("\nProcessing: Latest news about artificial intelligence")
    news_result = await news_circuit.process({"query": "Latest news about artificial intelligence"})
    print("\nResults:")
    print(news_result.get("formatted_result", "No results"))
    
    # Example 3: Complex Information Pipeline
    print("\n===== Example 3: Complex Information Pipeline =====")
    info_pipeline = create_info_pipeline(llm_config)
    print(CircuitVisualizer.visualize_circuit(info_pipeline, "text"))
    
    print("\nProcessing: What's the weather in New York and the latest news about climate change?")
    pipeline_result = await info_pipeline.process({"query": "What's the weather in New York and the latest news about climate change?"})
    print("\nResults:")
    print(pipeline_result.get("formatted_result", "No results"))
    
    # Example 4: Dynamic Routing
    print("\n===== Example 4: Dynamic Routing Based on Query Type =====")
    routing_circuit = create_dynamic_routing_circuit(llm_config)
    print(CircuitVisualizer.visualize_circuit(routing_circuit, "text"))
    
    # Test with a weather query
    weather_query = "What's the weather forecast for London?"
    print(f"\nTesting with weather query: '{weather_query}'")
    weather_routing_result = await routing_circuit.process({"query": weather_query})
    print(f"Routed to: {weather_routing_result.get('route_to', 'unknown')}")
    print(f"Results:\n{weather_routing_result.get('formatted_result', 'No results')}")
    
    # Test with a news query
    news_query = "Latest news about space exploration"
    print(f"\nTesting with news query: '{news_query}'")
    news_routing_result = await routing_circuit.process({"query": news_query})
    print(f"Routed to: {news_routing_result.get('route_to', 'unknown')}")
    print(f"Results:\n{news_routing_result.get('formatted_result', 'No results')}")
    
    print("\n---------------------------------------------------")
    print("  Circuit Examples Complete")
    print("---------------------------------------------------")

if __name__ == "__main__":
    asyncio.run(run_example_circuits())
