"""
Pions Example Agents

This module provides example implementations of agents and tools using the pions library.
"""

import os
import json
import aiohttp
from typing import Dict, Any, List, Optional

from bhumi.base_client import BaseLLMClient, LLMConfig

from .agent import Agent
from .tools import BaseTool
from .controller import ControllerAgent

class SimpleSearchTool(BaseTool):
    """A simple search tool that uses a mock API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the search tool."""
        super().__init__(name="search")
        self.api_key = api_key
    
    async def execute(self, query: str, **kwargs) -> Dict[str, Any]:
        """Execute a search and return results."""
        # In a real implementation, this would call an actual search API
        # This is just a mock implementation for demonstration
        
        # If there's an API key and we have aiohttp, do a mock API call
        if self.api_key:
            try:
                async with aiohttp.ClientSession() as session:
                    # This is a mock URL - in a real implementation, use an actual search API
                    url = "https://example.com/api/search"
                    headers = {"Authorization": f"Bearer {self.api_key}"}
                    params = {"q": query}
                    
                    # For the example, we'll just return mock data instead of making an actual request
                    # async with session.get(url, headers=headers, params=params) as response:
                    #     data = await response.json()
                    
                    # Mock search results
                    data = {
                        "results": [
                            {"title": f"Result 1 for {query}", "url": "https://example.com/1"},
                            {"title": f"Result 2 for {query}", "url": "https://example.com/2"},
                            {"title": f"Result 3 for {query}", "url": "https://example.com/3"}
                        ]
                    }
                    
                    return {
                        "query": query,
                        "results": data["results"]
                    }
            except Exception as e:
                return {
                    "query": query,
                    "error": str(e),
                    "results": []
                }
        
        # Fallback to mock results if no API key or if there was an error
        return {
            "query": query,
            "results": [
                {"title": f"Mock Result 1 for {query}", "url": "https://example.com/mock1"},
                {"title": f"Mock Result 2 for {query}", "url": "https://example.com/mock2"}
            ]
        }

class ResearchAgent(Agent):
    """An agent for conducting research on a query."""
    
    def __init__(self, search_api_key: Optional[str] = None, llm_config: Optional[Dict[str, Any]] = None):
        """Initialize the research agent."""
        super().__init__(name="research", llm_config=llm_config)
        self.search_tool = SimpleSearchTool(api_key=search_api_key)
    
    async def process(self, query: str, **kwargs) -> Dict[str, Any]:
        """Process a research query by searching and analyzing results."""
        # Step 1: Execute the search tool to get results
        search_results = await self.search_tool.execute(query)
        
        # Step 2: Use LLM to analyze results if client is available
        if self.llm_client:
            # Prepare search results for analysis
            search_text = "\n\n".join([f"Title: {result.get('title', '')}\nURL: {result.get('url', '')}" 
                                      for result in search_results.get("results", [])])
            
            # Create prompt for analysis
            analysis_prompt = f"""Analyze the following search results for the query: '{query}'\n\n{search_text}
            
            Please provide:  
            1. A brief summary of the search results
            2. Key points from the search results
            3. An assessment of the quality of sources (high, medium, low)
            """
            
            try:
                # Use the LLM to generate analysis
                analysis_text = await self.generate(
                    prompt=analysis_prompt,
                    system_prompt="You are a research analyst. Analyze the search results and provide insights.",
                    temperature=0.3
                )
                
                # Extract structured information from analysis (in a real implementation, we might use a more robust approach)
                analysis = {
                    "summary": analysis_text.split("\n\n")[0] if "\n\n" in analysis_text else analysis_text,
                    "key_points": analysis_text.split("Key points:")[1].split("\n") if "Key points:" in analysis_text else 
                                 ["Point extracted from analysis"],
                    "source_quality": "high" if "high" in analysis_text.lower() else 
                                    "medium" if "medium" in analysis_text.lower() else "low"
                }
            except Exception as e:
                # Fallback to mock analysis in case of error
                analysis = {
                    "summary": f"Analysis of search results for '{query}'",
                    "key_points": [
                        "Failed to generate analysis using LLM: {str(e)}",
                        "Falling back to mock analysis",
                        "Consider checking LLM configuration"
                    ],
                    "source_quality": "unknown"
                }
        else:
            # Fallback to mock analysis if no LLM client
            analysis = {
                "summary": f"Analysis of search results for '{query}'",
                "key_points": [
                    "This is a mock analysis point 1",
                    "This is a mock analysis point 2",
                    "This is a mock analysis point 3"
                ],
                "source_quality": "high"  # mock quality assessment
            }
        
        # Return combined results
        return {
            "query": query,
            "search_results": search_results.get("results", []),
            "analysis": analysis
        }

class SummarizationTool(BaseTool):
    """A tool for summarizing text."""
    
    def __init__(self):
        """Initialize the summarization tool."""
        super().__init__(name="summarize")
    
    async def execute(self, text: str, **kwargs) -> Dict[str, Any]:
        """Summarize the given text."""
        # In a real implementation, this would use an LLM or other summarization method
        # This is just a mock implementation for demonstration
        
        # Mock implementation - just return first and last sentences
        sentences = text.split('.')
        if len(sentences) > 2:
            summary = f"{sentences[0].strip()}. ... {sentences[-2].strip()}."
        else:
            summary = text
        
        return {
            "original_length": len(text),
            "summary_length": len(summary),
            "summary": summary
        }

class ContentAnalysisAgent(Agent):
    """An agent for analyzing content and providing summaries."""
    
    def __init__(self, llm_config: Optional[Dict[str, Any]] = None):
        """Initialize the content analysis agent."""
        super().__init__(name="content_analysis", llm_config=llm_config)
        self.summarization_tool = SummarizationTool()
    
    async def process(self, content: str, **kwargs) -> Dict[str, Any]:
        """Analyze content and provide insights."""
        # Get a summary - first try using the LLM if available
        if self.llm_client:
            try:
                # Use LLM for summarization
                summary_prompt = f"""Summarize the following content concisely:\n\n{content}"""
                
                summary = await self.generate(
                    prompt=summary_prompt,
                    system_prompt="You are a content summarizer. Provide a concise summary of the content.",
                    temperature=0.3
                )
                
                # Use LLM for sentiment and keyword analysis
                analysis_prompt = f"""Analyze the following content:\n\n{content}\n\n
                Please provide:  
                1. Sentiment (positive, negative, or neutral)
                2. Key keywords (up to 5)
                """
                
                analysis_text = await self.generate(
                    prompt=analysis_prompt,
                    system_prompt="You are a content analyst. Extract information from the content.",
                    temperature=0.3
                )
                
                # Extract sentiment and keywords
                sentiment = "positive" if "positive" in analysis_text.lower() else \
                           "negative" if "negative" in analysis_text.lower() else "neutral"
                
                # Extract keywords (simplistic approach - in a real implementation we'd use a more robust method)
                keywords_section = analysis_text.split("keywords:")[1] if "keywords:" in analysis_text.lower() else ""
                keywords = [k.strip() for k in keywords_section.split(",") if k.strip()] if keywords_section else \
                          [k.strip() for k in keywords_section.split("\n") if k.strip()]
                
                if not keywords:
                    keywords = ["key1", "key2", "key3"]  # fallback
            except Exception as e:
                # Fall back to the tool if LLM fails
                summary_result = await self.summarization_tool.execute(content)
                summary = summary_result["summary"]
                word_count = len(content.split())
                sentiment = "positive" if "good" in content.lower() else "neutral"
                keywords = ["key1", "key2", "key3", f"error: {str(e)}"]  # indicate error
        else:
            # Fall back to the tool if no LLM
            summary_result = await self.summarization_tool.execute(content)
            summary = summary_result["summary"]
            word_count = len(content.split())
            sentiment = "positive" if "good" in content.lower() else "neutral"
            keywords = ["key1", "key2", "key3"]  # mock keywords
        
        # Return analysis results
        return {
            "word_count": len(content.split()),
            "summary": summary,
            "sentiment": sentiment,
            "keywords": keywords
        }

# Example of usage
async def example_usage():
    """Example of how to use the pions library."""
    # Get API key from environment or use a default for examples
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    
    # Configure LLM if API key is available
    llm_config = None
    if gemini_api_key:
        llm_config = {
            "api_key": gemini_api_key,
            "model": "gemini/gemini-1.5-flash-latest",  # Use a smaller model for faster response
            "debug": False
        }
    
    # Create a controller agent
    controller = ControllerAgent(name="example_controller", output_dir="./output")
    
    # Create and register specialized agents with LLM config
    research_agent = ResearchAgent(llm_config=llm_config)
    content_agent = ContentAnalysisAgent(llm_config=llm_config)
    
    controller.register_agent(research_agent)
    controller.register_agent(content_agent)
    
    # Register tools directly with the controller
    controller.register_tool(SimpleSearchTool())
    controller.register_tool(SummarizationTool())
    
    # Process a query using the research agent
    result = await controller.process("How does artificial intelligence work?", agent_name="research")
    print("Research Agent Result:", json.dumps(result, indent=2))
    
    # Process content using the content analysis agent
    content = "This is a sample text that we want to analyze. It contains multiple sentences with various information. The goal is to extract meaningful insights and provide a concise summary."
    result = await controller.process(content, agent_name="content_analysis")
    print("Content Analysis Result:", json.dumps(result, indent=2))
    
    # Execute a tool directly
    result = await controller.execute_tool("search", "Python programming")
    print("Search Tool Result:", json.dumps(result, indent=2))
    
    # Run a pipeline
    pipeline_result = await controller.run_pipeline(
        "Machine learning fundamentals",
        ["research", "content_analysis"]
    )
    print("Pipeline Result:", json.dumps(pipeline_result, indent=2))

# This can be run as a script to demonstrate functionality
if __name__ == "__main__":
    import asyncio
    asyncio.run(example_usage())
