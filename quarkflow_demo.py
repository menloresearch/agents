#!/usr/bin/env python3
"""
Quarkflow Demo Script

This script demonstrates how to use the quarkflow library to create and run agents.
"""
import dotenv
from dotenv import load_dotenv
load_dotenv()
import asyncio
import os
import sys
import json
from typing import Dict, Any, List, Optional

# Import the bhumi client for LLM inference
from bhumi.base_client import BaseLLMClient, LLMConfig

# Import the quarkflow library components
from quarkflow import Agent, BaseTool, ControllerAgent
from quarkflow.examples import ResearchAgent, ContentAnalysisAgent, SimpleSearchTool, SummarizationTool

# Create a custom agent that combines functionality
class CombinedAgent(Agent):
    """Custom agent that combines research and content analysis."""
    
    def __init__(self, search_api_key: Optional[str] = None, llm_config: Optional[Dict[str, Any]] = None):
        """Initialize the combined agent."""
        super().__init__(name="combined", llm_config=llm_config)
        self.research_agent = ResearchAgent(search_api_key=search_api_key, llm_config=llm_config)
        self.content_agent = ContentAnalysisAgent(llm_config=llm_config)
    
    async def process(self, query: str, **kwargs) -> Dict[str, Any]:
        """Process a query by performing research and analyzing the results."""
        # First, perform research
        research_result = await self.research_agent.process(query)
        
        # Extract key information from the research
        search_results = research_result.get("search_results", [])
        
        # Create a combined text from search results to analyze
        combined_text = "\n".join([
            f"Title: {result.get('title', '')}" 
            for result in search_results
        ])
        
        # If we have content to analyze, do so
        if combined_text:
            analysis_result = await self.content_agent.process(combined_text)
            
            # Combine the results
            return {
                "query": query,
                "research": research_result,
                "analysis": analysis_result,
                "summary": analysis_result.get("summary", "No summary available")
            }
        else:
            # Just return the research results if no content to analyze
            return {
                "query": query,
                "research": research_result,
                "summary": "No search results to analyze"
            }

# Create a custom tool
class LoggingTool(BaseTool):
    """A simple tool that logs information."""
    
    def __init__(self, log_file: Optional[str] = None):
        """Initialize the logging tool."""
        super().__init__(name="logger")
        self.log_file = log_file
    
    async def execute(self, message: str, **kwargs) -> Dict[str, Any]:
        """Log a message."""
        print(f"LOG: {message}")
        
        if self.log_file:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(f"{message}\n")
        
        return {
            "message": message,
            "logged": True,
            "log_file": self.log_file
        }

async def main():
    """Run the quarkflow demo."""
    print("===== Quarkflow Demo =====")
    
    # Create output directory
    os.makedirs("demo_output", exist_ok=True)
    
    # Get API keys from environment variables
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    search_api_key = os.getenv("BRAVE_API_KEY")  # Or any search API key
    
    # Configure LLM if API key is available
    llm_config = None
    if gemini_api_key:
        print("Using Gemini for LLM inference")
        llm_config = {
            "api_key": gemini_api_key,
            "model": "gemini/gemini-1.5-flash-latest",  # Faster, lighter model for demos
            "debug": True  # Set to True to see detailed logs
        }
    else:
        print("No LLM API key found. Using fallback mock responses.")
    
    # Create a controller agent
    controller = ControllerAgent(
        name="demo_controller",
        output_dir="demo_output"
    )
    
    # Create and register agents with LLM config
    research_agent = ResearchAgent(search_api_key=search_api_key, llm_config=llm_config)
    content_agent = ContentAnalysisAgent(llm_config=llm_config)
    combined_agent = CombinedAgent(search_api_key=search_api_key, llm_config=llm_config)
    
    controller.register_agent(research_agent)
    controller.register_agent(content_agent)
    controller.register_agent(combined_agent)
    
    # Create and register tools
    search_tool = SimpleSearchTool()
    summarize_tool = SummarizationTool()
    logging_tool = LoggingTool(log_file="demo_output/agent_log.txt")
    
    controller.register_tool(search_tool)
    controller.register_tool(summarize_tool)
    controller.register_tool(logging_tool)
    
    # Demo questions to process
    demo_questions = [
        "What are neural networks?",
        "How does quantum computing work?",
        "Explain the basics of reinforcement learning"
    ]
    
    for question in demo_questions:
        print(f"\nProcessing question: {question}")
        
        # Log the question
        await controller.execute_tool("logger", f"New question: {question}")
        
        # Process with the combined agent
        result = await controller.process(question, agent_name="combined")
        
        # Display the results
        print(f"Summary: {result.get('summary', 'No summary available')}")
        print("Research results:")
        for item in result.get("research", {}).get("search_results", []):
            print(f"- {item.get('title', 'Untitled')}")
    
    print("\n===== Pipeline Demo =====")
    
    # Demonstrate pipeline functionality with fixed error handling
    try:
        print("\nRunning a research + content analysis pipeline...")
        pipeline_result = await controller.run_pipeline(
            "The future of artificial intelligence",
            ["research", "content_analysis"]
        )
        print(f"Pipeline summary: {pipeline_result.get('summary', 'No summary available')}")
    except Exception as e:
        print(f"Pipeline error: {str(e)}")
        print("This is expected if no LLM API key is provided")
    
    # Summary is now printed inside the try block to avoid reference errors
    
    print("\n===== Demo Complete =====")

if __name__ == "__main__":
    asyncio.run(main())
