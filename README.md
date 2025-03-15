# Deep Research Tool

A powerful research tool that leverages the Brave Search API to find relevant information, extracts content using pathik, and analyzes it using Google's Gemini AI model.

## Features

- **Web Search**: Search the web using Brave Search API
- **Content Extraction**: Extract content from web pages using pathik
- **AI Analysis**: Analyze the content using Google's Gemini AI model
- **Multiple Analysis Types**: Choose from comprehensive analysis, quick summary, facts extraction, or insights
- **Domain Filtering**: Filter search results by domain
- **Customizable**: Adjust search parameters, language, country, and more
- **Gemini-Powered Reports**: Generate comprehensive research reports with executive summaries, detailed analysis, and recommendations
- **Parallel Processing**: Crawl multiple web pages simultaneously for faster research
- **High-Volume Research**: Analyze up to dozens of web pages in a single research session

## Research Agent

The Research Agent extends basic research capabilities with:

- **Intent Recognition**: Automatically determines if a query requires fresh research, existing knowledge retrieval, or simple conversation
- **Memory System**: Stores and retrieves conversation history for contextual awareness
- **RAG Capabilities**: Leverages previously researched information to answer questions without new research
- **Vector Database**: Stores and retrieves documents based on semantic similarity
- **Embedding System**: Converts text into vector representations for efficient retrieval
- **Multi-modal Response**: Generates responses using both fresh research and existing knowledge

### Agent Components

- **Vector Database**: Abstracted storage system for document embeddings with support for multiple backends
- **Embedding Provider**: Converts text to vector representations with pluggable provider architecture
- **Chat Memory**: Manages conversation history with persistence across sessions
- **Research Tool**: Core implementation of web search, content extraction, and analysis
- **Report Generator**: Creates comprehensive research reports in multiple formats

## Advanced Usage

The tool provides several command-line options:

```bash
python deep_research_cli.py "climate change solutions" --count 15 --pages 5 --analysis insights --filter-domains "nature.com,science.org,nationalgeographic.com" --generate-report
```

### Command Line Arguments

- `query` - Your research query (required)
- `--count` - Number of search results to fetch (default: 20, max: 20)
- `--pages` - Maximum number of pages to analyze (default: 10)
- `--country` - Country code for search results (default: US)
- `--lang` - Search language (default: en)
- `--filter-domains` - Comma-separated list of domains to include
- `--analysis` - Type of analysis to perform (choices: "comprehensive", "summary", "facts", "insights", default: "comprehensive")
- `--timeout` - Reserved for future use (default: 30, currently not affecting crawling)
- `--output-dir` - Directory to save output files (default: output_data)
- `--debug` - Enable debug mode
- `--generate-report` - Generate a comprehensive report using Gemini's advanced analysis capabilities
- `--model` - Gemini model to use for report generation (default: gemini/gemini-1.5-flash-8b)
- `--max-concurrent` - Maximum number of concurrent URL crawling tasks (default: 5)
- `--test-apis` - Test the Brave Search and Gemini APIs without performing research

## Performance Tips

### Optimizing Parallel Processing

The tool now supports parallel processing, which significantly improves performance when analyzing multiple pages. You can adjust the degree of parallelism with the `--max-concurrent` parameter:

```bash
# Process up to 10 URLs in parallel
python deep_research_cli.py "your research topic" --max-concurrent 10
```

For most systems, setting `--max-concurrent` between 5-10 provides a good balance between speed and system resource usage. Higher values may improve performance on systems with many CPU cores and good internet connections.

### Handling Large Volumes of Data

For very comprehensive research, you can increase the number of pages:

```bash
# Analyze up to 20 pages from 30 search results
python deep_research_cli.py "deep learning neural networks" --count 30 --pages 20
```

Note that analyzing more pages requires more processing time and may increase the cost of API usage for Brave Search and Gemini.

### Brave Search API Limitations

The Brave Search API has certain limitations to be aware of:

- **Results Count**: The API limits the number of results per request to a maximum of 20
- **Query Length**: Very long queries may be truncated
- **Rate Limits**: There may be rate limits depending on your API tier
- **Valid Parameters**: The API requires valid 2-letter country and language codes

If you encounter a validation error, try:
- Using shorter, more specific queries
- Ensuring your API key is valid and active
- Using standard 2-letter country and language codes (e.g., "US", "en")

### Testing API Connections

If you're having trouble with API connections, you can test the APIs before running a full research session:

```bash
python deep_research_cli.py --test-apis
```

This will:
- Test your Brave Search API key
- Display rate limit information if available
- Test your Gemini API key
- Provide detailed error messages if anything is wrong

Once you've verified your API connections are working, you can proceed with your research:

```bash
# First test APIs, then perform research if successful
python deep_research_cli.py --test-apis "quantum computing applications"
```

## Output

The tool saves the research results in JSON format in the `output_data` directory (or the directory specified with `--output-dir`). The filename includes the query and timestamp.

## Report Generation

When using the `--generate-report` flag, the tool will use Gemini to generate a comprehensive research report that includes:

1. **Executive Summary**: A concise overview of the key findings (250-300 words)
2. **Detailed Analysis**: In-depth examination of key themes and concepts (800-1000 words)
3. **Conclusions and Recommendations**: Summary of key takeaways and actionable recommendations
4. **Source Analysis**: Detailed insights for each source, highlighting unique contributions

Reports are generated in three formats:
- JSON: Contains all the raw data
- Markdown: Formatted for easy viewing in text editors or GitHub
- HTML: Beautifully styled with CSS for browser viewing

Example command to generate a report:
```bash
python deep_research_cli.py "quantum computing applications" --generate-report --model "gemini/gemini-1.5-flash-8b"
```