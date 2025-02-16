from langchain_community.tools.tavily_search import TavilySearchResults
from config.settings import *
from langchain.tools import Tool

tavily_search = TavilySearchResults(max_results=5)
tavily_tool = Tool(
    name="Tavily Search",
    description="A powerful web search tool that retrieves up-to-date information from across the internet, returning the most relevant results with summaries",
    func=tavily_search.run,
)
