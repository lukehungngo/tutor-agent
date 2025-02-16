from langchain_community.tools.tavily_search import TavilySearchResults
from config.settings import *
from langchain.tools import Tool

temp = TavilySearchResults(max_results=2)
tavily_tool = Tool(
    name="tavily_search",
    description="A powerful web search tool that retrieves up-to-date information from across the internet, returning the most relevant results with summaries",
    func=temp.run
)
