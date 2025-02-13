from retrievers import TavilySearch, WikipediaSearch, GoogleSearch, ArxivSearch
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from config.settings import settings
from models.topic import Subtopics

# Define Tools for Research
tools = {
    "tavily": Tool(
        name="Tavily Research",
        func=TavilySearch.search,
        description="Retrieve AI-powered research insights from Tavily."
    ),
    "wikipedia": Tool(
        name="Wikipedia Search",
        func=WikipediaSearch.search,
        description="Search Wikipedia for academic explanations."
    ),
    "google": Tool(
        name="Google Search",
        func=GoogleSearch.search,
        description="Search Google for academic explanations."
    ),
    "arxiv": Tool(
        name="Arxiv Search",
        func=ArxivSearch.search,
        description="Search Arxiv for academic explanations."
    ),
}

# Create ReAct Agent using LangGraph
agent = create_react_agent(
    model=settings.open_api_client,
    tools=list(tools.values()),
    response_format=Subtopics
)

# Define AI Research Function
def ai_research(subqueries):
    """Executes multiple research queries and aggregates results using LangGraph."""
    results = {}
    for query in subqueries:
        response = agent.invoke(query)
        # Fix: Ensure response contains a structured output
        if not response.get("structured_response"):
            response["structured_response"] = {"query": query, "error": "No valid response generated"}

        results[query] = response["structured_response"]
    return results
