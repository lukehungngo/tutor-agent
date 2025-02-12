from retrievers import TavilySearch, WikipediaSearch, GoogleSearch, ArxivSearch
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from config.settings import settings

tavily_tool = Tool(
    name="Tavily Research",
    func=TavilySearch.search,
    description="Retrieve AI-powered research insights from Tavily."
)

wikipedia_tool = Tool(
    name="Wikipedia Search",
    func=WikipediaSearch.search,
    description="Search Wikipedia for academic explanations."
)

google_tool = Tool(
    name="Google Search",
    func=GoogleSearch.search,
    description="Search Google for academic explanations."
)

arxiv_tool = Tool(
    name="Arxiv Search",
    func=ArxivSearch.search,
    description="Search Arxiv for academic explanations."
)

tools = [tavily_tool, wikipedia_tool, arxiv_tool, google_tool]

agent = initialize_agent(
    tools=tools,
    llm=settings.openai_provider,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

def ai_research(subqueries):
    """Executes multiple research queries and aggregates results."""
    results = {}
    for query in subqueries:
        results[query] = agent.run(query)
    return results