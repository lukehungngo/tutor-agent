from langchain.tools import Tool
from langchain_community.tools import DuckDuckGoSearchRun

search = DuckDuckGoSearchRun()

def duck_wrapper(input_text):
    search_results = search.run(f"site:webmd.com {input_text}")
    return search_results

duckduckgo_tool = Tool(
        name = "DuckDuckGo Search WebMD",
        func=duck_wrapper,
        description="useful for when you need to answer questions about education, health, and other topics"
    )
