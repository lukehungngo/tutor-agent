from googlesearch import search
from langchain.tools import Tool
from googlesearch import SearchResult

def google_search(query, max_results=5):
    results = search(query, num_results=max_results, advanced=True)
    results_dict = []
    for result in results:
        results_dict.append({"title": result.title, "content": result.description, "url": result.url})
    return str(results_dict)

google_tool = Tool(
    name="Google Search",
    func=google_search,
    description="useful for finding educational content from top learning platforms and academic institutions"
)
