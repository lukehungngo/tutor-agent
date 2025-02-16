from langchain.tools import Tool
# from langchain_community.tools import DuckDuckGoSearchRun
# search = DuckDuckGoSearchRun()

from duckduckgo_search import DDGS
ddg = DDGS()

def duckduckgo_search(input_text):
    sites = [
        "site:khanacademy.org",
        "site:coursera.org",
        "site:edx.org",
        "site:mit.edu",
        "site:stanford.edu",
        "site:harvard.edu",
        "site:ted.com",
        "site:britannica.com",
        "site:scholarpedia.org",
        "site:openlibrary.org",
        "site:quora.com",
        "site:medium.com",
    ]
    site_query = " OR ".join(sites)
    search_results = ddg.text(f"({site_query}) {input_text}", region="wt-wt", max_results=5)
    results = []
    for result in search_results:
        title = result.get("title", "")
        url = result.get("href", "")
        content = result.get("body", "")
        if title == "" and url == "" and content == "":
            result = result
        else:
            result = {
                "title": title,
                "url": url, 
                "content": content
            }
        results.append(result)
    return results

duckduckgo_tool = Tool(
    name="DuckDuckGo Search",
    func=duckduckgo_search,
    description="useful for finding educational content from top learning platforms and academic institutions"
)
