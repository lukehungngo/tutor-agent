from retrievers.arxiv import arxiv_search
from retrievers.wikipedia import wikipedia_summary
from retrievers.google import google_search
from retrievers.tavily import tavily_search
from retrievers.duckduckgo import duckduckgo_search


def test_arxiv():
    print("\nTesting Arxiv Search:")
    query = "quantum computing"
    results = arxiv_search(query, max_results=2)
    print(f"Query: {query}")
    print("Results:")
    print(results)


def test_wikipedia():
    print("\nTesting Wikipedia Search:")
    query = "artificial intelligence"
    try:
        results = wikipedia_summary(query, max_sentences=5)
        print(f"Query: {query}")
        print("Results:")
        print(results)
    except Exception as e:
        print(f"Error: {e}")


def test_google():
    print("\nTesting Google Search:")
    query = "best python tutorials"
    try:
        results = google_search(query, max_results=5)
        print(f"Query: {query}")
        print("Results:")
        print(results)
    except Exception as e:
        print(f"Error: {e}")


def test_tavily():
    print("\nTesting Tavily Search:")
    query = "quantum computing"
    results = tavily_search.run(query)
    print(f"Query: {query}")
    print("Results:")
    print(results)


def test_duckduckgo():
    print("\nTesting DuckDuckGo Search:")
    query = "quantum computing"
    results = duckduckgo_search(query)
    print(f"Query: {query}")
    print("Results:")
    print(results)


if __name__ == "__main__":
    print("Running retriever tests...")
    test_arxiv()
    test_wikipedia()
    test_google()
    test_tavily()
    test_duckduckgo()
    print("\nTests completed.")
