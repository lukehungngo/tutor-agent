import arxiv
from langchain.tools import Tool


def arxiv_search(query, max_results=5):
    search = arxiv.Search(
        query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance
    )
    results = []
    for result in search.results():
        results.append(
            {
                "title": result.title,
                "url": result.pdf_url,
                "content": result.summary,
                "authors": [str(author) for author in result.authors],
                "published": str(result.published),
                "primary_category": result.primary_category,
                "categories": list(result.categories),
            }
        )

    return str(results)


arxiv_tool = Tool(
    name="Arxiv Search",
    func=arxiv_search,
    description="useful for searching Arxiv, a free online archive of scientific papers",
)
