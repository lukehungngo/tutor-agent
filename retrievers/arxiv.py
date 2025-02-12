import arxiv

class ArxivSearch:
    def __init__(self, query: str):
        self.query = query

    def search(self, max_results: int = 3) -> list[str]:
        search = arxiv.Search(
            query=self.query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        results = []
        for result in search.results():
            results.append(f"{result.title} - {result.pdf_url}")
        return results