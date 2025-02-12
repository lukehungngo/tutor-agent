from googlesearch import search

class GoogleSearch:
    def __init__(self, query):
        self.query = query

    def search(self, max_results=3):
        return search(self.query, max_results)
