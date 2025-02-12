import wikipedia

class WikipediaSearch:
    def __init__(self, query):
        self.query = query

    def search(self, max_results=3):
        try:
            return wikipedia.summary(self.query, sentences=max_results)
        except:
            return "No Wikipedia results found."