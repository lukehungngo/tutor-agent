import requests
from langchain.tools import Tool
from config import *

class TavilySearch: 
    def __init__(self, query):
        self.query = query

    def search(self, max_results=3):
        """Calls Tavily API for AI-powered research results."""
        url = "https://api.tavily.com/search"
        headers = {"Authorization": f"Bearer {settings.tavily_api_key}"}
        params = {"query": self.query, "max_results": max_results}

        response = requests.get(url, headers=headers, params=params)
        return response.json()