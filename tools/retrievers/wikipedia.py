import wikipedia
from langchain.tools import Tool


def wikipedia_summary(query, max_sentences=5):
    result = wikipedia.summary(query, sentences=max_sentences)
    return str({"summary": result})


wikipedia_summary_tool = Tool(
    name="Wikipedia Search",
    func=wikipedia_summary,
    description="useful for getting summaries of Wikipedia articles, providing concise overviews or summaries of topics from the online encyclopedia",
)
