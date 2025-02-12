def generate_learning_report(task, research_results):
    """Creates a structured learning report as a JSON object."""
    learning_report = {
        "task": task,
        "learning_path": []
    }

    for topic, content in research_results.items():
        learning_report["learning_path"].append({
            "topic": topic,
            "summary": content
        })

    return learning_report