from IPython.display import Image, display
from datetime import datetime


def generate_graph(graph):
    try:
        graph_image = graph.get_graph().draw_mermaid_png()  # Get raw PNG data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Save the image to a file
        with open(f"generated/graph_{timestamp}.png", "wb") as f:
            f.write(graph_image)

    except Exception as e:
        print(f"Failed to display graph: {e}")
