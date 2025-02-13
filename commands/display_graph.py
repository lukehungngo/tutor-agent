from IPython.display import Image, display
from services.chatbot import graph

def display_graph(graph):
    try:
        graph_image = graph.get_graph().draw_mermaid_png()  # Get raw PNG data

        # Save the image to a file
        with open("generated/graph.png", "wb") as f:
            f.write(graph_image)

        # Display the saved image
        img = Image("graph.png")
        display(img)

    except Exception as e:
        print(f"Failed to display graph: {e}")

def main(*args):
    display_graph(graph)

if __name__ == "__main__":
    main()
