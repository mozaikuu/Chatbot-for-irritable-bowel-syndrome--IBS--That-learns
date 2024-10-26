import base64
from IPython.display import Image, display
import matplotlib.pyplot as plt

def mm(graph):
    # Validate the graph input
    if not isinstance(graph, str) or not graph.strip():
        raise ValueError("Graph must be a non-empty string.")
    
    graphbytes = graph.encode("utf8")
    base64_bytes = base64.urlsafe_b64encode(graphbytes)
    base64_string = base64_bytes.decode("ascii")
    display(Image(url="https://mermaid.ink/img/" + base64_string))

mm("""
graph LR;
    A--> B & C & D;
    B--> A & E;
    C--> A & E;
    D--> A & E;
    E--> B & C & D;
""")

