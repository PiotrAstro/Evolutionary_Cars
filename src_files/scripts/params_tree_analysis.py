import pickle

import matplotlib.pyplot as plt
import networkx as nx

FILE = r"C:\Piotr\AIProjects\Evolutionary_Cars\logs\EvMuPop1714058849\population_tree.pkl"


def add_nodes_edges(tree, graph, parent=None, pos={}, x=0, y=0, layer=1, id_count=None):
    if id_count is None:
        id_count = [0]
    node_value, children = tree
    node_id = id_count[0]
    graph.add_node(node_id, value=node_value)
    pos[node_id] = (x, y)

    if parent is not None:
        graph.add_edge(parent, node_id)

    id_count[0] += 1

    if children:
        dx = 1.0 / (layer + 1)
        next_x = x - (len(children) * dx) / 2
        for child in children:
            next_x += dx
            add_nodes_edges(child, graph, parent=node_id, pos=pos, x=next_x, y=y - 1, layer=layer + 1,
                            id_count=id_count)
    return pos


def draw_forest(trees):
    G = nx.DiGraph()
    pos = {}
    for i, tree in enumerate(trees):
        pos = add_nodes_edges(tree, G, x=10 * i, y=0)

    # Extract the 'value' attribute from nodes to use for coloring
    values = [G.nodes[node]['value'] for node in G.nodes()]
    min_val = min(values)
    max_val = max(values)
    colors = [plt.cm.viridis((value - min_val) / (max_val - min_val)) for value in values]

    fig, ax = plt.subplots(figsize=(12, 8))
    nx.draw(G, pos, ax=ax, with_labels=False, node_color=colors, node_size=300, edge_color='gray', font_weight='bold',
            font_size=9)
    plt.title("Forest Visualization")

    # Add the colorbar
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=min_val, vmax=max_val))
    sm._A = []  # Fake up the array of the scalar mappable. Urgh...
    plt.colorbar(sm, ax=ax, label='Node value')

    plt.show()


with open(FILE, "rb") as file:
    trees = pickle.load(file)
draw_forest(trees)
