import community
import infomap
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
from torch import seed
import argparse
import os
import numpy as np
import plotly.graph_objects as go   

def louvain(G, pos):
    # Compute the best partition using Louvain community detection
    partition = nx.algorithms.community.louvain_communities(G, seed=2024)


    # Prepare nodes and community color assignments
    nodes = []
    comm = []
    for i in range(len(partition)):
        nodes = nodes + list(partition[i])
        comm = comm + [i] * len(partition[i])

    # Get the positions of nodes
    node_pos = np.array([pos[node] for node in nodes])

    # Create Plotly scatter plot for the nodes
    edge_x = []
    edge_y = []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_y.append(y0)
        edge_y.append(y1)

    # Plot edges
    edge_trace = go.Scatter(x=edge_x, y=edge_y,
                            line=dict(width=0.5, color='#888'),
                            hoverinfo='none',
                            mode='lines')

    # Plot nodes with colors based on communities
    node_trace = go.Scatter(x=node_pos[:, 0], y=node_pos[:, 1],
                            mode='markers',
                            hoverinfo='text',
                            marker=dict(color=comm, size=15,
                                        colorscale='Viridis', line=dict(width=0), opacity=1.0))

    # Create the layout for the plot
    layout = go.Layout(title="Louvain Community Detection",
                       titlefont_size=16,
                       showlegend=False,
                       hovermode='closest',
                       xaxis=dict(showgrid=False, zeroline=False),
                       yaxis=dict(showgrid=False, zeroline=False),
                       dragmode='zoom',  # Enable zoom and pan interactions
                       margin=dict(l=0, r=0, t=40, b=0))  # Adjust margins

    # Plot everything
    fig = go.Figure(data=[edge_trace, node_trace], layout=layout)

    # Show the plot
    fig.show()

    fig.write_image(f"louvain.pdf", format="pdf")
    # Calculate modularity
    partition_dict = {k: v for k, v in zip(nodes, comm)}
    modularity = community.modularity(partition_dict, G)
    print([max(partition_dict.values()) + 1, modularity])

    return len(partition)


def random_walk(G, pos):
    # Run Infomap algorithm for community detection
    infomapWrapper = infomap.Infomap("--two-level --silent")
    for e in G.edges():
        infomapWrapper.addLink(*e)
    infomapWrapper.run()
    tree = infomapWrapper

    # Assign nodes to modules
    partition = {}
    for node in tree.nodes:
        partition[node.node_id] = node.module_id

    # Prepare data for visualization
    nodes = list(partition.keys())
    comm = list(partition.values())

    # Get positions of nodes
    node_pos = np.array([pos[node] for node in nodes])

    # Create Plotly scatter plot for the nodes
    edge_x = []
    edge_y = []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_y.append(y0)
        edge_y.append(y1)

    # Plot edges
    edge_trace = go.Scatter(x=edge_x, y=edge_y,
                            line=dict(width=0.5, color='#888'),
                            hoverinfo='none',
                            mode='lines')

    # Plot nodes with colors based on communities
    node_trace = go.Scatter(x=node_pos[:, 0], y=node_pos[:, 1],
                            mode='markers',
                            hoverinfo='text',
                            marker=dict(color=comm, size=15,
                                        colorscale='Viridis', line=dict(width=0), opacity=1.0))

    # Create the layout for the plot
    layout = go.Layout(title="Random Walk Community Detection",
                       titlefont_size=16,
                       showlegend=False,
                       hovermode='closest',
                       xaxis=dict(showgrid=False, zeroline=False),
                       yaxis=dict(showgrid=False, zeroline=False),
                       dragmode='zoom',  # Enable zoom and pan interactions
                       margin=dict(l=0, r=0, t=40, b=0))  # Adjust margins

    # Plot everything
    fig = go.Figure(data=[edge_trace, node_trace], layout=layout)
    fig.show()
    fig.write_image(f"Infomap.pdf", format="pdf")

    # Calculate modularity
    modularity = community.modularity(partition, G)
    print([max(partition.values()) + 1, modularity])

    return tree.numTopModules()

def label_propagation(G, pos):
    # Compute the best partition
    partition = nx.algorithms.community.label_propagation_communities(G)
    keys, values = [], []
    for i, item in enumerate(partition):
        keys = keys + list(item)
        values = values + [i] * len(item)
    partition_dict = {k: v for k, v in zip(keys, values)}

    # Get the positions of nodes
    node_pos = np.array([pos[node] for node in keys])

    # Create Plotly scatter plot for the nodes
    edge_x = []
    edge_y = []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_y.append(y0)
        edge_y.append(y1)

    # Plot edges
    edge_trace = go.Scatter(x=edge_x, y=edge_y,
                            line=dict(width=0.5, color='#888'),
                            hoverinfo='none',
                            mode='lines')

    # Plot nodes with colors based on communities
    node_trace = go.Scatter(x=node_pos[:, 0], y=node_pos[:, 1],
                            mode='markers',
                            hoverinfo='text',
                            marker=dict(color=values, size=15,
                                        colorscale='Viridis', line_width=0, opacity=1.0))

    # Create the layout for the plot
    layout = go.Layout(title="Synchronous LPA",
                       titlefont_size=16,
                       showlegend=False,
                       hovermode='closest',
                       xaxis=dict(showgrid=False, zeroline=False),
                       yaxis=dict(showgrid=False, zeroline=False),
                       dragmode='zoom',  # Enable zoom and pan interactions
                       margin=dict(l=0, r=0, t=40, b=0))  # Adjust margins

    # Plot everything
    fig = go.Figure(data=[edge_trace, node_trace], layout=layout)
    fig.show()
    fig.write_image(f"Label_Propagation.pdf", format="pdf")

    # Calculate modularity
    modularity = community.modularity(partition_dict, G)
    print([max(partition_dict.values()) + 1, modularity])

    return max(partition_dict.values()) + 1

def asyn_lpa(G, pos):
    # Compute the best partition
    partition = nx.algorithms.community.asyn_lpa_communities(G, seed=2024)
    keys, values = [], []
    for i, item in enumerate(partition):
        keys = keys + list(item)
        values = values + [i] * len(item)
    partition_dict = {k: v for k, v in zip(keys, values)}

    # Get the positions of nodes
    node_pos = np.array([pos[node] for node in keys])

    # Create Plotly scatter plot for the nodes
    edge_x = []
    edge_y = []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_y.append(y0)
        edge_y.append(y1)

    # Plot edges
    edge_trace = go.Scatter(x=edge_x, y=edge_y,
                            line=dict(width=0.5, color='#888'),
                            hoverinfo='none',
                            mode='lines')

    # Plot nodes with colors based on communities
    node_trace = go.Scatter(x=node_pos[:, 0], y=node_pos[:, 1],
                            mode='markers',
                            hoverinfo='text',
                            marker=dict(color=values, size=15,
                                        colorscale='Viridis', line_width=0, opacity=1.0))

    # Create the layout for the plot
    layout = go.Layout(title="Synchronous LPA",
                       titlefont_size=16,
                       showlegend=False,
                       hovermode='closest',
                       xaxis=dict(showgrid=False, zeroline=False),
                       yaxis=dict(showgrid=False, zeroline=False),
                       dragmode='zoom',  # Enable zoom and pan interactions
                       margin=dict(l=0, r=0, t=40, b=0))  # Adjust margins

    # Plot everything
    fig = go.Figure(data=[edge_trace, node_trace], layout=layout)
    fig.show()
    fig.write_image(f"Asynchronous_Label_Propagation.pdf", format="pdf")

    # Calculate modularity
    modularity = community.modularity(partition_dict, G)
    print([max(partition_dict.values()) + 1, modularity])

    return max(partition_dict.values()) + 1

if __name__ == "__main__":

    # Load the graph and layout
    src = "facebook_combined.txt"  
    G = nx.read_edgelist(src, nodetype=int)
    pos = nx.spring_layout(G)

    print("louvain test")
    louvain(G, pos)

    print("random walk test")
    random_walk(G, pos)

    print("label propagation test")
    label_propagation(G, pos)

    print("asynchronous label propagation test")
    asyn_lpa(G, pos)



