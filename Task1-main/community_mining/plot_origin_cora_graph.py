import networkx as nx
import numpy as np
import plotly.graph_objects as go
import os
from sklearn.utils import shuffle
def plot_graph(G, pos):
    # Get the positions of nodes
    node_pos = np.array([pos[node] for node in G.nodes()])

    # Create edge coordinates for plotting
    edge_x = []
    edge_y = []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_y.append(y0)
        edge_y.append(y1)

    # Create edge trace for Plotly
    edge_trace = go.Scatter(x=edge_x, y=edge_y,
                            line=dict(width=0.5, color='#888'),
                            hoverinfo='none',
                            mode='lines')

    # Create node trace for Plotly
    node_trace = go.Scatter(x=node_pos[:, 0], y=node_pos[:, 1],
                            mode='markers',
                            hoverinfo='text',
                            marker=dict(color='#87CEEB', size=10,
                                        line=dict(width=0), opacity=1.0))

    # Create the layout for the plot
    layout = go.Layout(title="Original Graph",
                       titlefont_size=16,
                       showlegend=False,
                       hovermode='closest',
                       xaxis=dict(showgrid=False, zeroline=False),
                       yaxis=dict(showgrid=False, zeroline=False),
                       dragmode='zoom',  # Enable zoom and pan interactions
                       margin=dict(l=0, r=0, t=40, b=0))  # Adjust margins

    # Plot the graph
    fig = go.Figure(data=[edge_trace, node_trace], layout=layout)
    fig.show()
    fig.write_image(f"CORA.pdf", format="pdf")

if __name__ == "__main__":
    # Load the graph and layout

    #loading the data

    all_data = []
    all_edges = []

    for root,dirs,files in os.walk('./cora'):
        for file in files:
            if '.content' in file:
                with open(os.path.join(root,file),'r') as f:
                    all_data.extend(f.read().splitlines())
            elif 'cites' in file:
                with open(os.path.join(root,file),'r') as f:
                    all_edges.extend(f.read().splitlines())

                    
    #Shuffle the data because the raw data is ordered based on the label
    random_state = 42
    all_data = shuffle(all_data,random_state=random_state)
    #loading the data

    all_data = []
    all_edges = []

    for root,dirs,files in os.walk('./cora'):
        for file in files:
            if '.content' in file:
                with open(os.path.join(root,file),'r') as f:
                    all_data.extend(f.read().splitlines())
            elif 'cites' in file:
                with open(os.path.join(root,file),'r') as f:
                    all_edges.extend(f.read().splitlines())

                    
    #Shuffle the data because the raw data is ordered based on the label
    random_state = 42
    all_data = shuffle(all_data,random_state=random_state)
    #parse the data
    labels = []
    nodes = []
    features = []

    for i,data in enumerate(all_data):
        elements = data.split('\t')
        labels.append(elements[-1])
        features.append(elements[1:-1])
        nodes.append(elements[0])

    features = np.array(features,dtype=int)
    a = features.shape[0] #the number of nodes
    b = features.shape[1] #the size of node features

    print('features shape: ', features.shape)

    #parse the edge
    edge_list=[]
    for edge in all_edges:
        e = edge.split('\t')
        edge_list.append((e[0],e[1]))

    print('\nNumber of nodes (a): ', a)
    print('\nNumber of features (b) of each node: ', b)
    print('\nCategories: ', set(labels))

    num_classes = len(set(labels))
    print('\nNumber of classes: ', num_classes)
    #build the graph
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edge_list)

    print(G)

    pos = nx.spring_layout(G)  # Position nodes using spring layout

    # Plot the original graph
    plot_graph(G, pos)
