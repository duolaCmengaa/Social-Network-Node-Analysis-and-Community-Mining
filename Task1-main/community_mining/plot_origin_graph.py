import networkx as nx
import numpy as np
import plotly.graph_objects as go

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
    fig.write_image(f"Facebook.pdf", format="pdf")

if __name__ == "__main__":
    # Load the graph and layout
    src = "facebook_combined.txt"  
    G = nx.read_edgelist(src, nodetype=int)
    pos = nx.spring_layout(G)  # Position nodes using spring layout

    # Plot the original graph
    plot_graph(G, pos)
