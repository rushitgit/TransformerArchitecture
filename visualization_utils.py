import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import random

def create_transformer_architecture_diagram(model_info):
    """
    Create an advanced visual diagram of the transformer architecture with dynamic elements.
    
    Args:
        model_info (dict): Model architecture information
        
    Returns:
        go.Figure: Plotly figure with animated transformer architecture
    """
    # Import inside function to avoid dependency issues
    import plotly.graph_objects as go
    import numpy as np
    from plotly.subplots import make_subplots
    import random
    
    # Extract model information
    num_layers = model_info['num_layers']
    num_heads = model_info['num_heads']
    hidden_size = model_info['hidden_size']
    model_name = model_info['model_name']
    
    # Create base figure
    fig = go.Figure()
    
    # Define colors with transparency
    colors = {
        'background': 'rgba(240, 240, 250, 0.95)',
        'input': 'rgba(65, 105, 225, 0.7)',
        'attention': 'rgba(220, 20, 60, 0.7)',
        'ffn': 'rgba(50, 205, 50, 0.7)',
        'output': 'rgba(255, 165, 0, 0.7)',
        'connection': 'rgba(128, 128, 128, 0.4)',
        'attention_head': 'rgba(70, 130, 180, 0.8)',
        'norm': 'rgba(255, 255, 255, 0.9)',
        'node': 'rgba(100, 100, 120, 0.7)'
    }
    
    # Set canvas dimensions
    width, height = 1000, 800
    x_center = width / 2
    
    # Calculate vertical spacing
    total_components = num_layers + 2  # Input, N transformer layers, Output
    layer_spacing = height / (total_components + 1)
    
    # Add background 
    fig.add_shape(
        type="rect",
        x0=0, x1=width,
        y0=0, y1=height,
        fillcolor=colors['background'],
        line_width=0,
        layer="below"
    )
    
    # Helper functions for creating nodes and connections
    def add_component(x, y, width, height, color, name, hovertext, opacity=1.0):
        # Adds a rectangular component with hover effect
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode='markers',
            marker=dict(
                symbol='square',
                size=[width],
                line=dict(color='rgba(0, 0, 0, 0.5)', width=2),
                color=color,
                opacity=opacity,
                sizemode='diameter'
            ),
            name=name,
            hoverinfo='text',
            hovertext=hovertext,
            showlegend=False
        ))
    
    def add_connection(x0, y0, x1, y1, color, width=1.5, dash='solid'):
        # Adds a connection line between components
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1],
            mode='lines',
            line=dict(color=color, width=width, dash=dash),
            hoverinfo='none',
            showlegend=False
        ))
    
    def add_animated_nodes(x, y, num_nodes=5, line_length=50, color='rgba(100, 100, 255, 0.7)'):
        # Adds small animated nodes that appear to flow through connections
        positions = []
        node_colors = []
        node_sizes = []
        
        for i in range(num_nodes):
            # Distribute nodes evenly along the line
            pos = i / (num_nodes - 1) if num_nodes > 1 else 0.5
            pos_x = x[0] + pos * (x[1] - x[0])
            pos_y = y[0] + pos * (y[1] - y[0])
            
            # Randomize positions slightly for more organic flow
            pos_x += random.uniform(-5, 5)
            pos_y += random.uniform(-5, 5)
            
            positions.append((pos_x, pos_y))
            
            # Varied colors and sizes for visual interest
            alpha = 0.5 + 0.5 * (1 - abs(pos - 0.5) * 2)  # Higher alpha in the middle
            node_colors.append(f'rgba(100, 100, 255, {alpha})')
            node_sizes.append(5 + 10 * alpha)
        
        # Add flow nodes
        x_coords = [pos[0] for pos in positions]
        y_coords = [pos[1] for pos in positions]
        
        fig.add_trace(go.Scatter(
            x=x_coords, y=y_coords,
            mode='markers',
            marker=dict(
                size=node_sizes,
                color=node_colors,
                symbol='circle'
            ),
            hoverinfo='none',
            showlegend=False
        ))
    
    # Add title and model information
    fig.add_annotation(
        x=x_center, y=height - 30,
        text=f"<b>Dynamic Transformer Architecture:</b> {model_name}",
        showarrow=False,
        font=dict(size=20, color='black')
    )
    
    fig.add_annotation(
        x=x_center, y=height - 60,
        text=f"Layers: {num_layers} | Attention Heads: {num_heads} | Hidden Size: {hidden_size}",
        showarrow=False,
        font=dict(size=14, color='darkblue')
    )
    
    # Define base component sizes
    component_width = 260
    component_height = 60
    
    # Input layer (embeddings + positional encoding)
    input_y = height - layer_spacing
    add_component(
        x_center, input_y, 
        component_width, component_height,
        colors['input'], 
        "Input Embeddings", 
        f"<b>Input Embeddings + Positional Encoding</b><br>Dimension: {hidden_size}"
    )
    
    # Create smaller components to represent sub-elements of input
    token_embedding_x = x_center - 60
    pos_encoding_x = x_center + 60
    
    # Token embeddings
    add_component(
        token_embedding_x, input_y - 20, 
        90, 30,
        'rgba(100, 149, 237, 0.8)', 
        "Token Embeddings", 
        "Token Embeddings: Convert tokens to vectors"
    )
    
    # Positional encodings
    add_component(
        pos_encoding_x, input_y - 20, 
        90, 30,
        'rgba(65, 105, 225, 0.8)', 
        "Positional Encoding", 
        "Positional Encoding: Add position information"
    )
    
    # Initialize previous layer y-coordinate
    prev_layer_y = input_y
    layer_width = 300
    
    # Add each transformer layer
    for layer_idx in range(num_layers):
        # Y position for this layer
        layer_y = input_y - (layer_idx + 1) * layer_spacing
        
        # Main layer background
        add_component(
            x_center, layer_y,
            layer_width, component_height + 40,
            'rgba(240, 240, 240, 0.5)', 
            f"Layer {layer_idx+1}", 
            f"<b>Transformer Layer {layer_idx+1}</b>"
        )
        
        # Layer number indicator
        fig.add_annotation(
            x=x_center - layer_width/2 + 30, y=layer_y + 30,
            text=f"Layer {layer_idx+1}",
            showarrow=False,
            font=dict(size=12, color='black')
        )
        
        # Add Layer Norm before attention
        norm1_x = x_center - 110
        norm1_y = layer_y + 25
        add_component(
            norm1_x, norm1_y,
            20, 50,
            colors['norm'], 
            "Layer Norm 1", 
            "Layer Normalization"
        )
        
        # Add Layer Norm before FFN
        norm2_x = x_center + 20
        norm2_y = layer_y + 25
        add_component(
            norm2_x, norm2_y,
            20, 50,
            colors['norm'], 
            "Layer Norm 2", 
            "Layer Normalization"
        )
        
        # Multi-head attention block
        attn_x = x_center - 70
        attn_y = layer_y
        add_component(
            attn_x, attn_y,
            120, component_height,
            colors['attention'], 
            f"Multi-Head Attention {layer_idx+1}", 
            f"<b>Multi-Head Attention</b><br>Heads: {num_heads}"
        )
        
        # Add individual attention heads in a circular pattern
        head_radius = 35
        for head_idx in range(min(num_heads, 8)):  # Show up to 8 heads for clarity
            angle = 2 * np.pi * head_idx / min(num_heads, 8)
            head_x = attn_x + head_radius * np.cos(angle)
            head_y = attn_y + head_radius * np.sin(angle)
            
            # Each attention head is a small circle
            fig.add_trace(go.Scatter(
                x=[head_x], y=[head_y],
                mode='markers',
                marker=dict(
                    size=14,
                    color=colors['attention_head'],
                    line=dict(color='white', width=1)
                ),
                hoverinfo='text',
                hovertext=f"Attention Head {head_idx+1}<br>Query, Key, Value projections",
                showlegend=False
            ))
        
        # Feed-forward network
        ffn_x = x_center + 70
        ffn_y = layer_y
        add_component(
            ffn_x, ffn_y,
            120, component_height,
            colors['ffn'], 
            f"Feed-Forward Network {layer_idx+1}", 
            f"<b>Feed-Forward Network</b><br>Activation: {model_info['activation']}"
        )
        
        # Add connections
        # Connection from previous layer to current layer
        add_connection(x_center, prev_layer_y, x_center, layer_y + component_height/2 + 20, colors['connection'])
        
        # Residual connection (skipping the whole layer)
        add_connection(
            x_center, prev_layer_y,
            x_center - 150, prev_layer_y - 20,
            colors['connection'], 1.5, 'dash'
        )
        add_connection(
            x_center - 150, prev_layer_y - 20,
            x_center - 150, layer_y - 20,
            colors['connection'], 1.5, 'dash'
        )
        add_connection(
            x_center - 150, layer_y - 20,
            x_center, layer_y - 50,
            colors['connection'], 1.5, 'dash'
        )
        
        # Add animated data flow nodes along the connections
        # Main path
        add_animated_nodes(
            [x_center, x_center],
            [prev_layer_y, layer_y + component_height/2 + 20],
            num_nodes=7
        )
        
        # Residual path
        add_animated_nodes(
            [x_center, x_center - 150],
            [prev_layer_y, prev_layer_y - 20],
            num_nodes=4,
            color='rgba(255, 100, 100, 0.7)'
        )
        add_animated_nodes(
            [x_center - 150, x_center - 150],
            [prev_layer_y - 20, layer_y - 20],
            num_nodes=5,
            color='rgba(255, 100, 100, 0.7)'
        )
        add_animated_nodes(
            [x_center - 150, x_center],
            [layer_y - 20, layer_y - 50],
            num_nodes=4,
            color='rgba(255, 100, 100, 0.7)'
        )
        
        # Connections between components within the layer
        # Norm1 to Attention
        add_connection(norm1_x + 10, norm1_y, attn_x - 60, attn_y, colors['connection'])
        
        # Attention to Norm2
        add_connection(attn_x + 60, attn_y, norm2_x - 10, norm2_y, colors['connection'])
        
        # Norm2 to FFN
        add_connection(norm2_x + 10, norm2_y, ffn_x - 60, ffn_y, colors['connection'])
        
        # Update previous layer y for next iteration
        prev_layer_y = layer_y
    
    # Output layer
    output_y = prev_layer_y - layer_spacing
    add_component(
        x_center, output_y,
        component_width, component_height,
        colors['output'], 
        "Output", 
        f"<b>Output Representations</b><br>Dimension: {hidden_size}"
    )
    
    # Connection from last layer to output
    add_connection(x_center, prev_layer_y, x_center, output_y + component_height/2, colors['connection'])
    
    # Add animated nodes for the final connection
    add_animated_nodes(
        [x_center, x_center],
        [prev_layer_y, output_y + component_height/2],
        num_nodes=7
    )
    
    # Add legend items with custom markers
    legend_items = [
        ("Input Embeddings", colors['input']),
        ("Multi-Head Attention", colors['attention']),
        ("Feed-Forward Network", colors['ffn']),
        ("Layer Normalization", colors['norm']),
        ("Output Layer", colors['output']),
        ("Residual Connection", colors['connection']),
    ]
    
    for i, (name, color) in enumerate(legend_items):
        marker_type = 'line' if name == "Residual Connection" else 'square'
        
        if marker_type == 'square':
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(size=15, color=color, symbol='square'),
                name=name,
                showlegend=True
            ))
        else:
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[i, i],
                mode='lines',
                line=dict(color=color, width=2, dash='dash' if name == "Residual Connection" else 'solid'),
                name=name,
                showlegend=True
            ))
    
    # Update layout
    fig.update_layout(
        width=width,
        height=height,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.1,
            xanchor="center",
            x=0.5,
            bgcolor='rgba(255,255,255,0.8)'
        ),
        margin=dict(l=20, r=20, t=80, b=80),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[0, width]
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[0, height]
        ),
        # Add hover mode and animations
        hovermode='closest',
        # Create animation frames for a more dynamic effect
        updatemenus=[{
            'type': 'buttons',
            'showactive': False,
            'buttons': [{
                'label': 'Play Animation',
                'method': 'animate',
                'args': [None, {
                    'frame': {'duration': 500, 'redraw': True},
                    'fromcurrent': True,
                    'transition': {'duration': 300, 'easing': 'quadratic-in-out'}
                }]
            }],
            'x': 0.1,
            'y': 0.05
        }]
    )
    
    # Add visual effects with annotations and shapes
    # Energy glow around attention heads
    for i in range(10):
        size = 50 + i * 3
        opacity = 0.2 - i * 0.02
        if opacity > 0:
            fig.add_shape(
                type="circle",
                xref="x", yref="y",
                x0=attn_x - size/2, y0=attn_y - size/2,
                x1=attn_x + size/2, y1=attn_y + size/2,
                fillcolor=f'rgba(255, 100, 100, {opacity})',
                line_width=0
            )
    
    # Add watermark
    fig.add_annotation(
        x=0.5, y=0.02,
        xref="paper", yref="paper",
        text="Transformer Architecture Visualizer",
        showarrow=False,
        font=dict(size=12, color="gray"),
        opacity=0.7
    )
    
    # Add dynamic frames to enable animation when hovering
    frames = []
    for i in range(5):
        # Create slightly varied positions for animated elements
        frame_data = []
        
        # For each trace in the figure, create a frame with modified data
        for trace in fig.data:
            # Only animate the marker traces (flow nodes)
            if trace.mode == 'markers' and 'showlegend' in trace and not trace.showlegend:
                # Get the original data
                x = trace.x
                y = trace.y
                
                # If this is a trace with multiple points (flow nodes)
                if len(x) > 1:
                    # Create new positions with small random movements
                    new_x = [xi + random.uniform(-5, 5) for xi in x]
                    new_y = [yi + random.uniform(-5, 5) for yi in y]
                    
                    # Create a new trace with updated positions
                    frame_data.append(go.Scatter(
                        x=new_x, y=new_y,
                        mode='markers',
                        marker=trace.marker,
                        showlegend=False
                    ))
                else:
                    # Keep static elements the same
                    frame_data.append(trace)
            else:
                # Keep non-animated traces the same
                frame_data.append(trace)
        
        # Add frame
        frames.append(go.Frame(data=frame_data, name=f"frame{i}"))
    
    # Add frames to figure
    fig.frames = frames
    
    return fig

def create_attention_heatmap(attention_data, layer_idx=0, head_idx=0, tokens=None):
    """
    Create an attention heatmap for a specific layer and attention head.
    
    Args:
        attention_data (dict): Attention data from the model
        layer_idx (int): Layer index to visualize
        head_idx (int): Attention head index to visualize
        tokens (list): List of tokens corresponding to the attention matrix
        
    Returns:
        go.Figure: Plotly figure with attention heatmap
    """
    layer_key = f"layer_{layer_idx}"
    head_key = f"head_{head_idx}"
    
    if layer_key not in attention_data["attention_matrices"]:
        raise ValueError(f"Layer {layer_idx} does not exist in the attention data")
    
    if head_key not in attention_data["attention_matrices"][layer_key]:
        raise ValueError(f"Head {head_idx} does not exist in layer {layer_idx}")
    
    attention_matrix = attention_data["attention_matrices"][layer_key][head_key]
    
    # Get tokens if not provided
    if tokens is None:
        tokens = attention_data["tokens"]
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=attention_matrix,
        x=tokens,
        y=tokens,
        colorscale='Viridis',
        hoverongaps=False
    ))
    
    fig.update_layout(
        title=f"Attention Weights - Layer {layer_idx+1}, Head {head_idx+1}",
        xaxis_title="Token (Key)",
        yaxis_title="Token (Query)",
        width=600,
        height=600
    )
    
    return fig

def visualize_token_embeddings(embedding_data, method='pca', perplexity=30):
    """
    Visualize token embeddings using dimensionality reduction.
    
    Args:
        embedding_data (dict): Embedding data from the model
        method (str): Dimensionality reduction method ('pca' or 'tsne')
        perplexity (int): Perplexity parameter for t-SNE (only used if method='tsne')
        
    Returns:
        go.Figure: Plotly figure with token embedding visualization
    """
    tokens = embedding_data["tokens"]
    # Convert dictionary to array
    embeddings = np.array([embedding_data["token_embeddings"][token] for token in tokens])
    
    # Apply dimensionality reduction
    if method.lower() == 'pca':
        reducer = PCA(n_components=2)
        title_method = "PCA"
    else:  # tsne
        reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        title_method = f"t-SNE (perplexity={perplexity})"
    
    reduced_embeddings = reducer.fit_transform(embeddings)
    
    # Create scatter plot
    fig = go.Figure(data=go.Scatter(
        x=reduced_embeddings[:, 0],
        y=reduced_embeddings[:, 1],
        mode='markers+text',
        text=tokens,
        textposition="top center",
        marker=dict(
            size=10,
            color=np.arange(len(tokens)),
            colorscale='Viridis',
            showscale=True
        )
    ))
    
    fig.update_layout(
        title=f"Token Embeddings - {title_method}",
        xaxis_title=f"{title_method} Component 1",
        yaxis_title=f"{title_method} Component 2",
        width=800,
        height=600
    )
    
    return fig

def create_attention_flow_visualization(attention_data, input_text, layer_idx=0):
    """
    Create a visualization showing attention flow between tokens.
    
    Args:
        attention_data (dict): Attention data from the model
        input_text (str): Original input text
        layer_idx (int): Layer index to visualize
        
    Returns:
        go.Figure: Plotly figure with attention flow visualization
    """
    layer_key = f"layer_{layer_idx}"
    if layer_key not in attention_data["attention_matrices"]:
        raise ValueError(f"Layer {layer_idx} does not exist in the attention data")
    
    tokens = attention_data["tokens"]
    
    # Average the attention across heads
    head_keys = list(attention_data["attention_matrices"][layer_key].keys())
    attention_matrices = [attention_data["attention_matrices"][layer_key][head] for head in head_keys]
    avg_attention = np.mean(attention_matrices, axis=0)
    
    # Create a sankey diagram
    source = []
    target = []
    value = []
    threshold = 0.05  # Only show connections with attention > threshold
    
    for i, token_from in enumerate(tokens):
        for j, token_to in enumerate(tokens):
            if i != j and avg_attention[i, j] > threshold:
                source.append(i)
                target.append(j)
                value.append(float(avg_attention[i, j]))
    
    # If there are no connections above threshold, use the top 10 connections
    if len(source) == 0:
        flat_indices = np.argsort(avg_attention.flatten())[::-1]
        for idx in flat_indices[:10]:
            i, j = np.unravel_index(idx, avg_attention.shape)
            if i != j:  # Skip self-attention
                source.append(i)
                target.append(j)
                value.append(float(avg_attention[i, j]))
    
    # Create labels with token text
    node_labels = [f"{i}: {token}" for i, token in enumerate(tokens)]
    
    # Create Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=node_labels,
            color="blue"
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            color=[f"rgba(100, 100, 255, {v*0.8+0.2})" for v in value]
        )
    )])
    
    fig.update_layout(
        title_text=f"Attention Flow - Layer {layer_idx+1}",
        font_size=10,
        width=800,
        height=800
    )
    
    return fig

def visualize_hidden_states_across_layers(hidden_states_data, token_idx=0):
    """
    Visualize how hidden states for a token change across layers.
    
    Args:
        hidden_states_data (dict): Hidden states data from the model
        token_idx (int): Index of the token to visualize
        
    Returns:
        go.Figure: Plotly figure with hidden states across layers
    """
    tokens = hidden_states_data["tokens"]
    if token_idx >= len(tokens):
        raise ValueError(f"Token index {token_idx} is out of bounds (max: {len(tokens)-1})")
    
    token = tokens[token_idx]
    
    # Get hidden states for each layer for the selected token
    layer_names = []
    hidden_data = []
    
    for layer_name, layer_data in hidden_states_data["hidden_states"].items():
        layer_names.append(layer_name)
        token_hidden = layer_data[token_idx]
        hidden_data.append(token_hidden)
    
    # Apply PCA to reduce dimensions for visualization
    hidden_data_array = np.array(hidden_data)
    
    # If we have more than 2 layers, use PCA to show trajectory
    if len(hidden_data) > 2:
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(hidden_data_array)
        
        # Create scatter plot with lines connecting points
        fig = go.Figure()
        
        # Add lines
        fig.add_trace(go.Scatter(
            x=reduced_data[:, 0],
            y=reduced_data[:, 1],
            mode='lines',
            line=dict(color='rgba(50, 50, 180, 0.6)', width=2),
            showlegend=False
        ))
        
        # Add points
        fig.add_trace(go.Scatter(
            x=reduced_data[:, 0],
            y=reduced_data[:, 1],
            mode='markers+text',
            text=layer_names,
            textposition="top center",
            marker=dict(
                size=10,
                color=np.arange(len(layer_names)),
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Layer")
            ),
            showlegend=False
        ))
        
        fig.update_layout(
            title=f"Hidden State Progression Across Layers for Token: '{token}'",
            xaxis_title="PCA Component 1",
            yaxis_title="PCA Component 2",
            width=700,
            height=500
        )
    else:
        # If we don't have enough layers, show the raw data
        fig = go.Figure()
        
        for i, (layer_name, hidden) in enumerate(zip(layer_names, hidden_data)):
            fig.add_trace(go.Bar(
                y=hidden[:50],  # Show first 50 dimensions
                name=layer_name
            ))
        
        fig.update_layout(
            title=f"Hidden State Values for Token: '{token}'",
            xaxis_title="Hidden Dimension (first 50)",
            yaxis_title="Activation Value",
            width=800,
            height=400,
            barmode='group'
        )
    
    return fig

def create_self_attention_mechanism_diagram(tokens=None):
    """
    Create a diagram explaining the self-attention mechanism.
    
    Args:
        tokens (list, optional): Actual tokens from the input text. If None, uses a default example.
        
    Returns:
        go.Figure: Plotly figure with self-attention mechanism diagram
    """
    # Create figure
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Query-Key Attention", "Value Weighting", "Output"),
        specs=[[{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}]]
    )
    
    # Define a simple example if no tokens provided
    if tokens is None or len(tokens) < 3:
        # Default example
        tokens = ["The", "cat", "sits"]
        is_example = True
    else:
        # Use first 3 tokens from input or a reasonable subset
        tokens = tokens[:3]
        is_example = False
    
    # Create sample query, key, and value representations
    queries = np.array([
        [0.8, 0.2],
        [0.3, 0.7],
        [0.5, 0.5]
    ])
    
    keys = np.array([
        [0.9, 0.1],
        [0.2, 0.8],
        [0.6, 0.4]
    ])
    
    values = np.array([
        [0.7, 0.3],
        [0.4, 0.6],
        [0.5, 0.5]
    ])
    
    # Calculate attention scores (just for this example)
    attention_scores = np.array([
        [0.9, 0.1, 0.3],
        [0.1, 0.8, 0.2],
        [0.3, 0.2, 0.7]
    ])
    
    # Illustration 1: Query-Key pairs
    # Show vectors and connections between tokens
    for i, token in enumerate(tokens):
        # Query vector
        fig.add_trace(
            go.Scatter(
                x=[0, queries[i, 0]],
                y=[0, queries[i, 1]],
                mode="lines+markers",
                name=f"{token} Query",
                line=dict(color="blue", width=2),
                showlegend=(i == 0)
            ),
            row=1, col=1
        )
        
        # Key vector
        fig.add_trace(
            go.Scatter(
                x=[2, 2 + keys[i, 0]],
                y=[0, keys[i, 1]],
                mode="lines+markers",
                name=f"{token} Key",
                line=dict(color="red", width=2),
                showlegend=(i == 0)
            ),
            row=1, col=1
        )
        
        # Token labels
        fig.add_annotation(
            x=0,
            y=i*0.3 - 0.6,
            text=token,
            showarrow=False,
            row=1, col=1
        )
        fig.add_annotation(
            x=2,
            y=i*0.3 - 0.6,
            text=token,
            showarrow=False,
            row=1, col=1
        )
    
    # Illustration 2: Value weighting
    x_positions = [0, 1, 2]
    
    for i, token in enumerate(tokens):
        # Value bars
        fig.add_trace(
            go.Bar(
                x=[f"{token} dim1", f"{token} dim2"],
                y=[values[i, 0], values[i, 1]],
                name=f"{token} Values",
                marker_color=["rgba(0, 100, 255, 0.6)", "rgba(0, 100, 255, 0.6)"],
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Attention weights visualization
        for j, target in enumerate(tokens):
            fig.add_annotation(
                x=i*2,
                y=-0.2 - j*0.2,
                text=f"{target}: {attention_scores[i, j]:.2f}",
                showarrow=False,
                font=dict(
                    size=10,
                    color=f"rgba(0, 0, 0, {min(1, attention_scores[i, j] + 0.3)})"
                ),
                row=1, col=2
            )
    
    # Illustration 3: Output representation
    # Simplified output representation
    output_vectors = np.array([
        [0.75, 0.25],
        [0.35, 0.65],
        [0.55, 0.45]
    ])
    
    for i, token in enumerate(tokens):
        # Output vector
        fig.add_trace(
            go.Scatter(
                x=[0, output_vectors[i, 0]],
                y=[0, output_vectors[i, 1]],
                mode="lines+markers",
                name=f"{token} Output",
                line=dict(color="green", width=2),
                showlegend=(i == 0)
            ),
            row=1, col=3
        )
        
        # Token labels
        fig.add_annotation(
            x=0,
            y=i*0.3 - 0.6,
            text=token,
            showarrow=False,
            row=1, col=3
        )
    
    # Add a note about whether this is the actual tokens or an example
    title_text = "Self-Attention Mechanism"
    if is_example:
        title_text = "Self-Attention Mechanism (Illustrative Example)"
        fig.add_annotation(
            x=0.5, y=-0.15,
            xref="paper", yref="paper",
            text="Note: This is a simplified example to illustrate the mechanism, not your actual input",
            showarrow=False,
            font=dict(size=12, color="red"),
        )
    
    # Update layout
    fig.update_layout(
        title_text=title_text,
        height=400,
        width=1200
    )
    
    fig.update_xaxes(range=[-0.5, 3], row=1, col=1)
    fig.update_yaxes(range=[-1, 1], row=1, col=1)
    
    fig.update_xaxes(title_text="Values", row=1, col=2)
    fig.update_yaxes(title_text="Magnitude", row=1, col=2)
    
    fig.update_xaxes(range=[-0.5, 1.5], row=1, col=3)
    fig.update_yaxes(range=[-1, 1], row=1, col=3)
    
    return fig 