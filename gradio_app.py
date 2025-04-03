import gradio as gr
import torch
import numpy as np
import plotly.io as pio
from transformer_model import TransformerVisualizer
import visualization_utils as viz

# Configure plotly for better display in Gradio
pio.templates.default = "plotly_white"

# Cache the visualizer to avoid reloading the model repeatedly
visualizer_cache = {}

def get_visualizer(model_name):
    """Get or create a cached visualizer instance for the given model name."""
    if model_name not in visualizer_cache:
        try:
            visualizer_cache[model_name] = TransformerVisualizer(model_name=model_name)
            print(f"Model {model_name} loaded and cached")
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            raise gr.Error(f"Failed to load model {model_name}. Error: {str(e)}")
    return visualizer_cache[model_name]

def process_text(text, model_name, progress=gr.Progress()):
    """Process text and generate all visualizations with progress updates."""
    try:
        # Get visualizer (from cache if possible)
        progress(0.1, desc="Loading model...")
        visualizer = get_visualizer(model_name)
        
        # Process text through model
        progress(0.2, desc="Processing text...")
        processed_data = visualizer.process_text(text)
        
        # Get model information
        progress(0.3, desc="Getting model info...")
        model_info = visualizer.get_model_architecture()
        model_info_str = f"""
        Model: {model_info['model_name']}
        Layers: {model_info['num_layers']}
        Attention Heads: {model_info['num_heads']}
        Hidden Size: {model_info['hidden_size']}
        Vocabulary Size: {model_info['vocab_size']}
        Activation: {model_info['activation']}
        """
        
        # Extract visualization data
        progress(0.4, desc="Preparing visualizations...")
        embedding_data = visualizer.get_embedding_visualization_data(processed_data)
        attention_data = visualizer.get_attention_visualization_data(processed_data)
        hidden_states_data = visualizer.get_hidden_state_visualization_data(processed_data)
        
        # Create all visualizations
        progress(0.5, desc="Creating architecture diagram...")
        architecture_fig = viz.create_transformer_architecture_diagram(model_info)
        architecture_fig.update_layout(
            height=600,
            width=800,
            title_font_size=20,
            font_size=14
        )
        
        progress(0.6, desc="Creating attention visualizations...")
        attention_fig = viz.create_attention_heatmap(attention_data, layer_idx=0, head_idx=0)
        attention_fig.update_layout(height=500, width=700)
        
        flow_fig = viz.create_attention_flow_visualization(attention_data, text, layer_idx=0)
        flow_fig.update_layout(height=600, width=800)
        
        progress(0.7, desc="Creating token embedding visualization...")
        embedding_fig = viz.visualize_token_embeddings(embedding_data, method='pca')
        embedding_fig.update_layout(height=500, width=700)
        
        progress(0.8, desc="Creating hidden states visualization...")
        hidden_states_fig = viz.visualize_hidden_states_across_layers(hidden_states_data, token_idx=0)
        hidden_states_fig.update_layout(height=500, width=700)
        
        progress(0.9, desc="Creating self-attention explanation...")
        attention_mechanism_fig = viz.create_self_attention_mechanism_diagram(tokens=processed_data["tokens"])
        attention_mechanism_fig.update_layout(height=500, width=900)
        
        # Set up data for interactive components
        progress(0.95, desc="Finalizing...")
        layer_count = len(attention_data["attention_matrices"])
        head_count = len(next(iter(attention_data["attention_matrices"].values())))
        token_count = len(hidden_states_data["tokens"])
        
        # Return all visualizations and data
        progress(1.0, desc="Complete!")
        return {
            "tokens": processed_data["tokens"],
            "model_info": model_info_str,
            "architecture_fig": architecture_fig,
            "attention_fig": attention_fig,
            "flow_fig": flow_fig,
            "embedding_fig": embedding_fig,
            "hidden_states_fig": hidden_states_fig,
            "attention_mechanism_fig": attention_mechanism_fig,
            "layer_count": layer_count,
            "head_count": head_count,
            "token_count": token_count
        }
    except Exception as e:
        import traceback
        print(f"Error in process_text: {str(e)}")
        print(traceback.format_exc())
        raise gr.Error(f"Error processing text: {str(e)}")

def update_attention_viz(text, model_name, layer_idx, head_idx):
    """Update attention visualizations for a specific layer and head."""
    try:
        visualizer = get_visualizer(model_name)
        processed_data = visualizer.process_text(text)
        attention_data = visualizer.get_attention_visualization_data(processed_data)
        
        # Create attention heatmap for selected layer and head
        attention_fig = viz.create_attention_heatmap(attention_data, layer_idx=int(layer_idx), head_idx=int(head_idx))
        attention_fig.update_layout(height=500, width=700)
        
        # Create attention flow visualization for selected layer
        flow_fig = viz.create_attention_flow_visualization(attention_data, text, layer_idx=int(layer_idx))
        flow_fig.update_layout(height=600, width=800)
        
        return attention_fig, flow_fig
    except Exception as e:
        print(f"Error in update_attention_viz: {str(e)}")
        raise gr.Error(f"Error updating attention visualization: {str(e)}")

def update_embedding_viz(text, model_name, method, perplexity=30):
    """Update token embedding visualization with selected method."""
    try:
        visualizer = get_visualizer(model_name)
        processed_data = visualizer.process_text(text)
        embedding_data = visualizer.get_embedding_visualization_data(processed_data)
        
        # Create embedding visualization with selected method
        embedding_fig = viz.visualize_token_embeddings(
            embedding_data, 
            method=method.lower(), 
            perplexity=int(perplexity)
        )
        embedding_fig.update_layout(height=500, width=700)
        
        return embedding_fig
    except Exception as e:
        print(f"Error in update_embedding_viz: {str(e)}")
        raise gr.Error(f"Error updating embedding visualization: {str(e)}")

def update_hidden_states_viz(text, model_name, token_idx):
    """Update hidden states visualization for a specific token."""
    try:
        visualizer = get_visualizer(model_name)
        processed_data = visualizer.process_text(text)
        hidden_states_data = visualizer.get_hidden_state_visualization_data(processed_data)
        
        # Convert token index to integer
        token_idx = int(token_idx)
        
        # Get token info
        tokens = hidden_states_data["tokens"]
        if 0 <= token_idx < len(tokens):
            selected_token = tokens[token_idx]
            token_info = f"Token: '{selected_token}' (index {token_idx})"
        else:
            token_info = "Invalid token index"
            token_idx = 0  # Fallback to first token
        
        # Create hidden states visualization
        hidden_states_fig = viz.visualize_hidden_states_across_layers(hidden_states_data, token_idx=token_idx)
        hidden_states_fig.update_layout(height=500, width=700)
        
        return hidden_states_fig, token_info
    except Exception as e:
        print(f"Error in update_hidden_states_viz: {str(e)}")
        raise gr.Error(f"Error updating hidden states visualization: {str(e)}")

def initialize_interface(initial_results=None):
    """Update sliders and other UI elements based on the current model results."""
    if initial_results is None:
        return (
            gr.update(maximum=2, value=0),  # layer_slider
            gr.update(maximum=4, value=0),  # head_slider
            gr.update(maximum=10, value=0)   # token_slider
        )
    else:
        return (
            gr.update(maximum=initial_results["layer_count"]-1, value=0),
            gr.update(maximum=initial_results["head_count"]-1, value=0),
            gr.update(maximum=initial_results["token_count"]-1, value=0)
        )

# Define CSS for better styling
css = """
.container {
    max-width: 1200px;
    margin: auto;
}
.title {
    text-align: center;
    margin-bottom: 1em;
}
.figure-container {
    background-color: #f8f9fa;
    border-radius: 10px;
    padding: 20px;
    margin-top: 10px;
}
.explanation {
    background-color: #f0f0f0;
    border-left: 5px solid #4c72b0;
    padding: 10px 15px;
    margin: 10px 0;
    border-radius: 0 5px 5px 0;
}
.control-panel {
    background-color: #f8f8f8;
    padding: 10px;
    border-radius: 10px;
    margin-bottom: 15px;
}
"""

# Define the Gradio interface
with gr.Blocks(title="Transformer Visualizer", theme=gr.themes.Soft(), css=css) as app:
    # Initialize state for storing results between tab changes
    results_state = gr.State(None)
    
    # Header
    with gr.Row(elem_classes=["title"]):
        gr.Markdown("# Transformer Architecture Visualizer")
    
    # Input Panel
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Column(elem_classes=["control-panel"]):
                gr.Markdown("### Input Controls")
                model_dropdown = gr.Dropdown(
                    choices=[
                        "prajjwal1/bert-tiny",
                        "prajjwal1/bert-mini",
                        "prajjwal1/bert-small"
                    ],
                    value="prajjwal1/bert-tiny",
                    label="Select Model",
                )
                
                text_input = gr.Textbox(
                    value="Transformers are powerful neural networks for NLP tasks.",
                    label="Input Text",
                    lines=3
                )
                
                process_btn = gr.Button("Process Text", variant="primary")
                
        with gr.Column(scale=1):
            with gr.Column(elem_classes=["figure-container"]):
                gr.Markdown("### Model Information")
                model_info_output = gr.Textbox(
                    label="Model Details",
                    lines=8,
                    interactive=False
                )
                tokens_output = gr.JSON(
                    label="Processed Tokens"
                )
    
    # Visualization Tabs
    with gr.Tabs() as tabs:
        # Tab 1: Architecture
        with gr.TabItem("Architecture"):
            with gr.Column(elem_classes=["figure-container"]):
                architecture_output = gr.Plot(
                    label="Transformer Architecture Diagram",
                    show_label=True,
                    visible=True
                )
            
            with gr.Column(elem_classes=["explanation"]):
                gr.Markdown("""
                **Architecture Components:**
                
                - **Input Layer**: Embeddings + Positional Encoding
                - **Multi-Head Attention**: Multiple attention mechanisms running in parallel
                - **Feed-Forward Networks**: Point-wise fully connected layers
                - **Layer Normalization**: Stabilizes training
                - **Residual Connections**: Allow gradients to flow directly
                """)
        
        # Tab 2: Attention Visualization
        with gr.TabItem("Attention Visualization"):
            with gr.Column(elem_classes=["control-panel"]):
                with gr.Row():
                    layer_slider = gr.Slider(
                        minimum=0, maximum=2, step=1, value=0,
                        label="Layer"
                    )
                    head_slider = gr.Slider(
                        minimum=0, maximum=4, step=1, value=0,
                        label="Attention Head"
                    )
                
                update_attention_btn = gr.Button("Update Visualization")
            
            with gr.Row():
                with gr.Column():
                    with gr.Column(elem_classes=["figure-container"]):
                        gr.Markdown("### Attention Heatmap")
                        attention_output = gr.Plot()
                
                with gr.Column():
                    with gr.Column(elem_classes=["figure-container"]):
                        gr.Markdown("### Attention Flow")
                        attention_flow_output = gr.Plot()
            
            with gr.Column(elem_classes=["explanation"]):
                gr.Markdown("""
                **Attention Visualizations:**
                
                - **Heatmap**: Shows how much each token (vertical axis) attends to other tokens (horizontal axis)
                - **Flow Diagram**: Shows connections between tokens based on attention strength
                
                Different attention heads often specialize in different patterns: some focus on adjacent words, 
                others on syntactic or semantic relationships across longer distances.
                """)
        
        # Tab 3: Token Embeddings
        with gr.TabItem("Token Embeddings"):
            with gr.Column(elem_classes=["control-panel"]):
                with gr.Row():
                    method_radio = gr.Radio(
                        choices=["PCA", "t-SNE"],
                        value="PCA",
                        label="Dimensionality Reduction Method"
                    )
                    perplexity_slider = gr.Slider(
                        minimum=5, maximum=50, step=1, value=30,
                        label="t-SNE Perplexity",
                        visible=False
                    )
                
                update_embedding_btn = gr.Button("Update Visualization")
            
            with gr.Column(elem_classes=["figure-container"]):
                gr.Markdown("### Token Embeddings")
                embedding_output = gr.Plot()
            
            with gr.Column(elem_classes=["explanation"]):
                gr.Markdown("""
                **Token Embeddings:**
                
                Token embeddings are dense vector representations of words/subwords in high-dimensional space.
                This visualization reduces dimensions to 2D for visualization:
                
                - **PCA**: Preserves global structure and variance
                - **t-SNE**: Better preserves local relationships between tokens
                
                Similar tokens tend to cluster together in the embedding space.
                """)
        
        # Tab 4: Hidden States
        with gr.TabItem("Hidden States"):
            with gr.Column(elem_classes=["control-panel"]):
                with gr.Row():
                    token_slider = gr.Slider(
                        minimum=0, maximum=10, step=1, value=0,
                        label="Token Index"
                    )
                    token_info = gr.Textbox(
                        label="Selected Token",
                        interactive=False
                    )
                
                update_hidden_btn = gr.Button("Update Visualization")
            
            with gr.Column(elem_classes=["figure-container"]):
                gr.Markdown("### Hidden States Transformation")
                hidden_states_output = gr.Plot()
            
            with gr.Column(elem_classes=["explanation"]):
                gr.Markdown("""
                **Hidden States Visualization:**
                
                This shows how a token's representation evolves through the network:
                
                - Each point represents the hidden state at a specific layer
                - The progression shows how token representations transform through the model
                - Earlier layers capture more syntactic features
                - Deeper layers develop more contextual semantic representations
                """)
        
        # Tab 5: Self-Attention Explained
        with gr.TabItem("Self-Attention Explained"):
            with gr.Column(elem_classes=["figure-container"]):
                attention_mechanism_output = gr.Plot()
            
            with gr.Column(elem_classes=["explanation"]):
                gr.Markdown("""
                **Self-Attention Mechanism:**
                
                Self-attention allows each position to attend to all positions in the sequence:
                
                1. **Query-Key Pairs**: The attention score between tokens is calculated as the dot product of Query and Key vectors
                2. **Value Weighting**: Attention scores are normalized using softmax, then used to weight Value vectors
                3. **Output Representations**: The weighted values are summed to produce the output for each position
                
                In multi-head attention, this process happens in parallel across multiple heads, allowing the model to attend to different aspects simultaneously.
                """)
    
    # Event handlers
    
    # Process text button handler (with progress bar)
    def handle_process_text(text, model_name, progress=gr.Progress()):
        results = process_text(text, model_name, progress)
        
        # Update the sliders based on model
        layer_count = results["layer_count"]
        head_count = results["head_count"]
        token_count = results["token_count"]
        
        # Return all results in the same order as the outputs list
        return (
            results,  # results_state
            results["tokens"],  # tokens_output
            results["model_info"],  # model_info_output
            results["architecture_fig"],  # architecture_output
            results["attention_fig"],  # attention_output
            results["flow_fig"],  # attention_flow_output
            results["embedding_fig"],  # embedding_output
            results["hidden_states_fig"],  # hidden_states_output
            results["attention_mechanism_fig"],  # attention_mechanism_output
            gr.update(maximum=layer_count-1, value=0),  # layer_slider
            gr.update(maximum=head_count-1, value=0),  # head_slider
            gr.update(maximum=token_count-1, value=0),  # token_slider
            f"Token: '{results['tokens'][0]}' (index 0)"  # token_info
        )
    
    process_btn.click(
        handle_process_text,
        inputs=[text_input, model_dropdown],
        outputs=[
            results_state,
            tokens_output,
            model_info_output,
            architecture_output,
            attention_output,
            attention_flow_output,
            embedding_output,
            hidden_states_output,
            attention_mechanism_output,
            layer_slider,
            head_slider,
            token_slider,
            token_info
        ]
    )
    
    # Update attention visualization handler
    update_attention_btn.click(
        update_attention_viz,
        inputs=[text_input, model_dropdown, layer_slider, head_slider],
        outputs=[attention_output, attention_flow_output]
    )
    
    # Update embedding visualization handler
    update_embedding_btn.click(
        update_embedding_viz,
        inputs=[text_input, model_dropdown, method_radio, perplexity_slider],
        outputs=[embedding_output]
    )
    
    # Update hidden states visualization handler
    update_hidden_btn.click(
        update_hidden_states_viz,
        inputs=[text_input, model_dropdown, token_slider],
        outputs=[hidden_states_output, token_info]
    )
    
    # Show/hide perplexity slider based on method selection
    method_radio.change(
        lambda method: gr.update(visible=(method == "t-SNE")),
        inputs=[method_radio],
        outputs=[perplexity_slider]
    )

# Launch the app
if __name__ == "__main__":
    # Set up initial visualization on startup
    try:
        default_text = "Transformers are powerful neural networks for NLP tasks."
        default_model = "prajjwal1/bert-tiny"
        initial_results = process_text(default_text, default_model)
        print("Pre-loaded visualizations with default text and model")
        
        # Launch with pre-loaded visualizations
        app.launch(
            prevent_thread_lock=True,
            # Initialize UI with computed visualizations
            _js=f"""
            function() {{
                // Set the initial values
                document.addEventListener('DOMContentLoaded', function() {{
                    // Trigger a click on the process button after the UI loads
                    setTimeout(function() {{
                        document.querySelector('button.primary').click();
                    }}, 1000);
                }});
            }}
            """
        )
    except Exception as e:
        print(f"Warning: Could not pre-load visualizations: {str(e)}")
        # Launch without pre-loaded visualizations
        app.launch()
else:
    app.launch() 