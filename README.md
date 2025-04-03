# Transformer Architecture Visualizer

A minimalistic visualization tool that helps you understand how transformer models work under the hood. This project provides clear visualizations of transformer components, attention mechanisms, and token representations using a lightweight BERT model from Hugging Face.

## Features

- **Architecture Diagram**: Visual representation of the transformer architecture
- **Attention Visualization**: Heatmaps and flow diagrams showing how attention works across layers and heads
- **Token Embedding Explorer**: See how tokens are represented in embedding space
- **Hidden State Trajectory**: Track how token representations evolve through the transformer layers
- **Self-Attention Explainer**: Conceptual explanation of the self-attention mechanism

## Project Structure

The project consists of three Python files:

1. `transformer_model.py` - Core file that handles model loading and data processing
2. `visualization_utils.py` - Visualization utilities and graphing functions
3. `gradio_app.py` - Gradio application for interactive visualization

## Installation

1. Clone the repository:
```bash
git clone https://github.com/rushitgit/TransformerArchitecture.git
(make a venv- optional)
```


2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Gradio app:
```bash
python gradio_app.py
```

2. Open your browser and navigate to the displayed URL (typically http://127.0.0.1:7860)

3. Select a model from the dropdown, enter text to analyze, and click "Process Text"

4. Explore the different visualization tabs:
   - Architecture: View the transformer architecture diagram
   - Attention Visualization: Explore attention patterns between tokens
   - Token Embeddings: See how tokens are represented in vector space
   - Hidden States: Track how token representations evolve through layers
   - Self-Attention Explained: Understand the self-attention mechanism

## How It Works

This application uses a lightweight BERT model to demonstrate the inner workings of transformers:

1. **Model Loading**: Loads a pre-trained BERT model from Hugging Face
2. **Text Processing**: Tokenizes input text and runs it through the model
3. **Data Extraction**: Extracts attention matrices, embeddings, and hidden states
4. **Visualization**: Creates interactive visualizations using Plotly
5. **User Interface**: Presents the visualizations in a Gradio web application

## Acknowledgements

- Hugging Face for providing access to transformer models
- The "Attention Is All You Need" paper by Vaswani et al.
- Gradio for the interactive web application framework
- Plotly for the visualization capabilities

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
