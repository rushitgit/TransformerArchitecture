import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np

class TransformerVisualizer:
    def __init__(self, model_name="prajjwal1/bert-tiny"):
        """
        Initialize the TransformerVisualizer with a lightweight BERT model.
        
        Args:
            model_name (str): Name of the model to load from Hugging Face
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load model and tokenizer
        self.model = AutoModel.from_pretrained(model_name, output_attentions=True, 
                                              output_hidden_states=True, 
                                              attn_implementation="eager").to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Model architecture info
        self.num_layers = self.model.config.num_hidden_layers
        self.num_heads = self.model.config.num_attention_heads
        self.hidden_size = self.model.config.hidden_size
        self.vocab_size = self.model.config.vocab_size
        
        print(f"Model loaded: {model_name}")
        print(f"Layers: {self.num_layers}, Attention Heads: {self.num_heads}")
        print(f"Hidden Size: {self.hidden_size}, Vocabulary Size: {self.vocab_size}")
    
    def process_text(self, text):
        """
        Process text through the model and return embeddings, attention weights, 
        and hidden states for visualization.
        
        Args:
            text (str): Input text to process
            
        Returns:
            dict: Dictionary containing tokenized input, embeddings, attention weights, and hidden states
        """
        # Tokenize input
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        # Get model outputs
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Extract relevant information
        last_hidden_state = outputs.last_hidden_state
        hidden_states = outputs.hidden_states  # Tuple of tensors (num_layers + 1, batch_size, seq_len, hidden_size)
        attentions = outputs.attentions  # Tuple of tensors (num_layers, batch_size, num_heads, seq_len, seq_len)
        
        # Convert to numpy for easier processing
        hidden_states_np = [hs.cpu().numpy() for hs in hidden_states]
        attention_np = [att.cpu().numpy() for att in attentions]
        
        return {
            "text": text,
            "tokens": tokens,
            "token_ids": inputs['input_ids'][0].cpu().numpy(),
            "hidden_states": hidden_states_np,
            "attentions": attention_np,
            "last_hidden_state": last_hidden_state.cpu().numpy()
        }
    
    def get_model_architecture(self):
        """
        Return a dictionary with model architecture information.
        
        Returns:
            dict: Model architecture information
        """
        return {
            "model_name": self.model.config._name_or_path,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "hidden_size": self.hidden_size,
            "vocab_size": self.vocab_size,
            "activation": self.model.config.hidden_act,
            "max_position_embeddings": self.model.config.max_position_embeddings
        }
    
    def get_embedding_visualization_data(self, processed_data):
        """
        Extract embedding data suitable for visualization.
        
        Args:
            processed_data (dict): Output from process_text method
            
        Returns:
            dict: Dictionary with embedding visualization data
        """
        # Get the embedding layer output (first hidden state)
        embedding_output = processed_data["hidden_states"][0][0]  # First layer, first batch
        
        # Get token embeddings for each token
        token_embeddings = {}
        for i, token in enumerate(processed_data["tokens"]):
            token_embeddings[token] = embedding_output[i]
        
        # Get the embedding dimension
        embedding_dim = embedding_output.shape[-1]
        
        return {
            "token_embeddings": token_embeddings,
            "tokens": processed_data["tokens"],
            "embedding_dim": embedding_dim
        }
    
    def get_attention_visualization_data(self, processed_data):
        """
        Extract attention data suitable for visualization.
        
        Args:
            processed_data (dict): Output from process_text method
            
        Returns:
            dict: Dictionary with attention pattern data for visualization
        """
        attention_data = {}
        
        # For each layer
        for layer_idx, layer_attention in enumerate(processed_data["attentions"]):
            layer_attention = layer_attention[0]  # Get first batch
            
            # For each attention head
            head_data = {}
            for head_idx in range(layer_attention.shape[0]):
                head_attention = layer_attention[head_idx]
                head_data[f"head_{head_idx}"] = head_attention
            
            attention_data[f"layer_{layer_idx}"] = head_data
        
        return {
            "attention_matrices": attention_data,
            "tokens": processed_data["tokens"]
        }
    
    def get_hidden_state_visualization_data(self, processed_data):
        """
        Extract hidden state data suitable for visualization.
        
        Args:
            processed_data (dict): Output from process_text method
            
        Returns:
            dict: Dictionary with hidden state data for visualization
        """
        # Extract hidden states for each layer (including embedding layer)
        hidden_state_data = {}
        
        for layer_idx, hidden_state in enumerate(processed_data["hidden_states"]):
            # For embedding layer (idx 0) or transformer layers
            layer_name = "embedding" if layer_idx == 0 else f"layer_{layer_idx-1}"
            hidden_state_data[layer_name] = hidden_state[0]  # First batch
        
        return {
            "hidden_states": hidden_state_data,
            "tokens": processed_data["tokens"]
        }


if __name__ == "__main__":
    # Simple test
    visualizer = TransformerVisualizer()
    result = visualizer.process_text("Transformers are powerful neural networks for NLP tasks.")
    print(f"Processed {len(result['tokens'])} tokens: {result['tokens']}") 