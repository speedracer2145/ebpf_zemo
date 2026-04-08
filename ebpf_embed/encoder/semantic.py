from sentence_transformers import SentenceTransformer
import torch

class SemanticEncoder:
    def __init__(self, model_name="all-MiniLM-L6-v2", device="cpu"):
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "mps")
        print(f"Loading Semantic Encoder ({model_name}) on {self.device}...")
        
        self.model = SentenceTransformer(model_name, device=self.device)

    def embed(self, summary_text):
        """Generates a 384-dim embedding for a natural language summary."""
        # sentence-transformers.encode handles tokenization and pooling internally
        embedding = self.model.encode(summary_text, convert_to_tensor=True)
        
        # Ensure it has a batch dimension [1, 384]
        if len(embedding.shape) == 1:
            embedding = embedding.unsqueeze(0)
            
        return embedding

if __name__ == "__main__":
    test_summary = "This eBPF program is an XDP hook that filters incoming IPv4 traffic and updates a map with source port metrics."
    encoder = SemanticEncoder()
    emb = encoder.embed(test_summary)
    print(f"Embedding shape: {emb.shape}")
    print(f"Embedding snippet: {emb[0][:5]}")
