import torch
from transformers import AutoModel, AutoTokenizer

class StructuralEncoder:
    def __init__(self, model_name="Hustcw/clap-asm", device="cpu"):
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "mps")
        print(f"Loading Structural Encoder ({model_name}) on {self.device}...")
        
        # Monkey-patching transformers to fix CLAP-ASM compatibility with newer versions
        import transformers.modeling_utils
        if not hasattr(transformers.modeling_utils.PreTrainedModel, "all_tied_weights_keys"):
            # Use a property with a setter to handle both access and assignment
            def get_tied(self): return getattr(self, "_all_tied_weights_keys_patch", {})
            def set_tied(self, val): setattr(self, "_all_tied_weights_keys_patch", val if isinstance(val, dict) else {})
            transformers.modeling_utils.PreTrainedModel.all_tied_weights_keys = property(get_tied, set_tied)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(self.device)
        self.model.eval()

    def embed(self, asm_dict_list):
        """Generates a 768-dim embedding for an FCFG dict list."""
        # asm_dict_list is [{"1": "mov ebx, 1", ...}]
        inputs = self.tokenizer(
            asm_dict_list, 
            padding=True, 
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # CLAP-ASM might return a direct tensor or a ModelOutput object
            if torch.is_tensor(outputs):
                embeddings = outputs
            elif hasattr(outputs, 'last_hidden_state'):
                embeddings = outputs.last_hidden_state.mean(dim=1)
            else:
                # Fallback for pooled output if available
                embeddings = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs[0]
        
        return embeddings # Shape: [1, 768]

if __name__ == "__main__":
    # Test with the dict format
    test_input = [
        {
            "1": "w2 = *(u32 *)(r1 + 0x4)",
            "2": "w1 = *(u32 *)(r1 + 0x0)",
            "3": "r3 = r1",
            "4": "r3 += 0xe",
            "5": "if r3 > r2 goto +0x1a"
        }
    ]
    encoder = StructuralEncoder()
    emb = encoder.embed(test_input)
    print(f"Embedding shape: {emb.shape}")
    print(f"Embedding snippet: {emb[0][:5]}")
