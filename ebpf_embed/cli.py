import click
import os
import json
import torch
from ebpf_embed.extractor.fcfg import FCFGExtractor
from ebpf_embed.extractor.tokenizer import EBPFTokenizer
from ebpf_embed.extractor.serializer import FCFGSerializer
from ebpf_embed.encoder.structural import StructuralEncoder
from ebpf_embed.encoder.semantic import SemanticEncoder
from ebpf_embed.encoder.fusion import CrossAttentionFusion

class EBPFEmbedder:
    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        self.extractor = FCFGExtractor()
        self.tokenizer = EBPFTokenizer()
        self.serializer = FCFGSerializer()
        
        self.struct_enc = StructuralEncoder(device=device)
        self.sem_enc = SemanticEncoder(device=device)
        self.fusion = CrossAttentionFusion().to(device)
        
        # In a real scenario, we would load trained weights here
        # self.fusion.load_state_dict(torch.load("models/fusion.pt"))

    def get_embedding(self, file_path, summary_text=None):
        """Generates a fused 512-dim embedding."""
        # 1. Structural
        disasm = self.extractor.get_disassembly(file_path)
        instrs = self.extractor.parse_instructions(disasm)
        blocks = self.extractor.find_basic_blocks(instrs)
        graph = self.extractor.build_graph(blocks)
        graph = self.tokenizer.annotate_graph(graph)
        asm_dict_list = self.serializer.serialize_to_dict_list(graph)
        e_struct = self.struct_enc.embed(asm_dict_list)
        
        # 2. Semantic (use placeholder if no summary provided)
        if not summary_text:
            summary_text = "eBPF program bytecode"
        e_sem = self.sem_enc.embed(summary_text)
        
        # 3. Fusion
        with torch.no_grad():
            e_fused = self.fusion(e_struct, e_sem)
        
        return e_fused

def resolve_path(file_path):
    """Smartly resolves a file path, checking locally and then in the package data directory."""
    # 1. Check if it exists as is
    if os.path.exists(file_path):
        return file_path
    
    # 2. Check if it exists in the package's data directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "ebpf_embed", "data")
    
    # Try just the filename if a path was provided
    filename = os.path.basename(file_path)
    combined_path = os.path.join(data_dir, filename)
    
    if os.path.exists(combined_path):
        return combined_path
        
    # 3. If still not found, check if 'data/filename' was provided and strip 'data/'
    if file_path.startswith("data/"):
        stripped_path = os.path.join(data_dir, file_path[5:])
        if os.path.exists(stripped_path):
            return stripped_path

    # If really not found, raise a helpful error
    available = [f for f in os.listdir(data_dir) if f.endswith(".o")]
    error_msg = f"Path '{file_path}' does not exist and was not found in the data directory.\n"
    error_msg += f"Available programs: {', '.join(available)}"
    raise click.BadParameter(error_msg)

@click.group()
def main():
    """eBPF-Embed: Semantic Bytecode Embedding Engine"""
    pass

@main.command()
@click.argument('file_path')
def embed(file_path):
    """Generate a 512-dim fused embedding for an eBPF object."""
    file_path = resolve_path(file_path)
    embedder = EBPFEmbedder()
    emb = embedder.get_embedding(file_path)
    click.echo(f"Embedding generated for {os.path.basename(file_path)}")
    click.echo(f"Vector (first 5): {emb[0][:5].tolist()}")
    click.echo(f"Dimension: {emb.shape[1]}")

@main.command()
@click.argument('file1')
@click.argument('file2')
def similarity(file1, file2):
    """Calculate cosine similarity between two eBPF objects."""
    file1 = resolve_path(file1)
    file2 = resolve_path(file2)
    embedder = EBPFEmbedder()
    emb1 = embedder.get_embedding(file1)
    emb2 = embedder.get_embedding(file2)
    
    sim = torch.nn.functional.cosine_similarity(emb1, emb2)
    click.echo(f"Similarity between {os.path.basename(file1)} and {os.path.basename(file2)}: {sim.item():.4f}")

@main.command()
@click.argument('file_path')
def summary(file_path):
    """Show the cached semantic summary for an eBPF object."""
    file_path = resolve_path(file_path)
    # Ensure we find the summaries file regardless of where the command is run from
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    summaries_path = os.path.join(base_dir, "ebpf_embed", "data", "summaries.json")
    filename = os.path.basename(file_path)
    
    if os.path.exists(summaries_path):
        with open(summaries_path, 'r') as f:
            summaries = json.load(f)
        click.echo(f"Summary for {filename}:")
        click.echo(summaries.get(filename, "No summary found in cache."))
    else:
        click.echo("Summaries cache not found. Run generate_summaries.py first.")

if __name__ == "__main__":
    main()
