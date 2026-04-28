"""
ebpf_embed.cli
--------------
Command-line interface for eBPF-Zemo.

Commands:
    embed      <file.o>              — generate a 512-dim embedding
    similarity <file1.o> <file2.o>  — cosine similarity between two programs
    summary    <file.o>             — show cached LLM summary
    search     <file.o> [-k N]      — find top-K similar programs in the index
    index                           — (re)build the vector index
"""

import click
import os
import json
import numpy as np
import torch
import torch.nn.functional as F

from ebpf_embed.extractor.fcfg import FCFGExtractor
from ebpf_embed.extractor.tokenizer import EBPFTokenizer
from ebpf_embed.extractor.serializer import FCFGSerializer
from ebpf_embed.encoder.structural import StructuralEncoder
from ebpf_embed.encoder.semantic import SemanticEncoder
from ebpf_embed.encoder.fusion import CrossAttentionFusion

# ── Paths ─────────────────────────────────────────────────────────────────────
_BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_DIR       = os.path.join(_BASE_DIR, "ebpf_embed", "data")
_SUMMARIES_FILE = os.path.join(_DATA_DIR, "summaries.json")
_CHECKPOINT     = os.path.join(_BASE_DIR, "models", "fusion_best.pt")
_INDEX_NPZ      = os.path.join(_BASE_DIR, "models", "index.npz")
_INDEX_META     = os.path.join(_BASE_DIR, "models", "index_meta.json")
# ──────────────────────────────────────────────────────────────────────────────


def _auto_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class EBPFEmbedder:
    """Loads the full pipeline (encoders + trained fusion) and produces embeddings."""

    def __init__(self):
        device_str = _auto_device()
        self.device = torch.device(device_str)

        self.extractor  = FCFGExtractor()
        self.tokenizer  = EBPFTokenizer()
        self.serializer = FCFGSerializer()

        self.struct_enc = StructuralEncoder(device=device_str)
        self.sem_enc    = SemanticEncoder(device=device_str)
        self.fusion     = CrossAttentionFusion().to(self.device)

        # Load trained weights
        if os.path.exists(_CHECKPOINT):
            self.fusion.load_state_dict(
                torch.load(_CHECKPOINT, map_location=self.device)
            )
            click.echo(f"[✓] Loaded checkpoint: {_CHECKPOINT}")
        else:
            click.echo(
                f"[!] Warning: no checkpoint at {_CHECKPOINT}. "
                "Embeddings will use random fusion weights.\n"
                "    Run: python3 -m ebpf_embed.training.train"
            )

        self.fusion.eval()

        # Load summaries cache
        self._summaries = {}
        if os.path.exists(_SUMMARIES_FILE):
            with open(_SUMMARIES_FILE) as f:
                self._summaries = json.load(f)

    def get_embedding(self, file_path: str) -> torch.Tensor:
        """
        Returns a normalized [1, 512] embedding tensor for the given .o file.
        Uses the cached LLM summary if available, otherwise a generic fallback.
        """
        fname   = os.path.basename(file_path)
        summary = self._summaries.get(fname, "eBPF program bytecode")

        disasm        = self.extractor.get_disassembly(file_path)
        instrs        = self.extractor.parse_instructions(disasm)
        blocks        = self.extractor.find_basic_blocks(instrs)
        graph         = self.extractor.build_graph(blocks)
        graph         = self.tokenizer.annotate_graph(graph)
        asm_dict_list = self.serializer.serialize_to_dict_list(graph)

        with torch.no_grad():
            e_struct = self.struct_enc.embed(asm_dict_list).to(self.device)
            e_sem    = self.sem_enc.embed(summary).to(self.device)
            e_fused  = self.fusion(e_struct, e_sem)
            e_norm   = F.normalize(e_fused, dim=-1)   # [1, 512]

        return e_norm


# ── Path resolver ─────────────────────────────────────────────────────────────
def resolve_path(file_path: str) -> str:
    if os.path.exists(file_path):
        return file_path
    candidate = os.path.join(_DATA_DIR, os.path.basename(file_path))
    if os.path.exists(candidate):
        return candidate
    available = sorted(f for f in os.listdir(_DATA_DIR) if f.endswith(".o"))
    raise click.BadParameter(
        f"'{file_path}' not found.\nAvailable programs:\n  " +
        "\n  ".join(available[:20]) +
        ("\n  ..." if len(available) > 20 else "")
    )


# ── CLI ───────────────────────────────────────────────────────────────────────
@click.group()
def main():
    """eBPF-Zemo: Semantic Bytecode Embedding Engine"""
    pass


@main.command()
@click.argument("file_path")
def embed(file_path):
    """Generate a normalized 512-dim fused embedding for an eBPF object."""
    file_path = resolve_path(file_path)
    embedder  = EBPFEmbedder()
    emb       = embedder.get_embedding(file_path)
    click.echo(f"\nFile:      {os.path.basename(file_path)}")
    click.echo(f"Dimension: {emb.shape[-1]}")
    click.echo(f"Vector[:8]: {emb[0][:8].tolist()}")
    click.echo(f"L2 norm:   {emb.norm().item():.6f}  (should be ≈1.0)")


@main.command()
@click.argument("file1")
@click.argument("file2")
def similarity(file1, file2):
    """Calculate cosine similarity between two eBPF object files."""
    file1 = resolve_path(file1)
    file2 = resolve_path(file2)

    embedder = EBPFEmbedder()
    emb1 = embedder.get_embedding(file1)   # [1, 512]
    emb2 = embedder.get_embedding(file2)   # [1, 512]

    # Both are L2-normalised, so dot product == cosine similarity
    sim = (emb1 * emb2).sum().item()

    click.echo(f"\n  {os.path.basename(file1)}")
    click.echo(f"  {os.path.basename(file2)}")
    click.echo(f"\n  Cosine similarity: {sim:.4f}  {'★ very similar' if sim > 0.9 else '~ somewhat similar' if sim > 0.7 else '○ dissimilar'}")


@main.command()
@click.argument("file_path")
def summary(file_path):
    """Show the cached LLM-generated semantic summary for an eBPF object."""
    file_path = resolve_path(file_path)
    fname     = os.path.basename(file_path)

    if not os.path.exists(_SUMMARIES_FILE):
        click.echo("Summaries cache not found. Run generate_summaries first.")
        return

    with open(_SUMMARIES_FILE) as f:
        summaries = json.load(f)

    click.echo(f"\n{fname}:")
    click.echo(summaries.get(fname, "No summary found in cache."))


@main.command()
@click.argument("file_path")
@click.option("-k", "--top-k", default=5, show_default=True,
              help="Number of similar programs to return.")
@click.option("--min-sim", default=0.0, show_default=True,
              help="Minimum cosine similarity threshold.")
def search(file_path, top_k, min_sim):
    """
    Search the index for the top-K most similar eBPF programs.

    Requires a pre-built index. Build it with:
        python3 -m ebpf_embed.inference.indexer
    """
    # ── Check index exists ────────────────────────────────────────────────────
    if not os.path.exists(_INDEX_NPZ) or not os.path.exists(_INDEX_META):
        click.echo(
            "Index not found. Build it first:\n"
            "  python3 -m ebpf_embed.inference.indexer"
        )
        return

    # ── Load index ────────────────────────────────────────────────────────────
    data       = np.load(_INDEX_NPZ)
    emb_matrix = data["embeddings"]            # [N, 512]  float32, L2-normalised

    with open(_INDEX_META) as f:
        meta = json.load(f)

    click.echo(f"\nIndex loaded: {len(meta)} programs")

    # ── Embed query ───────────────────────────────────────────────────────────
    file_path = resolve_path(file_path)
    embedder  = EBPFEmbedder()
    query_emb = embedder.get_embedding(file_path)          # [1, 512]
    query_np  = query_emb.squeeze(0).cpu().numpy()         # [512]

    # ── Cosine similarity (= dot product since both are L2-normalised) ────────
    scores   = emb_matrix @ query_np                       # [N]
    top_idx  = np.argsort(scores)[::-1]                    # descending

    # ── Display results ───────────────────────────────────────────────────────
    query_name = os.path.basename(file_path)
    click.echo(f"\nQuery: {query_name}")
    click.echo("─" * 60)
    click.echo(f"{'Rank':<5} {'Score':<8} {'File':<35} {'Source'}")
    click.echo("─" * 60)

    shown = 0
    for rank, idx in enumerate(top_idx):
        # Skip the query file itself if it's in the index
        if meta[idx]["filename"] == query_name:
            continue
        score = scores[idx]
        if score < min_sim:
            break
        m = meta[idx]
        click.echo(
            f"{shown+1:<5} {score:.4f}   {m['filename']:<35} {m.get('source_repo', 'unknown')}"
        )
        click.echo(f"       ↳ {m['summary'][:90]}...")
        click.echo()
        shown += 1
        if shown >= top_k:
            break

    if shown == 0:
        click.echo(f"No results above similarity threshold {min_sim}.")


@main.command("index")
def build_index_cmd():
    """(Re)build the vector index from all dataset .o files."""
    from ebpf_embed.inference.indexer import build_index
    build_index()


if __name__ == "__main__":
    main()
