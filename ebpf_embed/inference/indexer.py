"""
ebpf_embed.inference.indexer
----------------------------
Builds and saves a vector index of all eBPF programs in the dataset.

Usage:
    python3 -m ebpf_embed.inference.indexer

Output:
    models/index.npz      — numpy array of all embeddings [N, 512]
    models/index_meta.json — filenames, summaries, and source metadata
"""

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
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR      = os.path.join(BASE_DIR, "ebpf_embed", "data")
SUMMARIES_FILE = os.path.join(DATA_DIR, "summaries.json")
METADATA_FILE  = os.path.join(DATA_DIR, "dataset_metadata.json")
MODEL_DIR     = os.path.join(BASE_DIR, "models")
CHECKPOINT    = os.path.join(MODEL_DIR, "fusion_best.pt")
INDEX_NPZ     = os.path.join(MODEL_DIR, "index.npz")
INDEX_META    = os.path.join(MODEL_DIR, "index_meta.json")
# ──────────────────────────────────────────────────────────────────────────────


def build_index():
    # ── Device ────────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Building index on: {device}")

    # ── Load models ───────────────────────────────────────────────────────────
    struct_enc = StructuralEncoder(device=device)
    sem_enc    = SemanticEncoder(device=device)
    fusion     = CrossAttentionFusion().to(device)

    if not os.path.exists(CHECKPOINT):
        raise FileNotFoundError(
            f"Trained checkpoint not found at {CHECKPOINT}.\n"
            "Run training first: python3 -m ebpf_embed.training.train"
        )
    fusion.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    fusion.eval()
    print(f"Loaded checkpoint: {CHECKPOINT}")

    # ── Load data ─────────────────────────────────────────────────────────────
    with open(SUMMARIES_FILE) as f:
        summaries = json.load(f)

    # Load optional metadata for richer search results
    # dataset_metadata.json is a dict keyed by filename: { "foo.o": { "source_repo": ..., ... } }
    metadata_map = {}
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE) as f:
            metadata_map = json.load(f)   # already { filename: {meta} }

    # Only index files that have valid summaries
    valid_files = [
        fname for fname in os.listdir(DATA_DIR)
        if fname.endswith(".o")
        and fname in summaries
        and not summaries[fname].startswith("Error")
    ]
    valid_files.sort()
    print(f"Indexing {len(valid_files)} programs...")

    # ── Extractors ────────────────────────────────────────────────────────────
    extractor  = FCFGExtractor()
    tokenizer  = EBPFTokenizer()
    serializer = FCFGSerializer()

    embeddings = []
    index_meta = []
    skipped    = 0

    for i, fname in enumerate(valid_files):
        fpath   = os.path.join(DATA_DIR, fname)
        summary = summaries[fname]

        try:
            disasm       = extractor.get_disassembly(fpath)
            instrs       = extractor.parse_instructions(disasm)
            blocks       = extractor.find_basic_blocks(instrs)
            graph        = extractor.build_graph(blocks)
            graph        = tokenizer.annotate_graph(graph)
            asm_dict_list = serializer.serialize_to_dict_list(graph)

            if not asm_dict_list or not asm_dict_list[0]:
                skipped += 1
                continue

            with torch.no_grad():
                e_struct = struct_enc.embed(asm_dict_list).to(device)
                e_sem    = sem_enc.embed(summary).to(device)
                e_fused  = fusion(e_struct, e_sem)               # [1, 512]

                # L2 normalise so cosine similarity = dot product
                e_norm = F.normalize(e_fused, dim=-1)

            embeddings.append(e_norm.squeeze(0).cpu().numpy())

            # Enrich metadata from dataset_metadata.json if available
            meta = metadata_map.get(fname, {})
            index_meta.append({
                "filename":    fname,
                "summary":     summary,
                "source_repo": meta.get("source_repo", "unknown"),
                "prog_type":   meta.get("program_type", "unknown"),
                "category":    meta.get("industry_category", "unknown"),
                "file_size":   os.path.getsize(fpath),
            })

            if (i + 1) % 10 == 0 or (i + 1) == len(valid_files):
                print(f"  [{i+1}/{len(valid_files)}] {fname}")

        except Exception as e:
            print(f"  ✗ Skipping {fname}: {type(e).__name__}: {e}")
            skipped += 1
            continue

    # ── Save ──────────────────────────────────────────────────────────────────
    emb_matrix = np.stack(embeddings, axis=0)          # [N, 512]
    np.savez_compressed(INDEX_NPZ, embeddings=emb_matrix)

    with open(INDEX_META, "w") as f:
        json.dump(index_meta, f, indent=2)

    print(f"\n✓ Index built: {len(embeddings)} programs indexed, {skipped} skipped.")
    print(f"  Embeddings: {INDEX_NPZ}  shape={emb_matrix.shape}")
    print(f"  Metadata:   {INDEX_META}")


if __name__ == "__main__":
    build_index()
