# eBPF-Zemo: Complete Project Workflow & Status

## Project Goal

Build an **industry-grade semantic embedding engine** for eBPF bytecode that can:
1. Ingest compiled eBPF object files (`.o`) from any source
2. Understand both their **structural behavior** (assembly CFG) and **semantic intent** (natural language)
3. Fuse both signals into a single 512-dimensional embedding vector
4. Allow **semantic similarity search** — e.g., "find programs like this XDP filter"

The system is modeled after CLIP (OpenAI) but applied to systems security / eBPF programs rather than image-text pairs.

---

## Full Architecture Diagram

```
                        ┌─────────────────────────────────────────────┐
                        │           INPUT: eBPF .o file               │
                        └──────────────────┬──────────────────────────┘
                                           │
                   ┌───────────────────────┼───────────────────────────┐
                   │                       │                           │
                   ▼                       ▼                           │
        ┌─────────────────────┐  ┌──────────────────────┐             │
        │   FCFG Extraction   │  │  LLM Summary (Ollama)│             │
        │   (llvm-objdump)    │  │  qwen2.5-coder:7b    │             │
        │                     │  │                      │             │
        │  disassemble → parse│  │  Prompt: "Describe   │             │
        │  instructions →     │  │   what this eBPF     │             │
        │  basic blocks →     │  │   program does"      │             │
        │  CFG graph →        │  │                      │             │
        │  tokenize →         │  │  Cached in           │             │
        │  serialize to       │  │  summaries.json      │             │
        │  [{instr_dict}]     │  └──────────┬───────────┘             │
        └──────────┬──────────┘             │                         │
                   │                        │                         │
                   ▼                        ▼                         │
        ┌─────────────────────┐  ┌──────────────────────┐             │
        │  STRUCTURAL ENCODER │  │  SEMANTIC ENCODER    │             │
        │  Hustcw/clap-asm    │  │  all-MiniLM-L6-v2   │             │
        │  (Assembly Xformer) │  │  (Sentence-BERT)     │             │
        │                     │  │                      │             │
        │  Input: [{str:str}] │  │  Input: string       │             │
        │  Output: [1, 768]   │  │  Output: [1, 384]    │             │
        │  Frozen weights     │  │  Frozen weights      │             │
        └──────────┬──────────┘  └──────────┬───────────┘             │
                   │                        │                         │
                   └────────────┬───────────┘                         │
                                ▼                                     │
                   ┌─────────────────────────┐                        │
                   │  CrossAttentionFusion   │                        │
                   │                         │                        │
                   │  proj_struct: 768→512   │                        │
                   │  proj_sem:    384→512   │                        │
                   │  MultiheadAttention     │◄───────────────────────┘
                   │    Q = x_struct         │   (trained weights:
                   │    K = V = x_sem        │    fusion_best.pt)
                   │  Residual + LayerNorm   │
                   │  Output: [1, 512]       │
                   └────────────┬────────────┘
                                │
                    L2 Normalize → [1, 512] unit vector
                                │
              ┌─────────────────┴──────────────────┐
              │                                    │
              ▼                                    ▼
   ┌──────────────────────┐           ┌───────────────────────┐
   │  VECTOR INDEX        │           │  SIMILARITY QUERY     │
   │  models/index.npz   │◄──────────│  python3 -m           │
   │  174 × 512 float32  │  cosine   │  ebpf_embed.cli       │
   │  L2-normalized       │  dot      │  search <file.o>      │
   │  models/index_meta  │  product  │  similarity <a> <b>   │
   └──────────────────────┘           └───────────────────────┘
```

---

## Phase 1: Dataset Collection ✅ DONE

### Goal
Curate 174+ production-grade eBPF `.o` files from real industry projects.

### Sources Scraped
| Repository | Programs | Type |
|---|---|---|
| **Cilium** (cilium/cilium) | `bpf_host.o`, `bpf_lxc.o`, `bpf_xdp.o`, `bpf_overlay.o`, `bpf_sock.o`, `bpf_wireguard.o`, `bpf_lb.o`, `bpf_network.o`, `bpf_netdev.o`, `bpf_cubic.o`, `bpf_dctcp.o` | XDP, TC, Sock |
| **Katran** (facebookincubator/katran) | `xdp_root.o`, `xdp_drop_kern.o`, `xdp_filter.o` | XDP Load Balancer |
| **Linux ebpf-samples** | 80+ programs (`tracex*.o`, `sockex*.o`, `xdp1_kern.o`...) | All types |
| **libbpf-bootstrap** | `fentry.bpf.o`, `kprobe.bpf.o`, `lsm.bpf.o`, `exitsnoop.bpf.o`, `tcpconnect.bpf.o`... | Tracing, LSM |
| **BCC tools** | `cpustat_kern.o`, `offwaketime_kern.o`, `kfree_skb.o`... | Profiling |

### Scripts Built
- **`collect_dataset.py`** — automated GitHub API scraper with `.o` file detection
- **`sweep_internet.py`** — extended scraper for additional repos
- **`curate_dataset.py`** — filter pipeline: removes trivial/test/hello-world programs
- **`dataset_metadata.json`** — `{ filename: { source_repo, program_type, industry_category, sha256_hash } }`

### Result
**174 valid `.o` files** in `ebpf_embed/data/`, ranging from 448 bytes (`filter.o`) to 4.2MB (`probe.o`)

---

## Phase 2: Semantic Summarization ✅ DONE

### Goal
Generate natural language descriptions of each eBPF program to provide the semantic signal for training.

### Method
- **Model**: `qwen2.5-coder:7b` running locally via Ollama
- **Script**: `ebpf_embed/data/generate_summaries.py`
- **Prompt strategy**: Feed disassembly excerpt + ask "what does this eBPF program do?"
- **Caching**: Results persisted in `summaries.json` — never re-generated unless missing

### Result
`summaries.json` — 174 LLM-generated natural language summaries, ~67KB total

---

## Phase 3: Extraction Pipeline ✅ DONE

### Goal
Convert raw `.o` files into a structured graph representation that the assembly transformer can process.

### Pipeline (`ebpf_embed/extractor/`)
```
.o file
  → FCFGExtractor.get_disassembly()     # llvm-objdump -d → raw text
  → FCFGExtractor.parse_instructions()  # regex → list of (addr, opcode, operands)
  → FCFGExtractor.find_basic_blocks()   # split at branches/labels
  → FCFGExtractor.build_graph()         # networkx DiGraph of basic blocks
  → EBPFTokenizer.annotate_graph()      # add semantic tags (MEM_READ, MAP_LOOKUP...)
  → FCFGSerializer.serialize_to_dict_list() # → [{"1": "mov r1, r2", "2": "add r3, 4"}]
```

### Key design decision
`serialize_to_dict_list()` outputs a **list of dicts** (one per basic block) because `CLAP-ASM`'s tokenizer expects this exact format. Each key is an instruction index, each value is the instruction string.

---

## Phase 4: Encoder Architecture ✅ DONE

### Structural Encoder (`encoder/structural.py`)
- **Model**: `Hustcw/clap-asm` — a RoBERTa-style transformer pre-trained on assembly code (CLAP paper)
- **Input**: `[{str: str}]` — serialized CFG dict list
- **Output**: `[1, 768]` — 768-dim embedding
- **Note**: Weights frozen during training; only `fusion.py` parameters are trainable
- **Patch**: Required monkey-patching `all_tied_weights_keys` for newer transformers versions

### Semantic Encoder (`encoder/semantic.py`)
- **Model**: `all-MiniLM-L6-v2` — a 6-layer distilled BERT for sentence embeddings
- **Input**: Natural language summary string
- **Output**: `[1, 384]` — 384-dim embedding
- **Weights**: Frozen during training

### CrossAttentionFusion (`encoder/fusion.py`)
- **`proj_struct`**: `nn.Linear(768 → 512)` — projects structural embedding to shared space
- **`proj_sem`**: `nn.Linear(384 → 512)` — projects semantic embedding to shared space
- **`self.attn`**: `nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)`
  - Q = `x_struct` (structural after projection)
  - K = V = `x_sem` (semantic after projection)
- **Residual**: `x = x_struct + Dropout(attn_output)` → `LayerNorm`
- **Output**: `[1, 512]` — fused cross-attended embedding

---

## Phase 5: Contrastive Training ✅ DONE

### Goal
Train the fusion model so that `embed(program_X_struct) ≈ embed(program_X_sem)` — the structural and semantic embeddings of the same program should be close together.

### Training Setup (`training/train.py`)
| Hyperparameter | Value | Reason |
|---|---|---|
| Optimizer | AdamW | Weight decay regularization |
| Learning Rate | 1e-4 | Tested; converges smoothly |
| Scheduler | CosineAnnealingLR (T_max=30, eta_min=1e-5) | Gradual LR decay, floor at 10% of LR |
| Gradient Clipping | max_norm=1.0 | Prevents exploding gradients |
| Batch Size | 2 | Limited by variable-length asm dicts |
| Epochs | 30 | Full convergence achieved |
| Temperature | 0.1 | Gentler than 0.07 for small batches |
| Device | MPS (Apple Metal) | Mac M-series GPU acceleration |

### Loss Function (Final Version)
```python
def normalized_contrastive_loss(queries, keys, temperature=0.1):
    queries = F.normalize(queries, dim=-1)  # L2 normalize
    keys    = F.normalize(keys,    dim=-1)
    logits  = queries @ keys.T / temperature  # cosine similarity matrix
    labels  = torch.arange(N)                 # diagonal = positives
    loss_q2k = CrossEntropyLoss(logits, labels)
    loss_k2q = CrossEntropyLoss(logits.T, labels)
    return (loss_q2k + loss_k2q) / 2.0        # symmetric
```

### Dual Loss (Key Fix)
The original code had `e_fused` computed but never used in the loss — meaning the cross-attention layer received zero gradients. Fixed with:
```python
loss_direct = normalized_contrastive_loss(proj_struct, proj_sem)   # trains projections
loss_fused  = normalized_contrastive_loss(e_fused, proj_sem)       # trains cross-attention
loss        = 0.5 * loss_direct + 0.5 * loss_fused
```

### Training History (Bugs Fixed Along the Way)
| Issue | Root Cause | Fix |
|---|---|---|
| Loss `0.6866` plateau | LR cosine decay to near-zero | Increased `eta_min` from `lr*0.01` to `lr*0.1` |
| Resume caused loss to jump from 0.69 → 1.20 | Only model weights saved, not optimizer/scheduler state | Saved full training state dict |
| Memory bank caused loss > 2.5 | Stale random embeddings in bank = impossible task | Removed bank entirely |
| `e_fused` unused | Bug in original architecture connection | Dual loss routing |

### Final Convergence
```
Epoch  1: 0.6017
Epoch  5: 0.1588
Epoch 10: 0.0570
Epoch 15: 0.0293
Epoch 20: 0.0249
Epoch 25: 0.0159
Epoch 30: 0.0105   ← best, saved as fusion_best.pt
```
**98.3% loss reduction** from start to finish.

---

## Phase 6: Indexing ✅ DONE

### Goal
Pre-compute all 174 embeddings and store them as a compressed matrix for fast search.

### Script: `ebpf_embed/inference/indexer.py`
1. Loads trained `fusion_best.pt`
2. Runs all 174 programs through the full pipeline
3. L2-normalizes each embedding → `[1, 512]`
4. Saves to `models/index.npz` — shape `(174, 512)` float32
5. Saves `models/index_meta.json` with `{filename, summary, source_repo, prog_type, category, file_size}`

### Search at runtime
```python
scores = emb_matrix @ query_np   # (174,) — cosine similarity via dot product
```
Sub-millisecond search because embeddings are pre-computed and L2-normalized.

---

## Phase 7: CLI Interface ✅ DONE

### Package entry point: `zemo` (`pyproject.toml` → `ebpf_embed.cli:main`)

| Command | Description |
|---|---|
| `zemo search <file.o> -k 10` | Top-K similar programs from index |
| `zemo similarity <a.o> <b.o>` | Pairwise cosine similarity score |
| `zemo embed <file.o>` | Print raw 512-dim embedding vector |
| `zemo summary <file.o>` | Show cached LLM summary |
| `zemo index` | Rebuild the vector index |

### Validated Search Results
- `bpf_host.o` → `bpf_xdp.o` (0.505) — both Cilium core programs ✓
- `xdp_root.o` → returns 10 XDP hook programs ✓
- `bpf_host.o` vs `xdp_filter.o` → 0.706 "somewhat similar" ✓

---

## What Is NOT Done ❌

### 1. Formal Evaluation Metrics (`eval/metrics.py`) — EMPTY
The `eval/` directory exists but contains only `__init__.py`. Nothing is implemented.

**What's needed:**
- **Retrieval@K** — given a query program, what % of the top-K results are "correct" (same program type/source)? Requires a ground-truth label file.
- **MRR (Mean Reciprocal Rank)** — for each query, at what rank does the first correct result appear?
- **ARI / NMI clustering metrics** — cluster all 174 embeddings (e.g., k-means with k=5 for {XDP, TC, Tracing, Socket, LSM}) and measure how well clusters match ground-truth program types.
- **t-SNE / UMAP visualization** — 512-dim → 2D plot colored by `program_type`

### 2. Similarity Visualization — NOT BUILT
`umap-learn` and `matplotlib` are in `requirements.txt` but no visualization script exists.

**What's needed:**
- `eval/visualize.py` — load `index.npz`, run UMAP (512→2D), scatter plot colored by `program_type` from `dataset_metadata.json`

### 3. Absolute Similarity Scores — WEAK
Scores top out at ~0.50 between semantically related programs. A production system (CLIP, FAISS-backed) achieves 0.85-0.98 for near-identical content.

**Root cause:** Batch size of 2 during training = only 1 negative per step. The model learned correct *relative ranking* but not tight *absolute clustering*.

**What would fix it (without more data):**
- Hard negative mining: explicitly find pairs within the 174 programs that are structurally similar but semantically different, and train on those
- GPU with batch_size ≥ 32 for richer contrastive signal

### 4. Source Attribution — INCOMPLETE
Many results show `source: unknown` because `dataset_metadata.json` doesn't have entries for all 174 files (some came from outside `ebpf-samples`).

---

## Repository State (GitHub: `speedracer2145/ebpf_zemo`)

| Path | Status | Description |
|---|---|---|
| `ebpf_embed/data/*.o` (174 files) | ✅ | Dataset |
| `ebpf_embed/data/summaries.json` | ✅ | LLM summaries |
| `ebpf_embed/data/dataset_metadata.json` | ✅ | Program metadata |
| `ebpf_embed/extractor/fcfg.py` | ✅ | CFG extraction |
| `ebpf_embed/extractor/tokenizer.py` | ✅ | Instruction tokenizer |
| `ebpf_embed/extractor/serializer.py` | ✅ | Dict serializer |
| `ebpf_embed/encoder/structural.py` | ✅ | CLAP-ASM wrapper |
| `ebpf_embed/encoder/semantic.py` | ✅ | MiniLM wrapper |
| `ebpf_embed/encoder/fusion.py` | ✅ | CrossAttentionFusion |
| `ebpf_embed/training/train.py` | ✅ | Full training loop |
| `models/fusion_best.pt` | ✅ | Trained weights (loss=0.0105) |
| `models/training_state.pt` | ✅ | Full optimizer/scheduler state |
| `models/index.npz` | ✅ | 174×512 embedding matrix |
| `models/index_meta.json` | ✅ | Index metadata |
| `ebpf_embed/inference/indexer.py` | ✅ | Index builder |
| `ebpf_embed/cli.py` | ✅ | Full CLI |
| `ebpf_embed/eval/` | ❌ | **Empty** |
| `eval/metrics.py` | ❌ | **Not built** |
| `eval/visualize.py` | ❌ | **Not built** |

---

## Remaining Work (Prioritized)

```
Priority 1 — Evaluation (proves the model works rigorously)
  [ ] eval/metrics.py
      - Retrieval@K (k=1,5,10)
      - MRR
      - ARI/NMI on k-means clusters vs ground-truth program_type labels

Priority 2 — Visualization (shows clustering quality visually)
  [ ] eval/visualize.py
      - UMAP 512→2D
      - Scatter plot colored by program_type
      - Labels for notable programs (bpf_host, xdp_root, etc.)

Priority 3 — Absolute score improvement (without new data)
  [ ] Hard negative mining script
      - Find within-dataset false-positives
      - Re-train with explicit hard negatives
      - Target: push related-program similarity > 0.80
```
