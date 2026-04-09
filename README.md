# eBPF-Zemo: Semantic Embedding Engine for eBPF Bytecode

eBPF-Zemo (or eBPF-Embed) is a research-grade tool designed to generate high-dimensional embeddings from eBPF object files. By combining structural analysis of bytecode with LLM-powered semantic summaries, it captures both the logical flow and the high-level intent of eBPF programs.

## 🚀 Key Features

- **Dual-Encoder Architecture**: 
  - **Structural Encoder**: Analyzes the Flattened Control Flow Graph (FCFG) and instruction patterns.
  - **Semantic Encoder**: Processes natural language summaries of the eBPF code (using `all-MiniLM-L6-v2`).
- **Cross-Attention Fusion**: Merges structural and semantic vectors into a unified 512-dimensional embedding.
- **Ollama Integration**: Uses LLMs (like `qwen2.5-coder`) to automatically generate human-readable summaries of eBPF bytecode.
- **Similarity Analysis**: Directly compare two eBPF programs using cosine similarity.

## 🛠️ Architecture

The system uses a sophisticated pipeline to transform raw bytecode into a semantic vector:
1. **Extractor**: Disassembles ELF files and builds a Function Control Flow Graph (FCFG).
2. **Tokenizer**: Annotates instructions with semantic tags (e.g., `MAP_LOOKUP`, `PKT_READ`).
3. **Encoders**: 
   - A structural encoder handles the graph-based features.
   - A semantic encoder handles the LLM-generated description.
4. **Fusion**: A cross-attention layer learns the interactions between structure and meaning.

## 📋 Prerequisites

- Python 3.8+
- [Ollama](https://ollama.com/) (for summary generation)
- PyTorch

## ⚙️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/ebpf_zemo.git
   cd ebpf_zemo
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 📖 Usage

### 1. Generate Summaries
Before embedding, you can generate semantic summaries for your eBPF `.o` files using Ollama:
```bash
python -m ebpf_embed.data.generate_summaries
```

### 2. Generate Embeddings
Generate a 512-dim fused embedding for a specific eBPF object file:
```bash
python -m ebpf_embed.cli embed data/your_program.o
```

### 3. Compare Programs
Calculate the cosine similarity between two eBPF programs:
```bash
python -m ebpf_embed.cli similarity data/prog1.o data/prog2.o
```

### 4. View Cached Summaries
```bash
python -m ebpf_embed.cli summary data/your_program.o
```

## 📁 Project Structure

- `ebpf_embed/extractor/`: Bytecode disassembly and FCFG generation.
- `ebpf_embed/encoder/`: Structural, Semantic, and Fusion model implementations.
- `ebpf_embed/data/`: Data management and summary generation scripts.
- `ebpf_embed/cli.py`: Primary command-line interface.

## ⚖️ License
MIT License. See [LICENSE](LICENSE) for details.
