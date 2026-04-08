import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import json
import os
from ebpf_embed.extractor.fcfg import FCFGExtractor
from ebpf_embed.extractor.tokenizer import EBPFTokenizer
from ebpf_embed.extractor.serializer import FCFGSerializer
from ebpf_embed.encoder.structural import StructuralEncoder
from ebpf_embed.encoder.semantic import SemanticEncoder
from ebpf_embed.encoder.fusion import CrossAttentionFusion

class EBPFDataset(Dataset):
    def __init__(self, data_dir="ebpf_embed/data", summaries_file="ebpf_embed/data/summaries.json"):
        self.data_dir = data_dir
        with open(summaries_file, 'r') as f:
            self.summaries = json.load(f)
        
        self.files = [f for f in os.listdir(data_dir) if f.endswith(".o") and f in self.summaries]
        
        self.extractor = FCFGExtractor()
        self.tokenizer = EBPFTokenizer()
        self.serializer = FCFGSerializer()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]
        file_path = os.path.join(self.data_dir, filename)
        summary = self.summaries[filename]
        
        # Extract and serialize to dict list for CLAP-ASM
        disasm = self.extractor.get_disassembly(file_path)
        instrs = self.extractor.parse_instructions(disasm)
        blocks = self.extractor.find_basic_blocks(instrs)
        graph = self.extractor.build_graph(blocks)
        graph = self.tokenizer.annotate_graph(graph)
        asm_dict_list = self.serializer.serialize_to_dict_list(graph)
        
        return asm_dict_list, summary

def contrastive_loss(queries, keys, temperature=0.07):
    """InfoNCE loss to align queries (struct) and keys (sem)."""
    # queries: [N, D], keys: [N, D]
    logits = torch.matmul(queries, keys.T) / temperature
    labels = torch.arange(queries.shape[0]).to(queries.device)
    return nn.CrossEntropyLoss()(logits, labels)

def train():
    device = torch.device("cpu") # For local dev
    
    # 1. Models
    struct_enc = StructuralEncoder(device=device)
    sem_enc = SemanticEncoder(device=device)
    fusion = CrossAttentionFusion().to(device)
    
    # 2. Data
    dataset = EBPFDataset()
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # 3. Optimizer
    optimizer = optim.Adam(list(fusion.parameters()), lr=1e-4) # We mainly train fusion/projections
    
    print("Starting Training Loop...")
    for epoch in range(5):
        epoch_loss = 0
        for asm_dicts, summaries in dataloader:
            # Note: DataLoader might nest our list-of-dicts, need careful handling
            # In simple batch size 2, asm_dicts might be a list of lists of dicts
            
            # Embeddings
            e_struct_list = []
            for item in asm_dicts:
                 e_struct_list.append(struct_enc.embed(item))
            e_struct = torch.cat(e_struct_list, dim=0)
            
            e_sem_list = []
            for item in summaries:
                e_sem_list.append(sem_enc.embed(item))
            e_sem = torch.cat(e_sem_list, dim=0)
            
            # Fusion
            e_fused = fusion(e_struct, e_sem)
            
            # Contrastive Objective: Align e_struct (projected) with e_sem (projected)
            # This is a simplification; in a real CLIP-like setup we'd align projections.
            # Here we'll align the fused output or the projections within the fusion layer.
            
            # For Day 3, we'll align the projected streams
            proj_struct = fusion.proj_struct(e_struct)
            proj_sem = fusion.proj_sem(e_sem)
            
            loss = contrastive_loss(proj_struct, proj_sem)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1}, Loss: {epoch_loss/len(dataloader):.4f}")

if __name__ == "__main__":
    train()
