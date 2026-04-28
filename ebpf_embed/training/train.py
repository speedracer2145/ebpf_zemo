import torch
import torch.nn as nn
import torch.nn.functional as F
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
        
        # Filter out files whose summaries contain errors from the generation step
        self.files = [
            f for f in os.listdir(data_dir)
            if f.endswith(".o")
            and f in self.summaries
            and not self.summaries[f].startswith("Error")
        ]
        print(f"Dataset: {len(self.files)} valid files loaded (skipped files with error summaries).")
        
        self.extractor = FCFGExtractor()
        self.tokenizer = EBPFTokenizer()
        self.serializer = FCFGSerializer()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]
        file_path = os.path.join(self.data_dir, filename)
        summary = self.summaries[filename]
        
        try:
            disasm = self.extractor.get_disassembly(file_path)
            instrs = self.extractor.parse_instructions(disasm)
            blocks = self.extractor.find_basic_blocks(instrs)
            graph = self.extractor.build_graph(blocks)
            graph = self.tokenizer.annotate_graph(graph)
            asm_dict_list = self.serializer.serialize_to_dict_list(graph)
            
            if not asm_dict_list or not asm_dict_list[0]:
                return None
                
            return asm_dict_list, summary
        except Exception as e:
            print(f"Warning: Skipping {filename} — {type(e).__name__}: {e}")
            return None


def normalized_contrastive_loss(queries, keys, temperature=0.1):
    """
    InfoNCE loss with L2-normalized embeddings and cosine similarity.
    
    Normalizing before the dot product ensures logits stay in [-1/t, 1/t],
    preventing the exploding logit problem that caused near-zero and spike losses.
    Temperature=0.1 is gentler than 0.07 for small batch sizes.
    """
    # L2 normalize — this constrains dot products to cosine similarity in [-1, 1]
    queries = F.normalize(queries, dim=-1)
    keys    = F.normalize(keys,    dim=-1)
    
    # Scaled cosine similarity matrix [N, N]
    logits = torch.matmul(queries, keys.T) / temperature
    
    # Diagonal entries are the correct (query_i, key_i) pairs
    labels = torch.arange(queries.shape[0], device=queries.device)
    
    # Symmetric loss: both directions
    loss_q2k = nn.CrossEntropyLoss()(logits,   labels)
    loss_k2q = nn.CrossEntropyLoss()(logits.T, labels)
    return (loss_q2k + loss_k2q) / 2.0



def ebpf_collate_fn(batch):
    """Custom collate — returns plain list to handle variable-length asm dicts.
    Also filters out None entries from __getitem__ failures."""
    batch = [item for item in batch if item is not None]
    if not batch:
        return None, None
    asm_dicts = [item[0] for item in batch]
    summaries = [item[1] for item in batch]
    return asm_dicts, summaries


def train(
    epochs=30,
    batch_size=2,
    lr=1e-4,
    checkpoint_dir="models",
    resume_from_checkpoint=False,
    temperature=0.1,
    unfreeze_encoder_layers=False
):
    # Auto-detect best available device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 1. Models
    struct_enc = StructuralEncoder(device=device)
    sem_enc    = SemanticEncoder(device=device)
    fusion     = CrossAttentionFusion().to(device)

    # Partially unfreeze encoders — only their final Linear/LayerNorm layers
    # This gives the model more capacity without the cost of full fine-tuning
    trainable_params = list(fusion.parameters())
    if unfreeze_encoder_layers:
        for enc_model in [struct_enc.model, sem_enc.model]:
            for name, param in enc_model.named_parameters():
                # Only unfreeze the last transformer layer and pooler
                if any(k in name for k in ["pooler", "11.", "10.", "layer_norm", "LayerNorm"]):
                    param.requires_grad = True
                    trainable_params.append(param)
                else:
                    param.requires_grad = False
        print(f"Trainable parameters: fusion + encoder final layers ({sum(p.numel() for p in trainable_params):,} params)")
    
    # 2. Data
    dataset    = EBPFDataset()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=ebpf_collate_fn)
    
    # 3. Optimizer — gentler LR scheduler: eta_min = 10% of lr (not 1%)
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.1   # floor at 10% not 1%
    )


    
    best_loss   = float("inf")
    start_epoch = 0
    history     = []
    
    # Resume full training state if requested
    state_ckpt = os.path.join(checkpoint_dir, "training_state.pt")
    if resume_from_checkpoint and os.path.exists(state_ckpt):
        state = torch.load(state_ckpt, map_location=device)
        fusion.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        best_loss   = state["best_loss"]
        start_epoch = state["epoch"]
        history     = state.get("history", [])
        print(f"Resumed from epoch {start_epoch} | best loss so far: {best_loss:.4f}")
    else:
        print("Starting fresh training run.")
    
    print(f"\nTraining Loop: {len(dataset)} samples | {len(dataloader)} batches/epoch | {epochs} epochs")
    print(f"Config: batch_size={batch_size} | lr={lr} | temperature={temperature}\n")
    
    for epoch in range(start_epoch, start_epoch + epochs):
        epoch_loss    = 0
        valid_batches = 0
        
        for batch_idx, (asm_dicts, summaries) in enumerate(dataloader):
            if asm_dicts is None:
                continue
            
            # --- Encode ---
            e_struct_list = [struct_enc.embed(item).to(device) for item in asm_dicts]
            e_struct = torch.cat(e_struct_list, dim=0)          # [N, D_struct]
            
            e_sem_list = [sem_enc.embed(item).to(device) for item in summaries]
            e_sem = torch.cat(e_sem_list, dim=0)                # [N, D_sem]
            
            # --- Fusion (cross-attention) ---
            e_fused = fusion(e_struct, e_sem)                   # [N, D_fused]
            
            # --- Project to shared embedding space ---
            # NOTE: fusion.forward() already calls proj_struct and proj_sem internally.
            # We call them again here explicitly so we can use them in a separate
            # direct loss term. The extra forward pass is cheap (single Linear layer).
            proj_struct = fusion.proj_struct(e_struct)   # [N, 512]
            proj_sem    = fusion.proj_sem(e_sem)         # [N, 512]
            
            # --- Dual contrastive loss ---
            #
            # loss_direct: aligns proj_struct with proj_sem directly.
            #   - Fast convergence signal. Trains proj_struct and proj_sem weights.
            #   - The cross-attention gets gradient here ONLY through e_fused (see below).
            #
            # loss_fused:  aligns e_fused (struct after attending to sem) with proj_sem.
            #   - This is the key fix: e_fused flows through the cross-attention,
            #     LayerNorm, and residual connection, so ALL fusion parameters
            #     receive gradients for the first time.
            #   - Semantics: "the structurally-attended representation should look
            #     like the semantic embedding" — which is exactly the alignment goal.
            #
            loss_direct = normalized_contrastive_loss(proj_struct, proj_sem, temperature)
            loss_fused  = normalized_contrastive_loss(e_fused,     proj_sem, temperature)
            loss = 0.5 * loss_direct + 0.5 * loss_fused
            
            # --- Backward ---
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping — prevents exploding gradients
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss    += loss.item()
            valid_batches += 1
            
            if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == len(dataloader):
                current_lr = scheduler.get_last_lr()[0]
                print(f"  Epoch {epoch+1}/{start_epoch+epochs} | "
                      f"Batch {batch_idx+1}/{len(dataloader)} | "
                      f"Loss: {loss.item():.4f} "
                      f"(direct={loss_direct.item():.4f}, fused={loss_fused.item():.4f}) | "
                      f"LR: {current_lr:.2e}")
        
        scheduler.step()
        
        avg_loss = epoch_loss / max(valid_batches, 1)
        history.append(avg_loss)
        print(f"Epoch {epoch+1} complete — Avg Loss: {avg_loss:.4f}\n")
        
        # Save full training state
        torch.save({
            "epoch":      epoch + 1,
            "model":      fusion.state_dict(),
            "optimizer":  optimizer.state_dict(),
            "scheduler":  scheduler.state_dict(),
            "best_loss":  best_loss,
            "history":    history,
        }, os.path.join(checkpoint_dir, "training_state.pt"))
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(checkpoint_dir, "fusion_best.pt")
            torch.save(fusion.state_dict(), best_path)
            print(f"  ✓ New best model saved (loss={best_loss:.4f})\n")
    
    print(f"Training complete! Best avg loss: {best_loss:.4f}")
    print(f"Loss history: {[f'{l:.4f}' for l in history]}")

if __name__ == "__main__":
    train()
