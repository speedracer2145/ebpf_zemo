import networkx as nx

class FCFGSerializer:
    def serialize(self, G):
        """Converts an annotated DiGraph into a linearized string."""
        serialized_blocks = []
        
        # We sort by block index to preserve some order
        for n in sorted(G.nodes):
            block_data = G.nodes[n]
            instrs = block_data['instructions']
            annotations = block_data.get('annotations', [])
            
            instr_str = " ; ".join([f"{i['mnemonic']} {i['operands']}" for i in instrs])
            ann_str = " ".join(annotations)
            
            serialized_blocks.append(f"[BB{n}] {instr_str} {ann_str}")
        
        return " ".join(serialized_blocks)

    def serialize_to_dict_list(self, G):
        """Converts an annotated DiGraph into a list of dicts for CLAP-ASM."""
        # CLAP-ASM expects a list of dicts, where each dict is a function/block.
        # We will treat each BB as a 'function' for the encoder's perspective 
        # or combine them? The research says 'one dict per function'.
        # Since we are doing BB-level, maybe we provide one dict for the whole program?
        
        # Let's try one dict for the whole program first, with sequential keys.
        asm_dict = {}
        idx = 1
        for n in sorted(G.nodes):
            block_data = G.nodes[n]
            for instr in block_data['instructions']:
                asm_str = f"{instr['mnemonic']} {instr['operands']}"
                asm_dict[str(idx)] = asm_str
                idx += 1
                
        return [asm_dict]

if __name__ == "__main__":
    from ebpf_embed.extractor.fcfg import FCFGExtractor
    from ebpf_embed.extractor.tokenizer import EBPFTokenizer
    
    extractor = FCFGExtractor()
    disasm = extractor.get_disassembly("ebpf_embed/data/xdp_drop_kern.o")
    instrs = extractor.parse_instructions(disasm)
    blocks = extractor.find_basic_blocks(instrs)
    graph = extractor.build_graph(blocks)
    
    tokenizer = EBPFTokenizer()
    graph = tokenizer.annotate_graph(graph)
    
    serializer = FCFGSerializer()
    output = serializer.serialize(graph)
    
    print("--- Serialized FCFG ---")
    print(output)
