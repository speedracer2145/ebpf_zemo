import networkx as nx

class EBPFTokenizer:
    def __init__(self):
        # Semantic tokens for eBPF
        self.HELPER_MAP = {
            1: "MAP_LOOKUP",
            2: "MAP_UPDATE",
            3: "MAP_DELETE",
            4: "PROBE_READ",
            5: "KTIME_GET_NS",
            6: "TRACE_PRINTK",
            # ... and so on. We can expand this.
        }
        self.XDP_ACTIONS = {
            0: "XDP_ABORTED",
            1: "XDP_DROP",
            2: "XDP_PASS",
            3: "XDP_TX",
            4: "XDP_REDIRECT",
        }

    def annotate_graph(self, G):
        """Annotates DiGraph nodes (basic blocks) with semantic tokens."""
        for n in G.nodes:
            block = G.nodes[n]['instructions']
            annotations = []
            
            for instr in block:
                mnemonic = instr['mnemonic']
                operands = instr['operands']
                
                # Detect Map Operations & Helpers
                if mnemonic == 'call':
                    # Extract helper ID if present (e.g., '0x1')
                    try:
                        helper_id = int(operands.strip(), 16)
                        token = self.HELPER_MAP.get(helper_id, f"HELPER_{helper_id}")
                        annotations.append(token)
                    except ValueError:
                        annotations.append("HELPER_CALL")
                
                # Detect register context access (r1 usually pkt context)
                if '*(u32 *)(r1 +' in operands or '*(u16 *)(r1 +' in operands:
                    annotations.append("PKT_READ")
                
                # Detect XDP return values
                if mnemonic in ['mov', 'mov64', 'r0 =']:
                    if '0x1' in operands or '1' in operands:
                        annotations.append("RET_DROP")
                    elif '0x2' in operands or '2' in operands:
                        annotations.append("RET_PASS")
                
                # Detect Map access patterns (r18/r1 loading map fds)
                if 'll' in operands and '0x0' not in operands:
                    annotations.append("MAP_REF")

            G.nodes[n]['annotations'] = annotations
        return G

if __name__ == "__main__":
    # Test with a dummy graph or integration test
    from ebpf_embed.extractor.fcfg import FCFGExtractor
    extractor = FCFGExtractor()
    disasm = extractor.get_disassembly("ebpf_embed/data/xdp_drop_kern.o")
    instrs = extractor.parse_instructions(disasm)
    blocks = extractor.find_basic_blocks(instrs)
    graph = extractor.build_graph(blocks)
    
    tokenizer = EBPFTokenizer()
    graph = tokenizer.annotate_graph(graph)
    
    for n in graph.nodes:
        print(f"Block {n} annotations: {graph.nodes[n]['annotations']}")
