import subprocess
import re
import networkx as nx
import os

class FCFGExtractor:
    def __init__(self, llvm_objdump_path="/opt/homebrew/opt/llvm/bin/llvm-objdump"):
        self.llvm_objdump_path = llvm_objdump_path

    def get_disassembly(self, file_path):
        """Runs llvm-objdump -d and returns the output."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        result = subprocess.run(
            [self.llvm_objdump_path, "-d", file_path],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout

    def parse_instructions(self, disassembly):
        """Parses disassembly lines into instruction objects."""
        instructions = []
        # Pattern for eBPF objdump lines:
        #   offset:  hex_bytes  mnemonic  operands  [<target>]
        # Example: 9: 2d 23 15 00 00 00 00 00 if r3 > r2 goto +0x15 <xdp_prog_func+0xf8>
        # Example: 16: 85 00 00 00 01 00 00 00 call 0x1
        
        # We look for lines starting with whitespace then offset then colon
        pattern = re.compile(r"^\s*([0-9a-f]+):\s+([0-9a-f ]+)\t(.*)$", re.IGNORECASE)
        
        current_function = None
        func_pattern = re.compile(r"^([0-9a-f]+) <(.*)>:$", re.IGNORECASE)

        lines = disassembly.splitlines()
        for line in lines:
            func_match = func_pattern.match(line)
            if func_match:
                current_function = func_match.group(2)
                continue

            instr_match = pattern.match(line)
            if instr_match:
                offset = int(instr_match.group(1), 16)
                # group 2 is hex bytes, ignore for now
                raw_content = instr_match.group(3).strip()
                
                # Split mnemonic and operands
                parts = raw_content.split(None, 1)
                mnemonic = parts[0]
                operands = parts[1] if len(parts) > 1 else ""
                
                # Identify if it's a jump or call
                is_jump = mnemonic in ['goto', 'exit', 'call'] or mnemonic.startswith('if')
                
                target_offset = None
                if 'goto' in raw_content or 'if' in raw_content:
                    # Look for <target> like <xdp_prog_func+0xf8>
                    target_match = re.search(r"<(.*)\+0x([0-9a-f]+)>", raw_content)
                    if target_match:
                        target_offset = int(target_match.group(2), 16)
                    else:
                        # Sometimes it's <label> without offset if it's the start
                        target_match = re.search(r"<(.*)>", raw_content)
                        if target_match:
                            # If it's just <label>, it's offset 0 of that label? 
                            # Usually objdump shows the absolute offset in the tag if it's within the section.
                            pass

                instructions.append({
                    'offset': offset,
                    'mnemonic': mnemonic,
                    'operands': operands,
                    'raw': raw_content,
                    'is_jump': is_jump,
                    'is_exit': mnemonic == 'exit',
                    'target_offset': target_offset,
                    'function': current_function
                })
        
        return instructions

    def find_basic_blocks(self, instructions):
        """Identifies basic block boundaries based on jumps and targets."""
        if not instructions:
            return []

        leaders = set()
        leaders.add(instructions[0]['offset']) # First instruction is a leader

        offset_to_idx = {instr['offset']: i for i, instr in enumerate(instructions)}

        for i, instr in enumerate(instructions):
            if instr['is_jump'] and instr['target_offset'] is not None:
                leaders.add(instr['target_offset'])
                # The instruction following a jump is a leader
                if i + 1 < len(instructions):
                    leaders.add(instructions[i+1]['offset'])
            
            if instr['is_exit']:
                if i + 1 < len(instructions):
                    leaders.add(instructions[i+1]['offset'])
            
            # For conditional jumps, the target is a leader AND the next instruction is a leader
            if instr['mnemonic'].startswith('if'):
                if i + 1 < len(instructions):
                    leaders.add(instructions[i+1]['offset'])

        sorted_leaders = sorted(list(leaders))
        
        blocks = []
        current_block = []
        for instr in instructions:
            if instr['offset'] in leaders and current_block:
                blocks.append(current_block)
                current_block = []
            current_block.append(instr)
        if current_block:
            blocks.append(current_block)
            
        return blocks

    def build_graph(self, blocks):
        """Builds a DiGraph from basic blocks."""
        G = nx.DiGraph()
        
        block_map = {b[0]['offset']: i for i, b in enumerate(blocks)}
        
        for i, block in enumerate(blocks):
            start_offset = block[0]['offset']
            end_instr = block[-1]
            
            G.add_node(i, offset=start_offset, instructions=block)
            
            # Determine successors
            successors = []
            if end_instr['is_exit']:
                pass # No successors
            elif 'goto' in end_instr['mnemonic']:
                if end_instr['target_offset'] in block_map:
                    successors.append(block_map[end_instr['target_offset']])
            elif end_instr['mnemonic'].startswith('if'):
                # Successor 1: Target of jump
                if end_instr['target_offset'] in block_map:
                    successors.append(block_map[end_instr['target_offset']])
                # Successor 2: Fall through (next block)
                if i + 1 < len(blocks):
                    successors.append(i + 1)
            else:
                # Normal fall through
                if i + 1 < len(blocks):
                    successors.append(i + 1)
            
            for succ in successors:
                G.add_edge(i, succ)
        
        return G

if __name__ == "__main__":
    extractor = FCFGExtractor()
    try:
        disasm = extractor.get_disassembly("ebpf_embed/data/xdp_drop_kern.o")
        instrs = extractor.parse_instructions(disasm)
        blocks = extractor.find_basic_blocks(instrs)
        graph = extractor.build_graph(blocks)
        
        print(f"Extraction successful: {len(instrs)} instructions, {len(blocks)} basic blocks.")
        print(f"Graph nodes: {graph.number_of_nodes()}, edges: {graph.number_of_edges()}")
        
        for n in graph.nodes:
            print(f"Block {n} (offset {graph.nodes[n]['offset']}):")
            for instr in graph.nodes[n]['instructions']:
                print(f"  {instr['offset']}: {instr['mnemonic']} {instr['operands']}")
            print(f"  Successors: {list(graph.successors(n))}")
            
    except Exception as e:
        print(f"Error: {e}")
