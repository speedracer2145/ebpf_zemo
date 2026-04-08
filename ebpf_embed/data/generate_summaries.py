import os
import json
import ollama
from ebpf_embed.extractor.fcfg import FCFGExtractor
from ebpf_embed.extractor.tokenizer import EBPFTokenizer
from ebpf_embed.extractor.serializer import FCFGSerializer

class SummaryGenerator:
    def __init__(self, model="qwen2.5-coder:7b"):
        self.model = model
        self.extractor = FCFGExtractor()
        self.tokenizer = EBPFTokenizer()
        self.serializer = FCFGSerializer()

    def generate_summary(self, file_path):
        """Generates a semantic summary for an eBPF object file."""
        try:
            # 1. Extract and serialize FCFG
            disasm = self.extractor.get_disassembly(file_path)
            instrs = self.extractor.parse_instructions(disasm)
            blocks = self.extractor.find_basic_blocks(instrs)
            graph = self.extractor.build_graph(blocks)
            graph = self.tokenizer.annotate_graph(graph)
            serialized = self.serializer.serialize(graph)
            
            # 2. Prepare prompt for Ollama
            prompt = f"""
            You are an eBPF security and systems researcher. 
            Analyze the following serialized eBPF Function Control Flow Graph (FCFG).
            The graph contains basic blocks [BBn] with instructions and semantic tags (e.g., MAP_LOOKUP, PKT_READ).
            
            FCFG Data:
            {serialized}
            
            Task:
            Provide a concise (2-3 sentences) natural language summary of what this eBPF program does.
            Focus on:
            - The hook point (XDP, TC, Kprobe, etc.) if detectable.
            - Map operations (lookup, update).
            - Packet handling behavior (drop, pass, redirect).
            - The core logic (e.g., "It filters traffic based on port", "It counts packets in a map").
            
            Output ONLY the summary text.
            """
            
            # 3. Call Ollama
            response = ollama.chat(model=self.model, messages=[
                {'role': 'user', 'content': prompt},
            ])
            
            return response['message']['content'].strip()
        except Exception as e:
            return f"Error generating summary: {str(e)}"

    def batch_generate(self, data_dir="ebpf_embed/data", output_file="ebpf_embed/data/summaries.json"):
        """Generates summaries for all .o files in the data directory."""
        summaries = {}
        
        # Load existing summaries if any
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                summaries = json.load(f)
        
        for filename in os.listdir(data_dir):
            if filename.endswith(".o"):
                file_path = os.path.join(data_dir, filename)
                if filename not in summaries:
                    print(f"Generating summary for {filename}...")
                    summary = self.generate_summary(file_path)
                    summaries[filename] = summary
                    print(f"Summary: {summary}")
                else:
                    print(f"Skipping {filename}, summary already exists.")
                    
        with open(output_file, 'w') as f:
            json.dump(summaries, f, indent=4)
        
        return summaries

if __name__ == "__main__":
    generator = SummaryGenerator()
    generator.batch_generate()
