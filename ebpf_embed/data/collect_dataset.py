import os
import subprocess
import hashlib
import json
import shutil

REPOS = {
    "ebpf-samples": "https://github.com/vbpf/ebpf-samples.git",
    "xdp-tutorial": "https://github.com/xdp-project/xdp-tutorial.git",
    "libbpf-bootstrap": "https://github.com/libbpf/libbpf-bootstrap.git"
}

TEMP_DIR = "/Users/alok/ebpf_zemo/ebpf_repos_temp"
DATA_DIR = "/Users/alok/ebpf_zemo/ebpf_embed/data"
METADATA_FILE = os.path.join(DATA_DIR, "dataset_metadata.json")

def get_sha256(filepath):
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def get_program_type(filename, path):
    lower_name = filename.lower()
    lower_path = path.lower()
    if 'xdp' in lower_name or 'xdp' in lower_path:
        return "XDP Hook"
    elif 'tc' in lower_name or 'cls' in lower_name or 'sch' in lower_name or 'tc' in lower_path:
        return "TC Hook"
    elif 'sock' in lower_name or 'skb' in lower_name or 'sock' in lower_path:
        return "Socket/Networking"
    elif 'kprobe' in lower_name or 'kretprobe' in lower_name:
        return "Kprobe"
    elif 'trace' in lower_name:
        return "Tracepoint"
    return "Other eBPF"

def get_industry_category(repo, path):
    lower_path = path.lower()
    if repo == "ebpf-samples":
        if 'cilium' in lower_path:
            return "Cloud Native Networking (Cilium)"
        elif 'suricata' in lower_path:
            return "Network Security (Suricata)"
        elif 'katran' in lower_path:
            return "L4 Load Balancing (Meta)"
        elif 'falco' in lower_path:
            return "Runtime Security (Falco)"
        elif 'bcc' in lower_path:
            return "Performance Tracing (BCC)"
        elif 'linux' in lower_path:
            return "Linux Kernel Testing"
        return "Misc eBPF Samples"
    elif repo == "xdp-tutorial":
        return "Networking/XDP Standard"
    elif repo == "libbpf-bootstrap":
        return "Modern eBPF Boilerplates"
    return "Unknown"

def main():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    # 1. Get hashes of existing files
    print("Hashing existing files in dataset...")
    existing_hashes = set()
    existing_files = os.listdir(DATA_DIR)
    for f in existing_files:
        if f.endswith(".o"):
            full_path = os.path.join(DATA_DIR, f)
            existing_hashes.add(get_sha256(full_path))
    
    print(f"Found {len(existing_hashes)} existing unique .o files.")

    # 2. Clone repos
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)
        
    for repo_name, repo_url in REPOS.items():
        repo_path = os.path.join(TEMP_DIR, repo_name)
        if not os.path.exists(repo_path):
            print(f"Cloning {repo_url}...")
            subprocess.run(["git", "clone", repo_url, repo_path], check=False)
        else:
            print(f"Repo {repo_name} already cloned.")

    # 3. Find and process .o files
    metadata = {}
    if os.path.exists(METADATA_FILE):
        try:
            with open(METADATA_FILE, 'r') as f:
                metadata = json.load(f)
        except Exception:
            pass

    new_files_added = 0

    for repo_name in REPOS.keys():
        repo_path = os.path.join(TEMP_DIR, repo_name)
        
        for root, dirs, files in os.walk(repo_path):
            # Skip invalid/test-build directories that might contain bad eBPF
            if 'invalid' in root.lower() or 'bad' in root.lower():
                continue
                
            for file in files:
                if file.endswith(".o"):
                    filepath = os.path.join(root, file)
                    file_hash = get_sha256(filepath)
                    
                    if file_hash in existing_hashes:
                        continue # Skip duplicates
                        
                    # We found a new unique file!
                    existing_hashes.add(file_hash)
                    
                    # Handle naming collisions
                    target_filename = file
                    counter = 1
                    while os.path.exists(os.path.join(DATA_DIR, target_filename)):
                        # Same name, different hash. Add prefix.
                        target_filename = f"{repo_name}_{counter}_{file}"
                        counter += 1
                        
                    target_filepath = os.path.join(DATA_DIR, target_filename)
                    shutil.copy2(filepath, target_filepath)
                    
                    # Compute metadata
                    rel_path = os.path.relpath(filepath, repo_path)
                    prog_type = get_program_type(file, rel_path)
                    industry = get_industry_category(repo_name, rel_path)
                    
                    metadata[target_filename] = {
                        "source_repo": repo_name,
                        "industry_category": industry,
                        "program_type": prog_type,
                        "original_path": rel_path,
                        "sha256_hash": file_hash
                    }
                    
                    new_files_added += 1
                    print(f"Added: {target_filename} ({prog_type})")

    # 4. Save metadata
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=4)
        
    print(f"\nCollection complete! Added {new_files_added} new unique .o files.")
    
    # Cleanup temp dir to save space
    # shutil.rmtree(TEMP_DIR)

if __name__ == "__main__":
    main()
