import os
import subprocess
import hashlib
import json
import shutil

REPOS = {
    "cloudflare_xdpcap": "https://github.com/cloudflare/xdpcap.git",
    "aqua_tracee": "https://github.com/aquasecurity/tracee.git",
    "isovalent_tetragon": "https://github.com/cilium/tetragon.git",
    "project_calico": "https://github.com/projectcalico/calico.git",
    "datadog_agent": "https://github.com/DataDog/datadog-agent.git"
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

def is_ebpf_source(filepath):
    """Check if a .c file contains common eBPF macros/headers."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            # Simple heuristic: look for section definitions or bpf headers
            if 'SEC(' in content or 'bpf_helpers.h' in content or '#include <linux/bpf.h>' in content:
                return True
    except Exception:
        pass
    return False

def get_industry_category(repo):
    if repo == "cloudflare_xdpcap":
        return "Network Observability/DDoS (Cloudflare)"
    elif repo == "aqua_tracee":
        return "Runtime Forensics (Aqua Security)"
    elif repo == "isovalent_tetragon":
        return "Security Observability (Tetragon)"
    elif repo == "project_calico":
        return "Kubernetes Networking (Calico)"
    elif repo == "datadog_agent":
        return "Performance Monitoring (Datadog)"
    return "Unknown"

def main():
    print("--- Starting Deep Internet Sweep ---")
    
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    # 1. Load existing metadata and hashes
    metadata = {}
    existing_hashes = set()
    if os.path.exists(METADATA_FILE):
        try:
            with open(METADATA_FILE, 'r') as f:
                metadata = json.load(f)
                for v in metadata.values():
                    if "sha256_hash" in v:
                        existing_hashes.add(v["sha256_hash"])
        except Exception:
            pass

    # 2. Clone repos
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)
        
    for repo_name, repo_url in REPOS.items():
        repo_path = os.path.join(TEMP_DIR, repo_name)
        if not os.path.exists(repo_path):
            print(f"\nCloning {repo_url}...")
            # Limit depth to save time
            subprocess.run(["git", "clone", "--depth", "1", repo_url, repo_path], check=False)
        else:
            print(f"\nRepo {repo_name} already cloned.")

    new_files_added = 0

    # 3. Hunt and Compile
    for repo_name in REPOS.keys():
        repo_path = os.path.join(TEMP_DIR, repo_name)
        if not os.path.exists(repo_path):
            continue
            
        print(f"\nHunting for eBPF code in {repo_name}...")
        for root, dirs, files in os.walk(repo_path):
            for file in files:
                if file.endswith(".c"):
                    filepath = os.path.join(root, file)
                    if is_ebpf_source(filepath):
                        print(f"Found eBPF source: {file}. Attempting compilation...")
                        
                        o_filename = f"{repo_name}_{file.replace('.c', '.o')}"
                        o_path = os.path.join(TEMP_DIR, o_filename)
                        
                        # Best effort compilation
                        result = subprocess.run(
                            ["clang", "-O2", "-target", "bpf", "-g", "-c", filepath, "-o", o_path, f"-I{root}"],
                            capture_output=True, text=True
                        )
                        
                        if result.returncode == 0 and os.path.exists(o_path):
                            file_hash = get_sha256(o_path)
                            
                            if file_hash in existing_hashes:
                                print("-> Compilation successful, but file is a duplicate. Skipping.")
                                os.remove(o_path)
                                continue
                                
                            existing_hashes.add(file_hash)
                            
                            # Move to DATA_DIR
                            final_o_path = os.path.join(DATA_DIR, o_filename)
                            shutil.move(o_path, final_o_path)
                            
                            rel_path = os.path.relpath(filepath, repo_path)
                            industry = get_industry_category(repo_name)
                            prog_type = "XDP Hook" if "xdp" in file.lower() else "TC Hook" if "tc" in file.lower() else "Other eBPF"
                            
                            metadata[o_filename] = {
                                "source_repo": repo_name,
                                "industry_category": industry,
                                "program_type": prog_type,
                                "original_path": rel_path,
                                "sha256_hash": file_hash
                            }
                            
                            new_files_added += 1
                            print(f"-> SUCCESS! Added {o_filename} to dataset.")
                        else:
                            # print(f"-> Compilation failed (missing internal headers). Skipping.")
                            pass

    # 4. Save metadata
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=4)
        
    print(f"\nSweep Complete! Successfully extracted {new_files_added} new compiled high-end .o files.")

if __name__ == "__main__":
    main()
