import os
import json
import subprocess
import hashlib
import shutil

DATA_DIR = "/Users/alok/ebpf_zemo/ebpf_embed/data"
TEMP_DIR = "/Users/alok/ebpf_zemo/ebpf_repos_temp"
METADATA_FILE = os.path.join(DATA_DIR, "dataset_metadata.json")

# 1. Patterns of files to explicitly DELETE
JUNK_PATTERNS = [
    "minimal", "bootstrap", "bad", "invalid", "loop", "tailcall", 
    "atomics", "test_spin_lock", "test_global", "map_ptr_kern", 
    "divzero", "infinite", "nullmapref", "filelife", "oomkill",
    "capable", "bashreadline", "spintest", "bpf_alignchecker",
    "freplace", "get_cgroup_id"
]

def is_junk(filename):
    lower_f = filename.lower()
    for pattern in JUNK_PATTERNS:
        if pattern in lower_f:
            return True
    return False

def get_sha256(filepath):
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def main():
    print("--- Starting Dataset Curation ---")
    
    # 1. Purge Junk
    removed_count = 0
    remaining_files = []
    
    for f in os.listdir(DATA_DIR):
        if f.endswith(".o"):
            if is_junk(f):
                os.remove(os.path.join(DATA_DIR, f))
                removed_count += 1
                print(f"Removed trivial file: {f}")
            else:
                remaining_files.append(f)
                
    print(f"\nPurged {removed_count} trivial/test files.")
    
    # 2. Re-evaluate Metadata
    metadata = {}
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, 'r') as f:
            old_metadata = json.load(f)
            # Only keep metadata for files that still exist
            for key, val in old_metadata.items():
                if key in remaining_files:
                    metadata[key] = val
                    
    # 3. Attempt to fetch specific high-value Katran files if they don't exist
    katran_repo = "https://github.com/facebookincubator/katran.git"
    katran_dir = os.path.join(TEMP_DIR, "katran_repo")
    
    if not os.path.exists(katran_dir):
        print("\nCloning Katran repository to hunt for balancer_kern.c and decap_kern.c...")
        subprocess.run(["git", "clone", katran_repo, katran_dir], check=False)
    
    # Search for high value .c files and try to compile them
    high_value_targets = ['balancer_kern.c', 'decap_kern.c', 'healthchecking_kern.c', 'ratelimiting_kern.c', 'xdp_pktcntr.c']
    added_high_value = 0
    
    if os.path.exists(katran_dir):
        for root, dirs, files in os.walk(katran_dir):
            for f in files:
                if f in high_value_targets:
                    c_path = os.path.join(root, f)
                    o_filename = f.replace(".c", ".o")
                    o_path = os.path.join(DATA_DIR, o_filename)
                    
                    if not os.path.exists(o_path):
                        print(f"Found {f}! Attempting to compile to {o_filename}...")
                        # Best effort compilation (might fail due to complex header dependencies)
                        result = subprocess.run(
                            ["clang", "-O2", "-target", "bpf", "-c", c_path, "-o", o_path, "-I", root],
                            capture_output=True, text=True
                        )
                        if result.returncode == 0 and os.path.exists(o_path):
                            print(f"Successfully compiled {f}!")
                            metadata[o_filename] = {
                                "source_repo": "katran",
                                "industry_category": "L4 Load Balancing (Meta)",
                                "program_type": "XDP Hook" if 'xdp' in f or 'balancer' in f else "Other eBPF",
                                "original_path": os.path.relpath(c_path, katran_dir),
                                "sha256_hash": get_sha256(o_path)
                            }
                            added_high_value += 1
                        else:
                            print(f"Failed to compile {f}. This is normal for complex raw .c files missing vmlinux.h.")
    
    # Save updated metadata
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=4)
        
    print(f"\nCuration complete! Added {added_high_value} new compiled high-value files.")
    print(f"Total highly professional eBPF objects in dataset: {len(metadata.keys())}")

if __name__ == "__main__":
    main()
