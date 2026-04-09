import os
import requests
import subprocess

class DataCollector:
    def __init__(self, target_dir="ebpf_embed/data"):
        self.target_dir = target_dir
        os.makedirs(target_dir, exist_ok=True)
        
        # Repositories and specific files to pull
        self.SOURCES = {
            "vbpf/ebpf-samples": [
                # Existing cilium-examples
                "cilium-examples/xdp_bpf_bpfel.o",
                "cilium-examples/kprobe_bpf_bpfel.o",
                "cilium-examples/fentry_bpf_bpfel.o",
                "cilium-examples/tcx_bpf_bpfel.o",
                "cilium-examples/cgroup_skb_bpf_bpfel.o",
                "cilium-examples/tracepoint_in_c_bpf_bpfel.o",
                "cilium-examples/ringbuffer_bpf_bpfel.o",
                # Linux Selftests
                "linux-selftests/fexit_sleep.o",
                "linux-selftests/bpf_cubic.o",
                "linux-selftests/bloom_filter_map.o",
                # Advanced Network Functions
                "katran/xdp_root.o",
                "falco/probe.o",
                "suricata/filter.o",
                "suricata/lb.o",
                "suricata/xdp_filter.o",
                "suricata/vlan_filter.o",
                "ovs/datapath.o",
                "cilium-core/bpf_host.o",
                "cilium-core/bpf_lxc.o",
                "cilium-core/bpf_network.o",
                "cilium-core/bpf_overlay.o",
                "cilium-core/bpf_xdp.o",
                "cilium-core/bpf_wireguard.o",
                "cilium-core/bpf_sock.o"
            ]
        }

    def download_file(self, repo, path):
        """Downloads a raw file from GitHub."""
        url = f"https://github.com/v1v/ebpf-samples/raw/master/xdp_drop_kern.o"
        # Actually use the subagent's found repo: vbpf/ebpf-samples
        url = f"https://github.com/{repo}/raw/master/{path}"
        
        filename = path.split("/")[-1]
        dest_path = os.path.join(self.target_dir, filename)
        
        print(f"Downloading {filename} from {repo}...")
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(dest_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Saved to {dest_path}")
            return dest_path
        else:
            print(f"Failed to download {filename} (Status: {response.status_code})")
            return None

    def collect(self):
        """Main collection loop."""
        collected_files = []
        for repo, paths in self.SOURCES.items():
            for path in paths:
                file_path = self.download_file(repo, path)
                if file_path:
                    collected_files.append(file_path)
        return collected_files

if __name__ == "__main__":
    collector = DataCollector()
    files = collector.collect()
    print(f"Collection complete. Total files: {len(files)}")
