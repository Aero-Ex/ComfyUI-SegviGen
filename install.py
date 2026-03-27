import os
import sys
import subprocess
import platform
import argparse
import shutil

def run_command(command):
    try:
        subprocess.check_call(command)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        return False
    return True

def is_uv_available():
    try:
        subprocess.check_call([sys.executable, "-m", "uv", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except:
        return False

def get_env_info():
    info = {}
    
    # OS
    if platform.system() == "Windows":
        info['platform'] = "win_amd64"
    else:
        info['platform'] = "manylinux_2_35_x86_64" # Default specialized tag
        
    # Python version
    py_ver = f"cp{sys.version_info.major}{sys.version_info.minor}"
    info['python'] = py_ver
    
    # Torch and CUDA
    try:
        import torch
        torch_raw = torch.__version__.split('+')[0]
        torch_ver = torch_raw.split('.')
        info['torch'] = f"torch{torch_ver[0]}{torch_ver[1]}"
        info['torch_raw'] = torch_raw
        
        if hasattr(torch.version, 'cuda') and torch.version.cuda:
            cuda_ver = torch.version.cuda.split('.')
            info['cuda'] = f"cu{cuda_ver[0]}{cuda_ver[1]}"
        else:
            info['cuda'] = None
    except ImportError:
        print("PyTorch not found. Please install PyTorch with CUDA support first.")
        sys.exit(1)
        
    return info

def show_recommendations():
    print("\n" + "="*50)
    print("RECOMMENDED ENVIRONMENTS")
    print("="*50)
    print("If you encounter 404 errors, it is likely because your environment")
    print("combination is not yet supported on the wheel repository.")
    print("\nPreferred configurations:")
    print("\nYou can install the preferred Torch version using:")
    print("pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124")
    print("="*50 + "\n")

def find_wheel_url(lib_name, env):
    import urllib.request
    import re
    
    index_url = f"https://pozzettiandrea.github.io/cuda-wheels/{lib_name}/"
    
    try:
        req = urllib.request.Request(index_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            html = response.read().decode('utf-8')
    except Exception as e:
        print(f"  Warning: Could not access {lib_name} wheel index: {e}")
        return None

    # Find all wheel links
    links = re.findall(r'https?://[^\s<>"]+\.whl', html)
    if not links:
        return None
        
    target_py = env['python']
    target_plat = "win_amd64" if platform.system() == "Windows" else "manylinux"
    target_cuda = env['cuda'] # 'cu128' or 'cu130'
    target_torch = env['torch'] # 'torch28' or 'torch210'
    
    best_url = None
    best_score = -1
    
    for url in links:
        filename = url.split('/')[-1]
        
        # 1. Platform & Python MUST match
        if target_plat not in filename:
            continue
        if target_py not in filename:
            continue
            
        # Parse CUDA and Torch from filename
        # Example: flash_attn-2.8.3+cu124torch2.4-cp312-cp312-win_amd64.whl
        cuda_match = re.search(r'cu(\d+)', filename)
        torch_match = re.search(r'torch(\d+\.?\d*)', filename)
        
        if not cuda_match or not torch_match:
            continue
            
        file_cuda = f"cu{cuda_match.group(1)}"
        raw_torch_ver = torch_match.group(1)
        # Handle cases where torch version might or might not have dots
        file_torch = raw_torch_ver.replace(".", "") # '24' or '210'
        file_torch_str = f"torch{file_torch}"
        
        # Score calculation
        score = 0
        
        # 1. CUDA Major Compatibility
        if file_cuda[:4] == target_cuda[:4]:
            score += 1000 # Same major version
        elif target_cuda.startswith('cu13') and file_cuda.startswith('cu12'):
            score += 100 # CUDA 13 -> 12 fallback
        else:
            continue # Major mismatch
            
        # 2. Torch Major Compatibility
        if not file_torch or len(target_torch) < 6:
            continue
        if target_torch.startswith(f"torch{file_torch[0]}"):
            score += 1000
        else:
            continue # Torch major mismatch
        
        # 3. Exact Version Bonuses
        if file_cuda == target_cuda:
            score += 5000 # Massive bonus for perfect CUDA
        else:
            # Low-priority tie-breaker: prefer higher versions within the same major
            try:
                score += int(file_cuda[2:])
            except: pass
            
        if file_torch_str == target_torch:
            score += 10000 # Massive bonus for perfect Torch
        else:
             # Prefer closest torch version
             try:
                 diff = abs(int(file_torch) - int(target_torch[5:]))
                 score += (500 - min(diff * 10, 500))
             except: pass
             
        if score > best_score:
            best_score = score
            best_url = url
            
    if best_url:
        print(f"  Selected best matching wheel: {best_url.split('/')[-1]} (Score: {best_score})")
        return best_url

    return None

def install_flash_attn(pip_base, env, dry_run=False):
    print("\n--- Installing flash-attn ---")
    wheel_url = find_wheel_url("flash-attn", env)
    
    if wheel_url:
        print(f"  Found matching wheel: {wheel_url}")
        if dry_run:
            print(f"  [Dry Run] Would run: {' '.join(pip_base + ['install', wheel_url])}")
            return True
        else:
            return run_command(pip_base + ["install", wheel_url])
    else:
        print(f"  Warning: No matching flash-attn wheel found for {env['cuda']}, {env['torch']}, {env['python']} on {env['platform']}.")
        if platform.system() == "Windows":
             print("  Skipping flash-attn on Windows as it is often incompatible and optional for inference.")
             return True
        return False

def install_triton_windows(pip_base, env, dry_run=False):
    if platform.system() != "Windows":
        return True

    print("\n--- Installing triton-windows ---")
    
    torch_raw = env.get('torch_raw', '2.0.0')
    try:
        from packaging import version
    except ImportError:
        class VersionFallback:
            def __init__(self, v): self.v = [int(x) for x in v.split('.')[:2]]
            def __ge__(self, other): return self.v >= [int(x) for x in other.split('.')[:2]]
        version = type('obj', (object,), {'parse': lambda v: VersionFallback(v)})

    v_torch = version.parse(torch_raw)
    triton_spec = "triton-windows"
    
    if v_torch >= version.parse("2.10.0"):
        triton_spec = "triton-windows<3.7"
    elif v_torch >= version.parse("2.9.0"):
        triton_spec = "triton-windows<3.6"
    elif v_torch >= version.parse("2.8.0"):
        triton_spec = "triton-windows<3.5"
    elif v_torch >= version.parse("2.7.0"):
        triton_spec = "triton-windows<3.4"
    elif v_torch >= version.parse("2.6.0"):
        triton_spec = "triton-windows<3.3"

    if dry_run:
        print(f"  [Dry Run] Would run: {' '.join(pip_base + ['install', '-U', triton_spec])}")
        return True
    else:
        return run_command(pip_base + ["install", "-U", triton_spec])

def install():
    parser = argparse.ArgumentParser(description="ComfyUI-SegviGen Installation")
    parser.add_argument("--dry-run", action="store_true", help="Print the steps without installing")
    args = parser.parse_args()

    print("--- ComfyUI-SegviGen Installation ---")
    if args.dry_run:
        print("(DRY RUN MODE)")
    
    env = get_env_info()
    print(f"Detected Environment:")
    print(f"  OS: {platform.system()}")
    print(f"  Python: {env['python']}")
    print(f"  PyTorch: {env['torch']}")
    print(f"  CUDA: {env['cuda']}")
    
    if not env['cuda']:
        print("Error: CUDA not detected in PyTorch. These wheels require CUDA.")
        sys.exit(1)
    
    # 1. Install standard requirements
    current_dir = os.path.dirname(os.path.abspath(__file__))
    requirements_path = os.path.join(current_dir, "requirements.txt")
    
    use_uv = is_uv_available()
    pip_base = [sys.executable, "-m", "uv", "pip"] if use_uv else [sys.executable, "-m", "pip"]
    
    if os.path.exists(requirements_path):
        print(f"\nInstalling requirements from {requirements_path}...")
        if args.dry_run:
            print(f"  [Dry Run] Would run: {' '.join(pip_base + ['install', '-r', requirements_path])}")
        else:
            run_command(pip_base + ["install", "-r", requirements_path])

    # 2. Install flash-attn
    install_flash_attn(pip_base, env, dry_run=args.dry_run)
    
    # 3. Install correct triton-windows version
    install_triton_windows(pip_base, env, dry_run=args.dry_run)
    
    # 4. Define other CUDA wheels
    packages = [
        {"name": "cumesh", "tag": "cumesh-latest"},
        {"name": "flex-gemm", "tag": "flex_gemm-latest"},
        {"name": "nvdiffrast", "tag": "nvdiffrast-latest"},
        {"name": "nvdiffrec-render", "tag": "nvdiffrec_render-latest"},
        {"name": "o-voxel", "tag": "o_voxel-latest"},
    ]
    
    print("\nInstalling CUDA wheels...")
    
    errors_occurred = False
    for pkg in packages:
        print(f"Installing {pkg['name']}...")
        wheel_url = find_wheel_url(pkg['name'], env)
        if wheel_url:
            print(f"  Found matching wheel: {wheel_url}")
            if args.dry_run:
                print(f"  [Dry Run] Would run: {' '.join(pip_base + ['install', wheel_url])}")
            else:
                if not run_command(pip_base + ["install", wheel_url]):
                    print(f"Warning: Failed to install {pkg['name']}.")
                    errors_occurred = True
        else:
            print(f"  Warning: No matching {pkg['name']} wheel found in the index for {env['platform']}.")
            # Try git fallbacks for known libraries on Windows
            git_fallbacks = {
                "nvdiffrast": "git+https://github.com/NVlabs/nvdiffrast",
                "flex_gemm": "git+https://github.com/NVlabs/flex-gemm",
                "flex-gemm": "git+https://github.com/NVlabs/flex-gemm"
            }
            if pkg['name'] in git_fallbacks:
                target = git_fallbacks[pkg['name']]
                print(f"  Attempting source installation for {pkg['name']} as fallback: {target}")
                if args.dry_run:
                    print(f"  [Dry Run] Would run: {' '.join(pip_base + ['install', target])}")
                elif run_command(pip_base + ["install", target]):
                    print(f"  Successfully installed {pkg['name']} from source.")
                    continue
            
            errors_occurred = True

    if errors_occurred:
        show_recommendations()

    print("\nInstallation complete!")

if __name__ == "__main__":
    install()
