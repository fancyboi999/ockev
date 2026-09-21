import os
import sys
from pathlib import Path
from modelscope.hub.api import HubApi

WEIGHTS_DIR = Path("/Users/nowcoder/Desktop/auto-code-work/ockev/weights")
M15_DIR = WEIGHTS_DIR / "kev_qwen1.5b_anti_fp"
M3_DIR = WEIGHTS_DIR / "kev_qwen3b_anti_fp"

def upload(token: str = None, username: str = "fancyboi999"):
    api = HubApi()
    if token:
        api.login(token)
    elif not api.token:
        print("Please provide a ModelScope SDK token or run: modelscope login --token <token>")
        return False
        
    # 1. Create and push 1.5B
    repo_15 = f"{username}/ockev"
    print(f"Creating ModelScope model: {repo_15}...")
    try:
        api.create_model(model_id=repo_15, visibility=1, license="Apache 2.0", chinese_name="Ockev 判别式门禁模型 1.5B")
    except Exception as e:
        print(f"Note on create_model: {e}")
        
    print(f"Pushing weights to {repo_15}...")
    api.push_model(model_id=repo_15, model_dir=str(M15_DIR))
    print(f"✓ Successfully uploaded {repo_15} to ModelScope!")
    
    # 2. Create and push 3B
    repo_3 = f"{username}/ockev-3b"
    print(f"Creating ModelScope model: {repo_3}...")
    try:
        api.create_model(model_id=repo_3, visibility=1, license="Apache 2.0", chinese_name="Ockev 判别式门禁模型 3B")
    except Exception as e:
        print(f"Note on create_model: {e}")
        
    print(f"Pushing weights to {repo_3}...")
    api.push_model(model_id=repo_3, model_dir=str(M3_DIR))
    print(f"✓ Successfully uploaded {repo_3} to ModelScope!")
    return True

if __name__ == "__main__":
    tok = sys.argv[1] if len(sys.argv) > 1 else None
    upload(tok)
