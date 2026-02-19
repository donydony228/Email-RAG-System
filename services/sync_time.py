import json
import sys
from pathlib import Path
from datetime import datetime, timezone
sys.path.insert(0, str(Path(__file__).parent.parent))

# 讀取
def load_last_sync():
    SYNC_FILE = Path("last_sync.json")
    if not SYNC_FILE.exists():
        return None
    with open(SYNC_FILE) as f:
        data = json.load(f)
    return datetime.fromisoformat(data["last_sync_at"])

# 寫入
def save_last_sync():
    now = datetime.now(timezone.utc)
    SYNC_FILE = Path("last_sync.json")
    with open(SYNC_FILE, "w") as f:
        json.dump({"last_sync_at": now.isoformat()}, f)
