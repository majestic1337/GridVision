import json
from pathlib import Path
from typing import Dict, Optional


class AssetIndex:
    def __init__(self, manifest_path: Path):
        self._by_asset_id: Dict[str, Dict] = {}
        if manifest_path.exists():
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            for item in data:
                asset_id = item.get("asset_id")
                if asset_id:
                    self._by_asset_id[str(asset_id)] = item

    def get(self, asset_id: str) -> Optional[Dict]:
        return self._by_asset_id.get(str(asset_id))
