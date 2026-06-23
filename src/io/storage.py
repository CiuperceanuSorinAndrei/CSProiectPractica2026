from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class StateStorage:
    """Persistenta simpla de stare ca fisiere JSON intr-un director cache.

    NOTA: momentan neutilizat in pipeline (starea de tracking traieste in memorie,
    in Orchestrator). Adus din branch-ul cinematici/tracking pentru completitudine.
    """
    _cache_dir: Path = None

    def __init__(self, cache_dir: str | Path | None = None):
        self._cache_dir = Path(cache_dir) if cache_dir is not None else Path("data") / "cache"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def save(self, name: str, state: dict[str, Any]) -> Path:
        """Salveaza starea ca JSON in directorul cache."""
        path = self._cache_dir / f"{name}.json"
        path.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")
        return path

    def load(self, name: str) -> dict[str, Any] | None:
        """Incarca starea din cache. Returneaza None daca nu exista."""
        path = self._cache_dir / f"{name}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def delete(self, name: str) -> bool:
        """Sterge un fisier de stare din cache."""
        path = self._cache_dir / f"{name}.json"
        if path.exists():
            path.unlink()
            return True
        return False
