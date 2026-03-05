from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

try:
    import yaml  # type: ignore
except Exception:
    yaml = None


class Settings:
    def __init__(self, raw: Dict[str, Any]) -> None:
        self.raw = raw

    @property
    def app(self) -> Dict[str, Any]:
        return self.raw["app"]

    @property
    def model(self) -> Dict[str, Any]:
        return self.raw["model"]

    @property
    def inference(self) -> Dict[str, Any]:
        return self.raw["inference"]

    @property
    def retrieval(self) -> Dict[str, Any]:
        return self.raw["retrieval"]

    @property
    def validation(self) -> Dict[str, Any]:
        return self.raw.get("validation", {})

    @property
    def paths(self) -> Dict[str, Any]:
        return self.raw["paths"]


def load_settings(path: str) -> Settings:
    """Loads YAML settings with a fallback parser when PyYAML is unavailable."""
    with Path(path).open("r", encoding="utf-8") as f:
        raw = f.read()
    if yaml is not None:
        return Settings(yaml.safe_load(raw))
    return Settings(_simple_yaml_load(raw))


def _simple_yaml_load(text: str) -> Dict[str, Any]:
    """Parses simple indentation-based YAML into nested dictionaries."""
    # Minimal fallback parser for indentation-based key/value YAML mappings.
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    root: Dict[str, Any] = {}
    stack: list[tuple[int, Dict[str, Any]]] = [(-1, root)]

    for line in lines:
        indent = len(line) - len(line.lstrip(" "))
        key, _, value = line.strip().partition(":")
        value = value.strip()

        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]

        if value == "":
            node: Dict[str, Any] = {}
            parent[key] = node
            stack.append((indent, node))
            continue

        if value.startswith('"') and value.endswith('"'):
            parent[key] = value[1:-1]
        elif value.startswith("'") and value.endswith("'"):
            parent[key] = value[1:-1]
        elif value.lower() in ("true", "false"):
            parent[key] = value.lower() == "true"
        else:
            stripped = value.split(" #", 1)[0].strip()
            try:
                parent[key] = int(stripped)
            except ValueError:
                try:
                    parent[key] = float(stripped)
                except ValueError:
                    parent[key] = stripped

    return root
