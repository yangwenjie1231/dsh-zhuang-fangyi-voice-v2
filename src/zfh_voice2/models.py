# -*- coding: utf-8 -*-
"""GPT-SoVITS v5dev 权重发现与选择。只读取头和文件元数据。"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

_RX = re.compile(r"_e(?P<epoch>\d+)_s(?P<step>\d+)_l\d+\.pth$", re.I)


@dataclass(frozen=True)
class V5Model:
    id: str
    path: str
    filename: str
    version: str
    epoch: int
    step: int
    size: int
    recommended: bool = False
    selected: bool = False

    def to_dict(self):
        return asdict(self)


def assert_model_contained(model_dir, path):
    root = (Path(model_dir).resolve() / "torch_weights").resolve()
    p = Path(path)
    if not p.is_absolute():
        p = Path(model_dir) / p
    p = p.resolve()
    try:
        p.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"权重必须位于模型目录 {root} 内: {p}") from exc
    return str(p)


def inspect_v5_weight(path):
    p = Path(path)
    if not p.is_file() or p.suffix.lower() != ".pth":
        return None
    try:
        with p.open("rb") as f:
            head = f.read(2)
    except OSError:
        return None
    if head != b"07":
        return None
    m = _RX.search(p.name)
    epoch = int(m.group("epoch")) if m else 0
    step = int(m.group("step")) if m else 0
    return V5Model(p.stem, str(p.resolve()), p.name, "v5dev",
                   epoch, step, p.stat().st_size)


def _configured_default(model_dir):
    p = Path(model_dir) / "configuration.json"
    try:
        value = json.loads(p.read_text(encoding="utf-8")).get("default_sovits")
        return assert_model_contained(model_dir, value) if value else None
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def scan_v5_models(model_dir):
    root = Path(model_dir).resolve()
    td = root / "torch_weights"
    found = [x for p in td.rglob("*.pth") if (x := inspect_v5_weight(p))]
    found.sort(key=lambda x: (x.epoch, x.step, x.filename.lower()))
    default = _configured_default(root)
    fallback = found[-1].path if found else None
    preferred = default if default and any(x.path == default for x in found) else fallback
    return [V5Model(**{**x.to_dict(), "recommended": x.path == preferred}) for x in found]


def resolve_selected_model(model_dir, explicit=None):
    rows = scan_v5_models(model_dir)
    if explicit:
        p = assert_model_contained(model_dir, explicit)
        row = inspect_v5_weight(p)
        if row is None:
            raise ValueError("v2 插件仅支持 GPT-SoVITS v5dev（权重头 07）；v2/v2Pro 请使用 v1 插件")
        return V5Model(**{**row.to_dict(), "selected": True})
    for row in rows:
        if row.recommended:
            return V5Model(**{**row.to_dict(), "selected": True})
    raise FileNotFoundError("找不到 GPT-SoVITS v5dev 权重；请运行 python download_models.py --torch-only")
