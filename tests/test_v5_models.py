# -*- coding: utf-8 -*-
import json
from pathlib import Path

import pytest

from zfh_voice2.models import (assert_model_contained, inspect_v5_weight,
                               resolve_selected_model, scan_v5_models)


def put(path: Path, head=b"07"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(head + b"payload")
    return path


def test_scan_accepts_only_v5dev_and_sorts_epochs(tmp_path):
    put(tmp_path / "torch_weights/sovits/v5_full_e32_final_e32_s1728_l0.pth")
    put(tmp_path / "torch_weights/sovits/v5_full_e32_final_e8_s432_l0.pth")
    put(tmp_path / "torch_weights/old_e12_s384_l0.pth", b"05")
    put(tmp_path / "torch_weights/turbo_e1_s1_l0.pth", b"08")
    rows = scan_v5_models(tmp_path)
    assert [x.epoch for x in rows] == [8, 32]
    assert rows[-1].recommended is True
    assert all(x.version == "v5dev" for x in rows)


def test_configuration_default_wins(tmp_path):
    p8 = put(tmp_path / "torch_weights/sovits/model_e8_s432_l0.pth")
    put(tmp_path / "torch_weights/sovits/model_e32_s1728_l0.pth")
    (tmp_path / "configuration.json").write_text(json.dumps({
        "default_sovits": "torch_weights/sovits/model_e8_s432_l0.pth"}),
        encoding="utf-8")
    assert resolve_selected_model(tmp_path).path == str(p8.resolve())


def test_explicit_model_must_be_v5_and_contained(tmp_path):
    p = put(tmp_path / "torch_weights/sovits/model_e16_s864_l0.pth")
    assert resolve_selected_model(tmp_path, p).epoch == 16
    outside = put(tmp_path.parent / "outside_e32_s1_l0.pth")
    with pytest.raises(ValueError, match="模型目录"):
        assert_model_contained(tmp_path, outside)


def test_inspect_rejects_v2pro(tmp_path):
    p = put(tmp_path / "torch_weights/v2pro_e12_s384_l0.pth", b"05")
    assert inspect_v5_weight(p) is None
