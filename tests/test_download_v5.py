# -*- coding: utf-8 -*-
"""下载计划固定为 v5 新仓库，不再指向旧 ONNX/v2Pro 资源。"""
import download_models as d


def test_sources_point_to_clean_v5_repositories():
    assert d.MS_REPO == "yangwenjie1231/dsh-zhuang-fangyi-voice-v2-model"
    assert d.GH_REPO == "yangwenjie1231/dsh-zhuang-fangyi-voice-v2"
    assert d.GH_TAG == "model-v2-v5-fullft"


def test_default_epoch_is_e32_and_all_four_are_known():
    assert d.WEIGHTS[32].endswith("e32_s1728_l0.pth")
    assert sorted(d.WEIGHTS) == [8, 16, 24, 32]


def test_reference_assets_cover_three_styles():
    files = set(d.REFS)
    for name in ("normal", "firm", "gentle"):
        assert f"ref/styles/{name}.wav" in files
        assert f"ref/styles/{name}.txt" in files


def test_no_legacy_onnx_assets_are_downloaded():
    text = open(d.__file__, encoding="utf-8").read()
    assert ".onnx" not in text
