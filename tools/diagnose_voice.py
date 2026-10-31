# -*- coding: utf-8 -*-
"""庄方宜语音 v2 · 一键诊断

回答一个具体问题：**"我听到的声音不对（气声重 / 发沙 / 爆音 / 不像），
是哪一环没配对？"**

用法：
    python tools/diagnose_voice.py
    python tools/diagnose_voice.py --model-dir D:\\models --gsv-root D:\\GPT-SoVITS
    python tools/diagnose_voice.py --wav out.wav          # 只查一个已生成的音频
    python tools/diagnose_voice.py --json report.json     # 输出机器可读报告

退出码：0 = 未发现明显问题；1 = 发现问题（详见输出）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from io import BytesIO
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "src"))

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
results: list[dict] = []


def add(level: str, title: str, detail: str = "", fix: str = "") -> None:
    results.append({"level": level, "title": title, "detail": detail, "fix": fix})
    mark = {"PASS": "✓", "WARN": "!", "FAIL": "✗"}[level]
    print(f"  [{mark}] {title}")
    if detail:
        for line in str(detail).splitlines():
            print(f"        {line}")
    if fix and level != PASS:
        for line in fix.splitlines():
            print(f"        → {line}")


def section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


# ---------------------------------------------------------------- 音频体检
def audio_health(path: Path, label: str) -> None:
    """削顶 / 电平 / 采样率 —— 最容易把"削顶"误当成"气声"的一步。"""
    try:
        import numpy as np
        import soundfile as sf
        x, sr = sf.read(str(path), dtype="float32", always_2d=False)
    except Exception as exc:  # noqa: BLE001
        add(FAIL, f"{label}：无法读取", f"{path}\n{type(exc).__name__}: {exc}")
        return
    if x.ndim == 2:
        x = x.mean(axis=1)
    if x.size == 0:
        add(FAIL, f"{label}：空音频", str(path))
        return

    peak = float(np.abs(x).max())
    rms = float(np.sqrt(np.mean(x ** 2)))
    clip = int((np.abs(x) >= 0.999).sum())
    dur = x.size / sr

    detail = f"sr={sr}  duration={dur:.2f}s  peak={peak:.3f}  rms={rms:.4f}  clipped={clip}"
    if peak >= 0.999 and clip > 100:
        add(FAIL, f"{label}：**削顶**（会被误听成气声/爆音）", detail,
            "写文件时没有把 int16 幅值归一：\n"
            "if np.abs(wav).max() > 2: wav = wav / 32768.0\n"
            "（本插件的 torch 后端已内含这一步；自定义脚本需要自己补）")
    elif rms > 0.35:
        add(WARN, f"{label}：RMS 偏高，可能增益过大", detail,
            "检查后处理增益；正常语音 rms 约 0.05~0.15")
    elif peak < 0.05:
        add(WARN, f"{label}：电平过低", detail, "检查增益/参考音频音量")
    else:
        add(PASS, f"{label}：电平正常", detail)

    # 频谱气声指标
    try:
        n, hop = 1024, 256
        if x.size >= n * 2:
            win = np.hanning(n).astype(np.float32)
            freqs = np.fft.rfftfreq(n, 1 / sr)
            band = (freqs >= 500) & (freqs <= 8000)
            hf = (freqs >= 6000) & (freqs <= min(10000, sr / 2 - 1))
            hff = (freqs >= 4000) & (freqs <= min(10000, sr / 2 - 1))
            ratios, flats, hnr = [], [], []
            for i in range(0, x.size - n, hop):
                seg = x[i:i + n] * win
                if float(np.sqrt(np.mean(seg ** 2))) < 1e-5:
                    continue
                spec = np.abs(np.fft.rfft(seg)) ** 2
                base = spec[band].sum()
                if base <= 0:
                    continue
                ratios.append(float(spec[hf].sum() / base))
                s = spec[hff]
                flats.append(float(np.exp(np.mean(np.log(s + 1e-12))) / (s.mean() + 1e-12)))
                ac = np.correlate(seg, seg, "full")[n - 1:]
                if ac[0] > 0:
                    lo, hi = max(1, int(sr / 500)), min(len(ac) - 1, int(sr / 60))
                    if hi > lo:
                        hnr.append(float(ac[lo:hi].max() / ac[0]))
            if ratios:
                hf_pct = float(np.median(ratios)) * 100
                detail = (f"6-10k 能量 {hf_pct:.2f}%（健康约 1.0~2.0%）  "
                          f"谱平坦度 {float(np.median(flats)):.3f}  "
                          f"谐噪比 {float(np.median(hnr)):.3f}")
                if hf_pct > 3.0:
                    add(WARN, f"{label}：高频气声偏重", detail,
                        "优先检查参考音频是否本身偏气（近距离麦克风/耳语/换气声）；\n"
                        "其次确认 is_half、temperature 未被调高")
                else:
                    add(PASS, f"{label}：气声在正常范围", detail)
    except Exception as exc:  # noqa: BLE001
        add(WARN, f"{label}：频谱分析跳过", f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------- 权重体检
def weight_health(path: Path, label: str = "SoVITS 权重") -> None:
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        add(WARN, f"{label}：无法检查（没装 torch）", f"{type(exc).__name__}: {exc}")
        return
    if not path.is_file():
        add(FAIL, f"{label}：文件不存在", str(path),
            "运行 python download_models.py --torch-only")
        return

    data = path.read_bytes()
    head = data[:2]
    size_mb = len(data) / 1e6
    if head == b"PK":
        add(FAIL, f"{label}：**没有 v5 版本头**（前两字节是 PK）", f"size={size_mb:.1f}MB",
            "推理端会按文件大小猜版本，可能加载 24kHz BigVGAN → 音质崩。\n"
            "先补头：python -m zfh_voice2 ... 或使用仓库的 fix_weight_header 工具")
    elif head == b"07":
        add(PASS, f"{label}：版本头 07（v5dev）", f"size={size_mb:.1f}MB")
    elif head == b"08":
        add(WARN, f"{label}：版本头 08（v5turbo）", f"size={size_mb:.1f}MB",
            "本插件只支持 v5dev；v5turbo 需要不同的 vocoder")
    else:
        add(FAIL, f"{label}：未知版本头 {head!r}", str(path),
            "v2 插件仅支持 GPT-SoVITS v5dev（07）；v2/v2Pro 请使用 v1 插件")

    try:
        obj = torch.load(BytesIO(b"PK" + data[2:]), map_location="cpu",
                         weights_only=False)
    except Exception as exc:  # noqa: BLE001
        add(WARN, f"{label}：无法反序列化（版本头可能不匹配）",
            f"{type(exc).__name__}: {str(exc)[:200]}")
        return

    weight = obj.get("weight") if isinstance(obj, dict) else None
    if not isinstance(weight, dict):
        add(FAIL, f"{label}：结构异常（缺少 weight 字典）", str(path))
        return

    keys = set(weight)
    groups = {g: sum(k.startswith(g + ".") for k in keys)
              for g in ("enc_p", "ssl_proj", "quantizer", "cfm")}
    detail = (f"keys={len(keys)}  enc_p={groups['enc_p']}  "
              f"ssl_proj={groups['ssl_proj']}  quantizer={groups['quantizer']}  "
              f"cfm={groups['cfm']}")
    if groups["quantizer"] == 0 or groups["ssl_proj"] == 0:
        add(FAIL, f"{label}：**权重不完整**（导出时漏掉冻结模块）", detail,
            "冻结参数被按 requires_grad 排除 → 推理时随机初始化 → 发沙/气声/时长异常。\n"
            "必须用完整导出重新训练或补键；这不是推理参数能修的。")
    elif len(keys) < 650:
        add(WARN, f"{label}：键数偏少", detail,
            "对照合格值：keys 681 / enc_p 235 / ssl_proj 2 / quantizer 4 / cfm 363")
    else:
        add(PASS, f"{label}：权重完整", detail)


# ---------------------------------------------------------------- 参考音频
def reference_health(model_dir: Path) -> None:
    wav = model_dir / "ref" / "default.wav"
    txt = model_dir / "ref" / "default.txt"
    if not wav.is_file():
        add(FAIL, "参考音频：缺失 ref/default.wav", str(wav),
            "运行 python download_models.py --torch-only")
        return
    audio_health(wav, "参考音频 ref/default.wav")

    if not txt.is_file():
        add(FAIL, "参考文本：缺失 ref/default.txt", str(txt),
            "参考文本必须与音频逐字对应，否则音色/韵律提示失效")
    else:
        text = txt.read_text(encoding="utf-8", errors="replace").strip()
        if text == "":
            add(FAIL, "参考文本：内容为空", str(txt))
        else:
            add(PASS, "参考文本：存在", f"{len(text)} 字：{text[:40]}…")

    styles = model_dir / "ref" / "styles"
    if styles.is_dir():
        names = sorted(p.stem for p in styles.glob("*.wav"))
        add(PASS if names else WARN, "风格参考音频", ", ".join(names) or "(空)")
    else:
        add(WARN, "风格参考音频：目录不存在", str(styles),
            "没有 styles/ 就只能用 normal；运行下载器可补齐")


# ---------------------------------------------------------------- 推理环境
def env_health(gsv_root: Path | None) -> None:
    try:
        import torch
        add(PASS, "torch", f"{torch.__version__}  cuda={torch.cuda.is_available()}")
    except Exception as exc:  # noqa: BLE001
        add(FAIL, "torch：未安装", f"{type(exc).__name__}: {exc}",
            "torch 后端必需：pip install torch")
        return

    if gsv_root is None:
        add(FAIL, "gsv-root：未指定（torch 后端必需）", "",
            "用 --gsv-root 指定 GPT-SoVITS 检出目录")
        return

    required = {
        "TTS 推理代码": gsv_root / "GPT_SoVITS/TTS_infer_pack/TTS.py",
        "v5 底模": gsv_root / "GPT_SoVITS/pretrained_models/gsv-v5-pretrained/s2Gv5dev.pth",
        "v5 vocoder": gsv_root / "GPT_SoVITS/pretrained_models/gsv-v5-pretrained/vocoder.pth",
        "BERT": gsv_root / "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
        "HuBERT": gsv_root / "GPT_SoVITS/pretrained_models/chinese-hubert-base",
    }
    bad = [k for k, p in required.items() if not p.exists()]
    if bad:
        add(FAIL, "gsv-root：缺少必要组件", ", ".join(bad),
            "必须是**支持 v5 的新版** GPT-SoVITS 检出；旧版只有 v2/v2Pro。")
    else:
        add(PASS, "gsv-root：组件齐全", str(gsv_root))

    tts_py = required["TTS 推理代码"]
    if tts_py.exists():
        text = tts_py.read_text(encoding="utf-8", errors="replace")
        if "v5dev" not in text:
            add(FAIL, "推理代码：**不认识 v5dev**", str(tts_py),
                "这是旧版检出，只能跑 v2/v2Pro。请更新 GPT-SoVITS。")
        elif "lora_rank" in text and "_zfy_has_lora" not in text:
            add(WARN, "推理代码：未打全量微调兼容补丁", str(tts_py),
                "加载自训的全量权重会报 KeyError: 'lora_rank'。\n"
                "打补丁：patch_v5_fullft_load.py")
        else:
            add(PASS, "推理代码：支持 v5dev + 全量加载", str(tts_py))


# ---------------------------------------------------------------- 缓存
def cache_health(model_dir: Path) -> None:
    """旧缓存串味：改了模型/参考/风格后仍播到旧结果。"""
    cache = Path(os.path.dirname(str(model_dir))) / "cache"
    if not cache.is_dir():
        add(PASS, "缓存：无（每次都会重新推理）", str(cache))
        return
    files = list(cache.rglob("*"))
    total = sum(f.stat().st_size for f in files if f.is_file()) / 1e6
    add(PASS, "缓存目录", f"{len(files)} 个条目  {total:.1f}MB  {cache}")
    add(WARN, "换了模型/参考后仍听到旧声音 → 先清缓存", "",
        "python -m zfh_voice2 cache --clear\n"
        "或用 TTS(use_cache=False) / POST /cache/clear 对比一次")


# ---------------------------------------------------------------- 采样参数
def sampling_health() -> None:
    try:
        from zfh_voice2.backends.torch_backend import TorchBackend
        d = TorchBackend.DEFAULT_SAMPLING
        add(PASS, "采样参数（默认）",
            f"top_k={d['top_k']} temperature={d['temperature']} "
            f"repetition_penalty={d['repetition_penalty']}",
            )
        if float(d.get("temperature", 1.0)) > 1.2:
            add(WARN, "temperature 偏高", str(d.get("temperature")),
                "会明显增多气声与不稳定；建议 1.0")
    except Exception as exc:  # noqa: BLE001
        add(WARN, "采样参数：无法读取", f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------- 主流程
def main() -> int:
    ap = argparse.ArgumentParser(description="庄方宜语音 v2 一键诊断")
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--gsv-root", default=None)
    ap.add_argument("--wav", action="append", default=[],
                    help="额外检查已生成的 wav（可重复）")
    ap.add_argument("--json", default=None, help="把报告写到这个 JSON 文件")
    a = ap.parse_args()

    print("庄方宜语音 v2 · 诊断")
    print(f"仓库：{HERE}")

    from zfh_voice2 import paths
    try:
        model_dir = Path(paths.resolve_model_dir(a.model_dir))
    except Exception as exc:  # noqa: BLE001
        model_dir = Path(a.model_dir).expanduser() if a.model_dir else None
        if model_dir is None:
            add(FAIL, "模型目录：找不到", f"{type(exc).__name__}: {str(exc)[:200]}",
                "用 --model-dir 指定，或运行 python download_models.py --torch-only")
    gsv_root = Path(a.gsv_root).expanduser() if a.gsv_root else None

    if model_dir:
        section("1. 模型目录")
        print(f"  {model_dir}")
        section("2. SoVITS 权重")
        try:
            from zfh_voice2 import models
            rows = models.scan_v5_models(model_dir)
            if not rows:
                # 常见误用：把权重目录本身当成了模型目录。
                loose = sorted(model_dir.glob("*.pth"))
                hint = "python download_models.py --torch-only"
                if loose:
                    hint = ("检测到目录里直接放着 .pth，但标准布局是\n"
                            f"  {model_dir / 'torch_weights' / 'sovits'}/*.pth\n"
                            f"  以及 {model_dir / 'torch_weights' / 'zfh-e4.ckpt'}\n"
                            "把权重移到 torch_weights/ 下即可；或直接 --model-dir 指向模型根目录。")
                add(FAIL, "没有可用的 v5 权重", str(model_dir / "torch_weights"), hint)
            else:
                add(PASS, f"发现 {len(rows)} 个 v5 权重",
                    "\n".join(f"e{r.epoch:<3d} {r.size/1e6:7.1f}MB  "
                              f"{'✓推荐 ' if r.recommended else '      '}{r.filename}"
                              for r in rows))
                chosen = next((r for r in rows if r.recommended), rows[-1])
                weight_health(Path(chosen.path), f"SoVITS e{chosen.epoch}")
        except Exception as exc:  # noqa: BLE001
            add(FAIL, "权重扫描失败", f"{type(exc).__name__}: {exc}")

        section("3. GPT 权重")
        gpt = model_dir / "torch_weights" / "zfh-e4.ckpt"
        if gpt.is_file():
            add(PASS, "GPT 权重存在", f"{gpt.name}  {gpt.stat().st_size/1e6:.1f}MB")
        else:
            add(FAIL, "GPT 权重缺失", str(gpt),
                "没有它无法合成；python download_models.py --torch-only")

        section("4. 参考音频 / 文本")
        reference_health(model_dir)

        section("5. 缓存")
        cache_health(model_dir)

    section("5. 推理环境")
    env_health(gsv_root)

    section("6. 采样参数")
    sampling_health()

    if a.wav:
        section("7. 已生成音频")
        for i, w in enumerate(a.wav, 1):
            audio_health(Path(w).expanduser(), f"音频 {i}（{Path(w).name}）")

    # ---------- 汇总 ----------
    fails = [r for r in results if r["level"] == FAIL]
    warns = [r for r in results if r["level"] == WARN]
    section("结论")
    print(f"  通过 {len(results)-len(fails)-len(warns)} 项 / 警告 {len(warns)} 项 / 失败 {len(fails)} 项")
    if fails:
        print()
        print("  必须先修：")
        for r in fails:
            print(f"    ✗ {r['title']}")
            if r["fix"]:
                for line in r["fix"].splitlines():
                    print(f"        {line}")
    if warns:
        print()
        print("  建议关注：")
        for r in warns:
            print(f"    ! {r['title']}")

    if a.json:
        Path(a.json).write_text(
            json.dumps({"model_dir": str(model_dir) if model_dir else None,
                        "gsv_root": str(gsv_root) if gsv_root else None,
                        "results": results}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
        print(f"\n  报告已写入 {a.json}")

    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
