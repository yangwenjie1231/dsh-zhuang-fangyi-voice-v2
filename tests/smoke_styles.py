# -*- coding: utf-8 -*-
"""冒烟测试：styles 模块（不需要模型权重）。"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from zfh_voice2 import styles  # noqa: E402

print("=" * 80)
print("styles 模块冒烟测试")
print("=" * 80)

print()
print("① 三种风格定义")
for s in styles.list_styles():
    print(f"  {s['name']:8s} {s['label']:8s} "
          f"available={s['available']!s:5s} default={s['is_default']!s:5s}")
    print(f"           {s['desc']}")

print()
print("② 规范名解析（未知 -> normal，不抛）")
cases = [None, "", "normal", "firm", "gentle", "FIRM", "  Firm  ",
         "hard", "strong", "soft", "sweet", "default", "unknown", 123, []]
for raw in cases:
    try:
        got = styles.normalize_style(raw)
        print(f"    {str(raw):14s} -> {got}")
    except Exception as e:  # noqa: BLE001
        print(f"    {str(raw):14s} -> **抛异常 {type(e).__name__}**")

print()
print("③ 风格文本（参考文本必须与音频严格对应，内置兜底）")
for k in ("normal", "firm", "gentle"):
    t = styles.style_text(k)
    print(f"    {k:8s} {(t[:52] + '…') if t and len(t) > 52 else (t or '(无)')}")

print()
print("④ 模型目录不存在时的行为（应返回 None，不抛）")
for md in (None, "/nonexistent/path", "D:/nope"):
    try:
        key, wav, txt = styles.resolve_style("firm", md)
        print(f"    model_dir={str(md):22s} -> key={key} wav={wav} "
              f"txt={'有' if txt else '无'}")
    except Exception as e:  # noqa: BLE001
        print(f"    model_dir={str(md):22s} -> **抛 {type(e).__name__}**")

print()
print("⑤ 默认风格必须是 normal（保证 v1 行为不变）")
assert styles.DEFAULT_STYLE == "normal", "默认风格被改了！"
assert styles.normalize_style(None) == "normal"
assert styles.normalize_style("bogus") == "normal"
print("    ✓ DEFAULT_STYLE == 'normal'")
print("    ✓ 未知输入回落 normal")

print()
print("=" * 80)
print("  ✓ 冒烟测试通过")
print("=" * 80)
