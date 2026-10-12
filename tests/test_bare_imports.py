# -*- coding: utf-8 -*-
"""裸环境守卫：诊断类命令必须在**没有任何第三方依赖**时也能跑

背景（真事故）：
    `zfh_voice/__init__.py` 曾急切 `from .api import TTS`，
    而 `api` → `audio` / `backends.base` 模块级 `import numpy`。
    结果在没装依赖的机器上：

        python -m zfh_voice doctor   →   ModuleNotFoundError: numpy

    而 doctor 存在的唯一意义就是"告诉用户缺什么"——它偏偏在
    最需要它的时候（什么都没装）崩掉。

怎么模拟"裸机器"：
    用 `python -S` 启动子进程 —— 不加载 `site`，于是 `site-packages`
    不在 `sys.path` 里，第三方包**真的**找不到。

    （注意：**不要**用"自定义 finder 抛 ImportError"来模拟 ——
     真实的裸机器上 `importlib.util.find_spec` 只是返回 None，
     不会抛。用抛异常模拟会测出假失败。）
"""
import os
import subprocess
import sys
import textwrap

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "src")

BODY = textwrap.dedent('''
    import sys
    sys.path.insert(0, {src!r})
    {body}
''')

# 裸环境下应能正常使用的第三方包（一个都不该被 import 到）
SHOULD_BE_ABSENT = ("numpy", "onnxruntime", "torch", "transformers",
                    "soundfile", "pypinyin", "cn2an", "jieba")


def _run(args, model_dir):
    """在**真正的裸环境**（-S，无 site-packages）里跑 CLI"""
    script = BODY.format(src=SRC, body=textwrap.dedent('''
        from zfh_voice.cli import main
        sys.exit(main(sys.argv[1:]))
    '''))
    env = dict(os.environ)
    env["PYTHONPATH"] = SRC
    env["PYTHONIOENCODING"] = "utf-8"
    env["ZFH_MODEL_DIR"] = model_dir
    r = subprocess.run(
        [sys.executable, "-S", "-c", script] + args,
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        env=env, timeout=180)
    return r


def _run_py(body, args=()):
    script = BODY.format(src=SRC, body=textwrap.dedent(body))
    env = dict(os.environ)
    env["PYTHONPATH"] = SRC
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run(
        [sys.executable, "-S", "-c", script] + list(args),
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        env=env, timeout=180)


@pytest.fixture
def empty_dir(tmp_path):
    d = tmp_path / "nope"
    d.mkdir()
    return str(d)


def test_sandbox_really_hides_third_party():
    """先证明"裸环境"这个前提成立，否则后面全是假绿"""
    r = _run_py('''
        import importlib.util
        present = [m for m in %r if importlib.util.find_spec(m) is not None]
        if present:
            raise SystemExit("这些包在裸环境里仍可见: " + ", ".join(present))
        print("BARE-OK")
    ''' % (SHOULD_BE_ABSENT,))
    assert "BARE-OK" in r.stdout, (
        f"裸环境构造失败，测试无意义：\nstdout={r.stdout}\nstderr={r.stderr}")


def test_doctor_works_without_any_dependency(empty_dir):
    """★ 核心回归：doctor 必须在裸环境下可用"""
    r = _run(["doctor"], empty_dir)
    assert r.returncode == 0, f"doctor 在裸环境失败：\n{r.stderr}"
    assert "ModuleNotFoundError" not in r.stderr
    assert "环境体检" in r.stdout
    assert "建议" in r.stdout
    # 应该如实报告"全都缺"，而不是崩掉
    assert "缺" in r.stdout


def test_status_works_without_any_dependency(empty_dir):
    r = _run(["status"], empty_dir)
    assert r.returncode in (0, 1), f"status 异常退出：\n{r.stderr}"
    assert "ModuleNotFoundError" not in r.stderr
    assert "模型目录" in r.stdout


def test_help_works_without_any_dependency(empty_dir):
    r = _run(["--help"], empty_dir)
    assert r.returncode == 0, f"--help 在裸环境失败：\n{r.stderr}"
    assert "doctor" in r.stdout and "status" in r.stdout


def test_import_package_alone_is_cheap():
    """`import zfh_voice` 本身不该拉起重依赖"""
    r = _run_py('''
        import zfh_voice
        assert zfh_voice.__version__
        assert callable(zfh_voice.resolve_model_dir)
        print("IMPORT-OK")
    ''')
    assert r.returncode == 0, f"裸 import 失败：\n{r.stderr}"
    assert "IMPORT-OK" in r.stdout


def test_lazy_attr_still_exposes_api():
    """懒加载不能把公开接口弄丢（正常环境下 TTS 仍可用）"""
    script = textwrap.dedent(f'''
        import sys
        sys.path.insert(0, {SRC!r})
        import zfh_voice
        assert callable(zfh_voice.TTS), "TTS 未暴露"
        assert callable(zfh_voice.SynthResult)
        assert "TTS" in dir(zfh_voice)
        try:
            zfh_voice.NoSuchThing
        except AttributeError:
            pass
        else:
            raise AssertionError("未知属性应抛 AttributeError")
        print("LAZY-OK")
    ''')
    r = subprocess.run([sys.executable, "-c", script],
                       capture_output=True, text=True, encoding="utf-8", errors="replace",
                       timeout=180)
    assert r.returncode == 0, r.stderr
    assert "LAZY-OK" in r.stdout


def test_doctor_survives_broken_find_spec():
    """find_spec 抛异常时应报"缺"而不是把体检工具弄崩"""
    r = _run_py('''
        import sys, importlib.util
        _orig = importlib.util.find_spec
        def boom(name, *a, **k):
            raise ImportError("simulated broken install: " + name)
        importlib.util.find_spec = boom
        from zfh_voice.doctor import has_module
        assert has_module("numpy") is False
        print("ROBUST-OK")
    ''')
    assert r.returncode == 0, f"has_module 未容错：\n{r.stderr}"
    assert "ROBUST-OK" in r.stdout


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
