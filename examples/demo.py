# -*- coding: utf-8 -*-
"""示例：三种用法

运行前先：
    pip install -r requirements.txt
    python download_models.py

    python examples/demo.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

from zfh_voice import TTS  # noqa: E402


def main():
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
    os.makedirs(out, exist_ok=True)

    tts = TTS()          # 默认 onnx 后端；有 torch 环境时可用 backend="torch"
    print(f"后端: {tts.backend.name}  参考音频: {os.path.basename(tts.ref_wav)}")

    # 1) 单句
    r = tts.say("管理员，今天的实验数据我已经整理好了，你要不要看看？")
    p = r.save(os.path.join(out, "01_daily.wav"))
    print(f"单句  {r.duration:.2f}s  ->  {p}")

    # 2) 换随机种子再生成一版（同一句话的不同读法）
    r2 = tts.say("今天天气不错，我们一起出去走走吧。", seed=7)
    print(f"带seed {r2.duration:.2f}s  cached={r2.cached}")

    # 3) 批量
    lines = [
        "该休息一下了，眼睛也要歇歇。",
        "任务完成，辛苦啦。",
        "嗯……这个问题，我再想想。",
    ]
    rs = tts.say_many(lines, os.path.join(out, "batch"))
    print(f"批量  {len(rs)} 条，合计 {sum(x.duration for x in rs):.1f}s")

    # 4) 不落盘，直接拿字节（可发给播放器 / HTTP 上传）
    data = r.to_wav_bytes()
    print(f"wav 字节数: {len(data)}")

    tts.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
