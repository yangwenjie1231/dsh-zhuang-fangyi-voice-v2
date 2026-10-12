# -*- coding: utf-8 -*-
"""命令行入口

  python -m zfh_voice2 say "今天天气不错"              # 合成一句
  python -m zfh_voice2 say "..." -o out.wav --seed 7
  python -m zfh_voice2 batch lines.txt -d out/        # 批量（每行一条）
  python -m zfh_voice2 serve --port 8766              # 起 HTTP 服务
  python -m zfh_voice2 status                         # 检查模型是否就绪
  python -m zfh_voice2 cache --clear                  # 查看/清理缓存
"""
import argparse
import json
import os
import sys

# ⚠️ 中文 Windows 控制台默认 GBK：`status` 里打印的 ✓/✗ 会直接
# `UnicodeEncodeError: 'gbk' codec can't encode character '\u2717'` ——
# 一个纯输出问题把整个命令弄崩（而且报错指向 print，看起来像别的毛病）。
# 与 `build_dataset.py` / 上游 synth 脚本同一处理。
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def _build_tts(args):
    from .api import TTS
    kw = {}
    if args.backend == "torch":
        if not args.gsv_root:
            print("错误：torch 后端需要 --gsv-root 指定 GPT-SoVITS 检出目录",
                  file=sys.stderr)
            sys.exit(2)
        kw["gsv_root"] = args.gsv_root
        if args.device:
            kw["device"] = args.device
    # --random-seed 显式要求不固定种子（default_seed=None 会透传给后端）
    dseed = None if getattr(args, "random_seed", False) else 42
    # --brightness-db 不传 → None → 交给 TTS 走 "auto"（按模型版本选校准值）
    br = getattr(args, "brightness_db", None)
    # --style 不传 → None → normal（**与 v1 行为一致**）
    return TTS(backend=args.backend, model_dir=args.model_dir,
               ref_wav=args.ref, ref_text=args.ref_text,
               style=getattr(args, "style", None),
               use_cache=not args.no_cache,
               default_seed=dseed,
               brightness_db=("auto" if br is None else br),
               localize=(args.localize if args.localize is not None else "auto"),
               **kw)


# CUDA / DirectML 下每次冷启动都要重付"加载 + 预热"的代价
# （本机实测：加载 3.7s、首句预热约 15s）。所以提供两条省时路径：
#   1) serve --idle-timeout  让服务常驻，空闲超时才释放
#   2) say --server          一次性命令直接复用已在运行的服务
DEFAULT_SERVER = "http://127.0.0.1:8766"


def _server_post(server, path, payload=None, timeout=600):
    """用标准库 POST，避免引入额外依赖"""
    import urllib.error
    import urllib.request
    url = server.rstrip("/") + path
    data = json.dumps(payload or {}).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()
    except urllib.error.URLError as e:
        raise RuntimeError(f"无法连接常驻服务 {server}：{e}") from e


def _server_alive(server, timeout=2):
    try:
        import urllib.request
        with urllib.request.urlopen(server.rstrip("/") + "/health",
                                    timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def _remote_say(server, text, seed, out=None, brightness_db=None):
    """通过常驻服务合成，返回 (wav_bytes, out_path)

    `brightness_db=None` → 请求体里**不带**该字段，让服务端用它自己的
    默认值（同样是 auto，按它加载的模型版本选）。
    传数字（含 0）才会显式覆盖。
    """
    payload = {"text": text, "seed": seed}
    if brightness_db is not None:
        payload["brightness_db"] = brightness_db
    audio = _server_post(server, "/tts", payload)
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
        with open(out, "wb") as f:
            f.write(audio)
    return audio, out


def cmd_say(args):
    text = " ".join(args.text).strip()
    if not text:
        print("错误：文本为空", file=sys.stderr)
        return 2
    out = args.out or os.path.join(args.outdir, "out.wav")

    # 走常驻服务：省掉每次冷启动的加载/预热
    server = _resolve_server(args)
    if server:
        try:
            audio, _ = _remote_say(server, text, args.seed, out,
                                   brightness_db=args.brightness_db)
            print(f"via {server}  {len(audio)/1024:.0f} KB  ->  "
                  f"{os.path.abspath(out)}")
            return 0
        except RuntimeError as e:
            if args.server == "auto":
                print(f"（常驻服务不可用，回退到本地加载：{e}）",
                      file=sys.stderr)
            else:
                print(f"错误：{e}", file=sys.stderr)
                return 3

    tts = _build_tts(args)
    r = tts.say(text, seed=args.seed, out=out, verbose=True)
    # 英文被转成中文读法时告诉用户实际念的是什么 ——
    # 否则用户听到"基皮尤"会以为念错了，其实是输入里写的 GPU
    if getattr(r, "spoken", text) != text:
        print(f"  实际合成: {r.spoken}")
    print(f"{r.duration:.2f}s  ->  {os.path.abspath(out)}")
    return 0


def _resolve_server(args):
    """决定是否走常驻服务：显式 --server URL / --server auto 探测 / 或 None"""
    s = getattr(args, "server", None)
    if not s:
        return None
    if s == "auto":
        return DEFAULT_SERVER if _server_alive(DEFAULT_SERVER) else None
    return s


def cmd_batch(args):
    text_s = [l.strip() for l in open(args.file, encoding="utf-8")] \
        if os.path.exists(args.file) else []
    if not text_s:
        print(f"错误：文件不存在或为空 {args.file}", file=sys.stderr)
        return 2
    texts = [l for l in text_s if l]

    server = _resolve_server(args)
    if server:
        os.makedirs(args.outdir, exist_ok=True)
        print(f"共 {len(texts)} 条 → {args.outdir}  (via {server})")
        ok = 0
        for i, t in enumerate(texts, 1):
            p = os.path.join(args.outdir, f"{i:03d}.wav")
            try:
                audio, _ = _remote_say(server, t, args.seed, p,
                                       brightness_db=args.brightness_db)
                ok += 1
                print(f"  [{i}/{len(texts)}] {len(audio)/1024:6.0f} KB  {t[:30]}")
            except RuntimeError as e:
                print(f"  [{i}/{len(texts)}] 失败: {e}")
        print(f"\n完成 {ok}/{len(texts)} 条")
        return 0 if ok == len(texts) else 1

    tts = _build_tts(args)
    print(f"共 {len(texts)} 条 → {args.outdir}")
    rs = tts.say_many(texts, args.outdir, seed=args.seed)
    total = sum(r.duration for r in rs)
    hit = sum(1 for r in rs if r.cached)
    print(f"\n完成 {len(rs)} 条，音频合计 {total:.1f}s，缓存命中 {hit}")
    return 0


def cmd_serve(args):
    from .server import serve
    tts = _build_tts(args)
    serve(tts, host=args.host, port=args.port,
          idle_timeout=getattr(args, "idle_timeout", 0),
          preload=not getattr(args, "no_preload", False))
    return 0


def cmd_doctor(args):
    from .doctor import main as doctor_main
    return doctor_main(args.model_dir, json_path=getattr(args, "json", None))


def cmd_status(args):
    from .paths import has_onnx, has_torch_weights, model_status, repo_root
    d, missing, ok = model_status(args.model_dir, backend=args.backend)
    print(f"仓库根目录 : {repo_root()}")
    print(f"模型目录   : {d}")
    print(f"后端       : {args.backend}")
    print(f"状态       : {'✓ 就绪' if ok else f'✗ 缺 {len(missing)} 个文件'}")
    # 顺带告知另一类模型是否也在，便于判断能否切换后端
    print(f"  另有 ONNX 模型      : {'是' if has_onnx(d) else '否'}")
    print(f"  另有 torch 权重     : {'是' if has_torch_weights(d) else '否'}")
    # v2：列出音色风格（不加载模型，纯查文件）
    from . import styles as _styles
    print("  音色风格 (v2)       :")
    for s in _styles.list_styles(d):
        mark = "✓" if s["available"] else "✗"
        star = " (默认)" if s["is_default"] else ""
        print(f"    {mark} {s['name']:8s} {s['label']:8s}{star}  {s['desc']}")
    if missing:
        for m in missing[:12]:
            print(f"  缺: {m}")
        if len(missing) > 12:
            print(f"  ... 共 {len(missing)} 项")
        if args.backend == "onnx":
            print("\n运行 `python download_models.py` 下载 ONNX 模型")
        else:
            print("\n运行 `python download_models.py --with-torch` 下载 torch 权重")
            print("（torch 后端不要求 ONNX 模型；两者互不通用）")
    return 0 if ok else 1


def cmd_styles(args):
    """列出音色风格（v2 新增）。

    **不加载模型** —— 只查文件，所以秒回。
    """
    import json as _json

    from . import paths as _paths
    from . import styles as _styles
    try:
        d = _paths.resolve_model_dir(args.model_dir, must_exist=False)
    except Exception:
        d = args.model_dir
    items = _styles.list_styles(d)
    if getattr(args, "json", False):
        print(_json.dumps({"model_dir": d, "styles": items},
                          ensure_ascii=False, indent=2))
        return 0
    print(f"模型目录 : {d}")
    print(f"默认风格 : {_styles.DEFAULT_STYLE}")
    print()
    print(f"  {'':2s} {'名称':10s} {'标签':10s} {'参考音频':28s} 说明")
    for s in items:
        mark = "✓" if s["available"] else "✗"
        star = " *" if s["is_default"] else "  "
        ref = s["resolved"] or "(缺)"
        print(f"  {mark}{star} {s['name']:10s} {s['label']:10s} "
              f"{os.path.basename(ref):28s} {s['desc']}")
    print()
    print("用法: python -m zfh_voice2 say \"你好\" --style firm")
    print("      POST /tts  {\"text\":\"...\",\"style\":\"firm\"}")
    return 0


def cmd_cache(args):
    from .api import TTS
    tts = _build_tts(args)
    if args.clear:
        n = tts.clear_cache()
        print(f"已清理 {n} 个缓存文件")
    else:
        files = [f for f in os.listdir(tts.cache_dir) if f.endswith(".wav")]
        size = sum(os.path.getsize(os.path.join(tts.cache_dir, f))
                   for f in files)
        print(f"缓存目录: {tts.cache_dir}")
        print(f"条目 {len(files)} 个，合计 {size/1024/1024:.1f} MB")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="zfh-voice2", description="庄方宜专属音色合成")
    ap.add_argument("--backend", default="onnx", choices=["onnx", "torch"],
                    help="推理后端（默认 onnx）")
    ap.add_argument("--model-dir", default=None, help="模型目录")
    ap.add_argument("--ref", default=None, help="参考音频路径")
    ap.add_argument("--ref-text", default=None, help="参考音频对应文本")
    ap.add_argument("--style", default=None,
                    help="音色风格（v2）：normal（默认，正常说话）/ "
                         "firm（语气强硬）/ gentle（语音娇）。"
                         "不传 = normal，**与 v1 行为一致**。"
                         "显式传 --ref 时 --ref 优先于 --style")
    ap.add_argument("--seed", type=int, default=None,
                    help="随机种子。不传用默认 42（短句更稳，可复现）")
    ap.add_argument("--random-seed", action="store_true",
                    help="不固定种子，走上游随机采样。"
                         "**不推荐**：短句会不稳定（实测 4 字句时长波动 23.5%%，"
                         "偶发截断到正常时长的一半）")
    ap.add_argument("--brightness-db", type=float, default=None,
                    help="亮度补偿（dB），补偿 48k→32k 重采样在 12–16kHz 的"
                         "滤波滚降。**不传 = auto**（按模型版本选校准值："
                         "v2 → 3，v2Pro → 0）；传 0 则显式关闭。")
    ap.add_argument("--no-cache", action="store_true", help="禁用合成缓存")
    ap.add_argument("--localize", dest="localize", action="store_true",
                    default=None,
                    help="把英文转成中文读法再合成（GPU→基皮尤）。"
                         "默认 auto：中英混排可用时用真英文发音，"
                         "不可用时自动转中文读法兜底")
    ap.add_argument("--no-localize", dest="localize", action="store_false",
                    help="总是不转（英文交给 LangSegmenter 分段，"
                         "用真英文音素念）")
    ap.add_argument("--gsv-root", default=None,
                    help="GPT-SoVITS 检出目录（torch 后端需要）")
    ap.add_argument("--device", default=None, help="torch 后端设备: cuda/cpu")
    ap.add_argument("--outdir", default="out", help="默认输出目录")

    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("say", help="合成一句")
    p.add_argument("text", nargs="+")
    p.add_argument("-o", "--out", default=None)
    p.add_argument("--server", default=None, nargs="?", const="auto",
                   help="复用已在运行的常驻服务，省掉冷启动加载。"
                        "给 URL 则连该地址；不带值(或 auto)则自动探测 "
                        f"{DEFAULT_SERVER}")
    p.set_defaults(func=cmd_say)

    p = sub.add_parser("batch", help="批量合成（每行一条）")
    p.add_argument("file")
    p.add_argument("-d", "--outdir", default="out")
    p.add_argument("--server", default=None, nargs="?", const="auto",
                   help="同 say --server")
    p.set_defaults(func=cmd_batch)

    p = sub.add_parser("serve", help="启动 HTTP 常驻服务")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8766)
    p.add_argument("--idle-timeout", type=float, default=0,
                   help="空闲多少秒后自动释放模型（腾出显存）；"
                        "0 = 一直常驻（默认）")
    p.add_argument("--no-preload", action="store_true",
                   help="启动时不预热，模型推迟到首次请求才加载")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("doctor", help="环境体检：该装哪套模型")
    p.add_argument("--json", default=None, metavar="PATH",
                   help="把体检结果写成 JSON（供程序读取，不打印报告）")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("status", help="检查模型是否就绪")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("styles", help="列出音色风格（v2）")
    p.add_argument("--json", action="store_true", help="输出 JSON")
    p.set_defaults(func=cmd_styles)

    p = sub.add_parser("cache", help="查看/清理缓存")
    p.add_argument("--clear", action="store_true")
    p.set_defaults(func=cmd_cache)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
