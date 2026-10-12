# -*- coding: utf-8 -*-
"""打包清单守卫：随包分发的数据文件必须真的进包

背景（真事故）：
    `pyproject.toml` 的 package-data 只写了 `_vendor/nltk_data/**/*.json`，
    而 `corpora/cmudict/cmudict` **没有扩展名**，于是它不会被打包。

    后果是静默的：源码树里英文正常（文件在），但**用户装完包英文就废了** ——
    `g2p_en` 的 `from nltk.corpus import cmudict` 找不到资源，而 nltk 的兜底
    是 `nltk.download('cmudict')`，在受限网络/SSRF 策略下会被直接挡住。

这一条测试用**真的构建产物**来守：读 pyproject 的 package-data，按
setuptools 的 glob 语义展开，断言关键文件都在。
（比"构建一次 sdist"快得多，且失败信息更直接。）
"""
import glob
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENDOR = os.path.join(REPO, "src", "zfh_voice", "_vendor")

# 必须随包分发、缺了就静默降级的关键文件
REQUIRED = [
    # 英文 G2P：g2p_en 依赖 nltk.corpus.cmudict
    "_vendor/nltk_data/corpora/cmudict/cmudict",
    "_vendor/nltk_data/corpora/cmudict/cmudict.phones",
    # 词性标注：english.py 用 nltk.pos_tag
    "_vendor/nltk_data/taggers/averaged_perceptron_tagger_eng/"
    "averaged_perceptron_tagger_eng.weights.json",
    # 上游英文词典（english.py 直接读）
    "_vendor/_gsv_text/cmudict.rep",
]


def _package_data_globs():
    """从 pyproject.toml 读 package-data["zfh_voice"]。

    不依赖 tomllib（py3.11+ 才有），手写一个够用的解析：
    只取 `[tool.setuptools.package-data]` 段里 `"zfh_voice" = [ ... ]` 的字符串。
    """
    path = os.path.join(REPO, "pyproject.toml")
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()

    in_section = False
    in_list = False
    out = []
    for line in lines:
        s = line.strip()
        if s.startswith("[") and s.endswith("]"):
            in_section = s == "[tool.setuptools.package-data]"
            in_list = False
            continue
        if not in_section:
            continue
        if not in_list:
            if s.startswith('"zfh_voice"') and "[" in s:
                in_list = True
                rest = s.split("[", 1)[1]
                if rest.strip().startswith('"'):
                    out.append(rest.strip().strip('",'))
            continue
        if s.startswith("]"):
            break
        if s.startswith('"'):
            out.append(s.strip().strip('",'))
    return out


def test_package_data_is_parseable():
    globs = _package_data_globs()
    assert globs, "没解析出任何 package-data 规则，解析器或 pyproject 有问题"
    assert any("nltk_data" in g for g in globs), \
        f"没找到 nltk_data 规则: {globs}"


def test_required_files_match_package_data():
    """每个必需文件都要被至少一条 package-data 规则覆盖"""
    globs = _package_data_globs()
    src_root = os.path.join(REPO, "src", "zfh_voice")
    missing = []
    for rel in REQUIRED:
        abs_p = os.path.join(REPO, "src", "zfh_voice", rel.replace("/", os.sep))
        if not os.path.exists(abs_p):
            missing.append(f"{rel}  (源码树里就不存在)")
            continue
        hit = False
        for g in globs:
            # setuptools 的 glob 以包目录为根；`**/*` 也要能匹配到子目录文件
            for cand in (g, g.rstrip("/") + "/**", g.replace("/**/*", "/**/*")):
                pat = os.path.join(src_root, cand.replace("/", os.sep))
                if glob.glob(pat, recursive=True):
                    if abs_p in {os.path.abspath(x) for x in glob.glob(pat, recursive=True)}:
                        hit = True
                        break
            if hit:
                break
        if not hit:
            missing.append(f"{rel}  (没有 package-data 规则覆盖它)")
    assert not missing, (
        "这些文件不会被装进包里（用户装上就缺）：\n  " + "\n  ".join(missing) +
        "\n\n提示：无扩展名文件需要 `**/*` 之类的规则，只写 `**/*.json` 会漏掉。"
    )


def test_extensionless_cmudict_is_covered():
    """专门的回归：`corpora/cmudict/cmudict` 没有扩展名，曾被漏掉"""
    globs = _package_data_globs()
    src_root = os.path.join(REPO, "src", "zfh_voice")
    target = os.path.abspath(
        os.path.join(src_root, "_vendor", "nltk_data", "corpora", "cmudict", "cmudict"))
    assert os.path.exists(target), "源码树里缺 cmudict，先修数据"

    covered = False
    for g in globs:
        pat = os.path.join(src_root, g.replace("/", os.sep))
        if target in {os.path.abspath(x) for x in glob.glob(pat, recursive=True)}:
            covered = True
            break
    assert covered, (
        "`corpora/cmudict/cmudict` 没被打包规则覆盖 —— "
        "装包后 g2p_en 找不到 cmudict，英文会静默失效。"
        "把规则改成 `_vendor/nltk_data/**/*`。"
    )
