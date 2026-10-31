# GPT-SoVITS v5 Model Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `dsh-zhuang-fangyi-voice-v2` a v5-only plugin that downloads, discovers, displays, selects, and runs the four GPT-SoVITS v5 full-FT weights distributed for the plugin.

**Architecture:** A focused Python `models.py` module owns v5 header parsing, recursive discovery, default selection, and containment checks. The existing torch backend consumes the selected path and rejects non-v5 models; the HTTP/service layer exposes discovery and switching. The downloader targets the new clean ModelScope model repo and GitHub Release, while the DSH settings surface uses the service API rather than duplicating scanning logic.

**Tech Stack:** Python 3.9+, Node.js 20 ESM, GPT-SoVITS `TTS.run`, DSH/Cordis plugin API, pytest, Node test runner, ModelScope/GitHub HTTP assets.

---

## File Map

- Create `src/zfh_voice2/models.py`: v5 weight inspection, recursive scanning, configuration default, sorting and containment.
- Create `tests/test_v5_models.py`: unit tests for header parsing, scanning, defaults and rejection.
- Modify `src/zfh_voice2/backends/torch_backend.py`: v5-only selection and version verification.
- Modify `src/zfh_voice2/api.py`: pass explicit SoVITS/GPT choices and expose models.
- Modify `src/zfh_voice2/server.py`: `/v2/models`, `/v2/model`, `/v2/models/rescan`.
- Modify `src/zfh_voice2/cli.py`: list/select diagnostics and v5-only messaging.
- Modify `index.js`: configuration, model service methods, switching lifecycle.
- Modify `tools/test-plugin.mjs`: JS service/config/switch tests.
- Modify `download_models.py`: new repositories, epoch selection, manifest validation and atomic downloads.
- Create `tests/test_download_v5.py`: download plan and manifest checks.
- Modify `cordis.patch.yml`: v5-only defaults and model selector registration/fields.
- Modify `README.md`, `CHANGELOG.md`, `docs/安装提示词.md`: user workflow and migration boundary.
- Modify version declarations in `package.json`, `pyproject.toml`, `src/zfh_voice2/__init__.py`.

---

### Task 1: v5 Weight Discovery Core

**Files:**
- Create: `src/zfh_voice2/models.py`
- Create: `tests/test_v5_models.py`

- [ ] **Step 1: Write failing header/discovery tests**

Add tests that create fake `.pth` files whose first two bytes are `07`, `08`, `05`, or `PK`; assert only `07` appears. Cover nested `torch_weights/sovits/`, flat `torch_weights/`, epoch/step parsing, size, sorting, and recommended e32.

```python
def test_scan_accepts_only_v5dev(tmp_path):
    tw = tmp_path / "torch_weights" / "sovits"
    tw.mkdir(parents=True)
    (tw / "v5_full_e32_final_e8_s432_l0.pth").write_bytes(b"07payload")
    (tw / "v5_full_e32_final_e32_s1728_l0.pth").write_bytes(b"07payload")
    (tw / "old_v2pro_e12_s384_l0.pth").write_bytes(b"05payload")
    rows = scan_v5_models(tmp_path)
    assert [r.epoch for r in rows] == [8, 32]
    assert rows[-1].recommended is True
```

- [ ] **Step 2: Run the focused test and verify failure**

Run:

```bash
python -m pytest tests/test_v5_models.py -q
```

Expected: import failure because `zfh_voice2.models` does not exist.

- [ ] **Step 3: Implement the discovery module**

Implement immutable `V5Model` records and:

```python
inspect_v5_weight(path) -> V5Model | None
scan_v5_models(model_dir) -> list[V5Model]
resolve_selected_model(model_dir, explicit=None) -> V5Model
assert_model_contained(model_dir, path) -> str
```

Read at most two bytes plus stat metadata; never deserialize weights during discovery. Read `configuration.json.default_sovits`, normalize it relative to `model_dir`, and otherwise choose maximum epoch.

- [ ] **Step 4: Run discovery tests**

Run:

```bash
python -m pytest tests/test_v5_models.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/zfh_voice2/models.py tests/test_v5_models.py
git commit -m "feat(models): discover GPT-SoVITS v5 weights"
```

---

### Task 2: v5-Only Torch Backend

**Files:**
- Modify: `src/zfh_voice2/backends/torch_backend.py`
- Modify: `src/zfh_voice2/api.py`
- Create: `tests/test_torch_v5.py`

- [ ] **Step 1: Write failing backend selection tests**

Mock the upstream TTS constructor and assert:

- nested e32 is selected by default;
- explicit e16 wins;
- config version passed to upstream is `v5dev`;
- loaded version must be `v5dev`;
- `05`/v2Pro gives an error directing the user to v1;
- missing `gsv-v5-pretrained/s2Gv5dev.pth` or `vocoder.pth` is explicit.

- [ ] **Step 2: Run the focused tests and verify failure**

```bash
python -m pytest tests/test_torch_v5.py -q
```

Expected: failures because the backend still scans only the top level and passes `version: "v2"`.

- [ ] **Step 3: Implement v5 loading**

Use `resolve_selected_model`; set:

```python
cfg = {"custom": {
    "device": device,
    "is_half": is_half,
    "version": "v5dev",
    "t2s_weights_path": gpt,
    "vits_weights_path": selected.path,
    "bert_base_path": bert,
    "cnhuhbert_base_path": hubert,
}}
```

After construction, require `str(self.tts.configs.version) == "v5dev"`; keep 48k output untouched. Add `self.selected_model` metadata.

- [ ] **Step 4: Run backend and existing tests**

```bash
python -m pytest tests/test_torch_v5.py tests/test_torch_sampling.py tests/test_paths.py -q
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/zfh_voice2/backends/torch_backend.py src/zfh_voice2/api.py tests/test_torch_v5.py
git commit -m "feat(torch): load v5 full-FT weights only"
```

---

### Task 3: Python HTTP Model API

**Files:**
- Modify: `src/zfh_voice2/api.py`
- Modify: `src/zfh_voice2/server.py`
- Modify: `src/zfh_voice2/cli.py`
- Create: `tests/test_model_api.py`

- [ ] **Step 1: Write failing API tests**

Exercise direct TTS model listing and HTTP handlers:

```text
GET  /v2/models
POST /v2/model {"model":"...e16...pth"}
POST /v2/models/rescan
```

Assert path containment, v5-only validation, selected metadata, and that v1/unprefixed endpoints remain unchanged.

- [ ] **Step 2: Run focused tests**

```bash
python -m pytest tests/test_model_api.py -q
```

Expected: routes/methods absent.

- [ ] **Step 3: Implement API state and endpoints**

Add `TTS.list_models()`, `TTS.set_model()`, and `TTS.rescan_models()`. Switching unloads the current backend and lazily recreates it with the selected path. Reject invalid requests with HTTP 400 and retain the previous choice.

- [ ] **Step 4: Add CLI diagnostics**

Add:

```bash
python -m zfh_voice2 models
python -m zfh_voice2 model <id-or-path>
```

The list command must not load a 766 MB model.

- [ ] **Step 5: Run API/regression tests**

```bash
python -m pytest tests/test_model_api.py tests/test_smoke.py tests/test_v1_compat.py -q
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/zfh_voice2/api.py src/zfh_voice2/server.py src/zfh_voice2/cli.py tests/test_model_api.py
git commit -m "feat(api): expose v5 model discovery and switching"
```

---

### Task 4: DSH Service and Settings Selection

**Files:**
- Modify: `index.js`
- Modify: `cordis.patch.yml`
- Modify: `tools/test-plugin.mjs`

- [ ] **Step 1: Write failing JS tests**

Add assertions for `sovitsPath` normalization, serve command propagation, service methods `models`, `setModel`, `rescanModels`, and stop-before-switch behavior. Assert failed switching keeps the previous configured path.

- [ ] **Step 2: Run Node tests and verify failure**

```bash
node tools/test-plugin.mjs
```

Expected: new assertions fail while existing 77 remain passing.

- [ ] **Step 3: Implement configuration and service methods**

Add `sovitsPath` and `gptPath` to `defaultConfig`, `normalizeConfig`, `serveOptions`, and command construction. Add HTTP client methods and service wrappers. `setModel` stops only the process owned by this plugin, updates config after server validation, then leaves restart lazy.

- [ ] **Step 4: Implement settings UI integration**

First inspect the installed DSH Loader configuration API. If dynamic enum providers are supported, bind them to `models()`. Otherwise register a scoped plugin settings panel with:

```text
model list, recommended badge, selected badge, refresh button, select button
```

Do not add a separate scanner in the UI.

- [ ] **Step 5: Run syntax and JS tests**

```bash
node --check index.js
node tools/test-plugin.mjs
```

Expected: zero failures.

- [ ] **Step 6: Commit**

```bash
git add index.js cordis.patch.yml tools/test-plugin.mjs
git commit -m "feat(plugin): add v5 model selector"
```

---

### Task 5: v5 Model Downloader

**Files:**
- Modify: `download_models.py`
- Create: `tests/test_download_v5.py`

- [ ] **Step 1: Write failing download-plan tests**

Assert source bases:

```text
https://modelscope.cn/models/yangwenjie1231/dsh-zhuang-fangyi-voice-v2-model/resolve/master
https://github.com/yangwenjie1231/dsh-zhuang-fangyi-voice-v2/releases/download/model-v2-v5-fullft
```

Test default e32, `--epoch`, `--all-epochs`, resources, manifest SHA256, `.part`, and no ONNX assets.

- [ ] **Step 2: Run and verify failure**

```bash
python -m pytest tests/test_download_v5.py -q
```

Expected: existing downloader points at old repositories/assets.

- [ ] **Step 3: Implement v5-only download plan**

Default assets:

```text
torch_weights/zfh-e4.ckpt
torch_weights/sovits/v5_full_e32_final_e32_s1728_l0.pth
ref/default.wav
ref/default.txt
ref/styles/*
configuration.json
distribution_manifest.json
SHA256SUMS.txt
```

Map GitHub flat release filenames to nested local paths. Download to `.part`, stream SHA256, then `os.replace`.

- [ ] **Step 4: Run downloader tests and local check mode**

```bash
python -m pytest tests/test_download_v5.py -q
python download_models.py --check --dir <fixture-dir>
```

Expected: pass and report selected/default epoch.

- [ ] **Step 5: Commit**

```bash
git add download_models.py tests/test_download_v5.py
git commit -m "feat(download): fetch v5 plugin model package"
```

---

### Task 6: Documentation, Versioning, and Full Regression

**Files:**
- Modify: `README.md`
- Modify: `CHANGELOG.md`
- Modify: `docs/安装提示词.md`
- Modify: `package.json`
- Modify: `pyproject.toml`
- Modify: `src/zfh_voice2/__init__.py`

- [ ] **Step 1: Update user documentation**

Document v1/v2 model boundary, new clean model repository, four epoch choices, settings selection, v5 `gsvRoot` prerequisites, download commands, 48k output, and migration from any old v2Pro directory.

- [ ] **Step 2: Bump version consistently**

Set all three declarations to `0.3.0` and add changelog entry “v5-only model selection”.

- [ ] **Step 3: Run the complete Python suite**

```bash
PYTHONPATH=src python -m pytest tests -q --no-header
```

Expected: all tests pass; only documented model-dependent skips remain.

- [ ] **Step 4: Run complete Node and compatibility suites**

```bash
node --check index.js
node tools/test-plugin.mjs
PYTHONPATH=src python tests/test_v1_compat.py
PYTHONPATH=src python tests/smoke_styles.py
PYTHONPATH=src python tests/verify_coexist.py
```

Expected: zero failures.

- [ ] **Step 5: Run real v5 smoke synthesis**

Against the downloaded e32 package and a v5-compatible GPT-SoVITS checkout:

```bash
python -m zfh_voice2 --backend torch --model-dir <v5-model-dir> --gsv-root <v5-gsv-root> say "让我想想。" --no-cache
```

Expected: 48 kHz WAV, nonzero duration, selected model reports e32/v5dev.

- [ ] **Step 6: Commit and push**

```bash
git add README.md CHANGELOG.md docs/安装提示词.md package.json pyproject.toml src/zfh_voice2/__init__.py
git commit -m "release: prepare v0.3.0 v5-only plugin"
git push origin main
git push ms HEAD:master
```

---

### Task 7: Distribution Verification and Release Publication

**Files:**
- No code changes unless verification exposes a defect.

- [ ] **Step 1: Verify ModelScope clean repository**

Clone with LFS smudge disabled; assert README, GPT, four SoVITS weights, references, configuration, manifest and checksums exist under the documented paths.

- [ ] **Step 2: Verify GitHub draft Release assets**

Use `gh release view model-v2-v5-fullft --json assets,isDraft`; compare asset sizes/names to `distribution_manifest.json`.

- [ ] **Step 3: Publish GitHub Release**

Only after both repositories and plugin downloader integration pass:

```bash
gh release edit model-v2-v5-fullft --repo yangwenjie1231/dsh-zhuang-fangyi-voice-v2 --draft=false
```

- [ ] **Step 4: Verify public download paths**

Perform HEAD/download checks for ModelScope and GitHub e32, configuration, manifest and resource ZIP. Confirm the plugin downloader succeeds from both sources.
