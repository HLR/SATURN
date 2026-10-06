<div align="center">

# SATURN: Symbolic Spatial Reasoning for Multi-Perspective Grounding

**EMNLP 2026**

Danial Kamali, Tanawan Premsri, Shreya Rajpal, Amir Zadeh, Chuan Li, Parisa Kordjamshidi

[![arXiv](https://img.shields.io/badge/arXiv-2606.22694-b31b1b.svg)](https://arxiv.org/abs/2606.22694)
![EMNLP 2026](https://img.shields.io/badge/EMNLP-2026-1f6feb.svg)
[![3D-FORCE](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-3D--FORCE-ffd21e.svg)](https://huggingface.co/datasets/iamdanialkamali/3D-FORCE-Zip)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

Official implementation of **SATURN** (EMNLP 2026) and the **3D-FORCE** benchmark (REF: 2,088 referring-expression items; SAG: 1,150 boolean items; [data on Hugging Face](https://huggingface.co/datasets/iamdanialkamali/3D-FORCE-Zip)).

## Overview

SATURN turns a set of images into an estimated 3D scene (SAM3 detection → VGGT reconstruction → Orient-Anything orientation) and exposes it through a small symbolic engine: **frames** (stand somewhere, face some way: `scene.frame(...)`, `.rotate(yaw=...)`, `.look_at(...)`) and **soft predicates** read from a frame (`first_person.left`, `facing.north`, `turned.right`, `third_person.behind`, `closeness`, and `score("is the object a yellow SUV?")` from the VLM). A code LLM writes one short program per question: **one formula** over the question's entities, and **one decision**: the option whose claim, joined with the formula, is most true (`max(options, key=lambda k: float((J & claim_k).exists()))`). A soft-logic engine (`&` = min, `iota`, `exists`, `assign`) executes it. The interface (a fixed engine, soft scores, a composed program) is the contribution; perception models are swappable.

## Layout

Everything importable lives in the `saturn` package; the layers import downward only (scene / predicates / soft_logic never import perception, vlm or serving).

| path | what |
|---|---|
| `saturn/cli/` | entry point (`python -m saturn.cli`) and its arguments |
| `saturn/pipeline/` | the per-question pipeline (plan → ground → build scene → generate → execute → evaluate), resume/merge of results |
| `saturn/soft_logic/` | the soft-logic engine (`ProbabilisticTensor`: `&`, `iota`, `exists`, `assign` over object variables) |
| `saturn/scene/` | scene construction and the predicate API: `fusion.py` (multi-view object fusion), `pose_solver.py`, `scene.py`; `scene/build/` loaders + cache |
| `saturn/predicates/` | frames and the anchor-conditioned soft predicates (`frame.py`, `relations.py`, metrics and scoring) |
| `saturn/planning/` | the query planner and the camera-constraint extractor |
| `saturn/codegen/` | code LLM wrapper + program cache |
| `saturn/perception/` | perception backbones: SAM3 detection, VGGT reconstruction, Orient-Anything orientation |
| `saturn/vlm/` | the VLM client used by `score(...)` and for grounding |
| `saturn/datasets/` | loaders: `force3d.py`, `mindcube.py`, `mmsi.py` |
| `saturn/serving/` | Ray Serve deployments (`python -m saturn.serving.deploy`) |
| `saturn/reports/` | per-sample HTML debug reports (`python -m saturn.reports.debug_report`) |
| `prompts/` | the codegen prompts: `vqa.txt` (MindCube and MMSI), `force3d_{ref,sag}.txt` (3D-FORCE), `constraint_extractor.txt` |
| `configs/` | `benchmarks.json`: the per-benchmark setup (codegen prompt, pose constraints); `settings.json`: every environment variable SATURN reads, with its default and purpose |
| `benchmark/3d-force/` | the 3D-FORCE benchmark card (tasks, item schema, data layout); the data is on [Hugging Face](https://huggingface.co/datasets/iamdanialkamali/3D-FORCE-Zip) |
| `scripts/` | `serve.sh`, `serve_vlm.sh`, `run_mindcube.sh`, `run_mmsi.sh`, `run_force3d.sh` (thin wrappers around `python -m saturn.cli`), `export_programs.py`, `release_reproduce.sh`, `release_score.py` |
| `tests/` | unit tests for the engine, predicates, fusion, planner and pipeline (`python -m pytest tests/`) |

## Install

Linux, Python 3.10, CUDA 12.4, ≥3 GPUs (2 for perception services, ≥1 for the VLM).

```bash
git clone https://github.com/HLR/SATURN.git && cd SATURN
bash setup/00_python_env.sh        # .venv (SATURN) + .vllm (VLM server)
bash setup/10_clone_vendored.sh    # SAM3, Orient-Anything-V2 (+VGGT) at pinned SHAs
bash setup/20_download_models.sh   # weights (SAM3 is gated: accept its license on HF first)
cp .env.example .env               # add DEEPSEEK_API_KEY
bash setup/30_datasets.sh          # 3D-FORCE, MindCube, MMSI-Bench (Hugging Face) -> data/
pip install -e .                   # the `saturn` package (pyproject.toml)
```

## Run

```bash
GPUS=0,1 scripts/serve.sh          # perception services on Ray Serve (port 8011)
GPUS=2   scripts/serve_vlm.sh      # Qwen3-VL-8B-Instruct via vLLM (port 8100)

scripts/run_mindcube.sh among 0    # MindCube slice (among | around | rotation), seed 0 | 1
scripts/run_mmsi.sh 0              # MMSI-Bench, all 1000 questions, seed 0 | 1
scripts/run_force3d.sh ref 0       # 3D-FORCE REF (ref) or SAG (puzzle), seed 0 | 1
```

Each script plans every question and writes its program through the code LLM (`DEEPSEEK_API_KEY`); the seed is the code LLM's. `scripts/release_reproduce.sh` runs every benchmark with seeds 0 and 1; `scripts/release_score.py` prints each run's accuracy and the mean over seeds.

Each script is a thin wrapper around `python -m saturn.cli …` (run it with `--help` for every flag). Results land in `experiments/<dataset>/<vlm>/<name>.json`, one record per question (`id`, `correct_final_answer`, `program_code`, timings); `python -m saturn.reports.debug_report --result_json <results.json>` renders per-sample HTML reports.

## Citation

```bibtex
@misc{kamali2026saturnsymbolicspatialreasoning,
      title={SATURN: Symbolic Spatial Reasoning for Multi-Perspective Grounding},
      author={Danial Kamali and Tanawan Premsri and Shreya Rajpal and Amir Zadeh and Chuan Li and Parisa Kordjamshidi},
      year={2026},
      eprint={2606.22694},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2606.22694},
}
```

## License

MIT (see `LICENSE`). SAM3, VGGT, Orient-Anything and the VLM/LLM weights are fetched from their upstream sources under their own licenses; they are not bundled.
