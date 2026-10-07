#!/usr/bin/env python3
"""
Export fine-tuned Qwen2.5-7B LoRA adapter to Ollama.

Steps:
  1. Merge LoRA adapter into base weights  (merge_and_unload)
  2. Save merged model to models/qwen25_merged/
  3. Convert to GGUF (Q4_K_M) via llama.cpp  convert_hf_to_gguf.py
  4. Write Ollama Modelfile
  5. Register with Ollama: ollama create qwen25-legal-ft

Usage:
    python scripts/13_export_lora_to_ollama.py
    python scripts/13_export_lora_to_ollama.py --skip-merge      # merged model already on disk
    python scripts/13_export_lora_to_ollama.py --skip-convert    # GGUF already on disk
    python scripts/13_export_lora_to_ollama.py --dry-run         # print plan, exit
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("PYTHONUTF8", "1")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import config

# ── Paths ──────────────────────────────────────────────────────────────────
ADAPTER_DIR    = config.LORA_ADAPTER_DIR
MERGED_DIR     = config.BASE_DIR / "models" / "qwen25_merged"
GGUF_PATH      = config.BASE_DIR / "models" / "qwen25_merged.gguf"
MODELFILE_PATH = config.BASE_DIR / "models" / "Modelfile"
OLLAMA_NAME    = config.LLM_FINETUNED_MODEL   # "qwen25-legal-ft"
LLAMA_CPP_DIR  = config.BASE_DIR / "tools" / "llama.cpp"
QUANT_TYPE     = "q8_0"


def _base_model_id() -> str:
    training_config_path = ADAPTER_DIR / "training_config.json"
    if training_config_path.exists():
        with open(training_config_path, encoding="utf-8") as f:
            training_config = json.load(f)
        return training_config.get("base_model", "Qwen/Qwen2.5-7B-Instruct")
    return "Qwen/Qwen2.5-7B-Instruct"


def _ollama_cmd() -> str:
    cmd = shutil.which("ollama")
    if cmd:
        return cmd
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        candidate = Path(local_app_data) / "Programs" / "Ollama" / "ollama.exe"
        if candidate.exists():
            return str(candidate)
    sys.exit("ERROR: ollama not found. Install/start Ollama or add it to PATH.")


def _step(n: int, title: str) -> None:
    print(f"\n{'='*66}\n  ADIM {n}: {title}\n{'='*66}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — merge LoRA into base model
# ─────────────────────────────────────────────────────────────────────────────
def merge_adapter(dry_run: bool) -> None:
    _step(1, "LoRA adaptörünü base model ile birleştir")
    hf_base_model = _base_model_id()
    print(f"  Base    : {hf_base_model}")
    print(f"  Adapter : {ADAPTER_DIR}")
    print(f"  Output  : {MERGED_DIR}")

    if dry_run:
        print("  [dry-run] Skipping.")
        return

    if not ADAPTER_DIR.exists():
        sys.exit(f"ERROR: adapter dir not found: {ADAPTER_DIR}")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("\n  Loading base model (bf16, device_map=auto) …")
    base = AutoModelForCausalLM.from_pretrained(
        hf_base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=config.TRUST_REMOTE_CODE,
    )
    tok = AutoTokenizer.from_pretrained(hf_base_model, trust_remote_code=config.TRUST_REMOTE_CODE)

    print("  Loading LoRA adapter …")
    model = PeftModel.from_pretrained(base, str(ADAPTER_DIR))

    print("  Merging weights (merge_and_unload) …")
    model = model.merge_and_unload()

    print(f"  Saving to {MERGED_DIR} …")
    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(MERGED_DIR), safe_serialization=True)
    tok.save_pretrained(str(MERGED_DIR))

    # Free GPU memory immediately
    del model, base
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    print("  ✓ Merge complete.")


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — convert to GGUF
# ─────────────────────────────────────────────────────────────────────────────
def _find_or_clone_llama_cpp() -> Path:
    """Return path to llama.cpp repo root, cloning it if necessary."""
    candidates = [
        LLAMA_CPP_DIR,
        Path.home() / "llama.cpp",
        Path("../llama.cpp"),
    ]
    for p in candidates:
        if (p / "convert_hf_to_gguf.py").exists():
            return p

    print(f"  llama.cpp not found — cloning into {LLAMA_CPP_DIR} …")
    LLAMA_CPP_DIR.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth=1",
         "https://github.com/ggerganov/llama.cpp", str(LLAMA_CPP_DIR)],
        check=True,
    )
    req = LLAMA_CPP_DIR / "requirements.txt"
    if req.exists():
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "-r", str(req)],
            check=True,
        )
    return LLAMA_CPP_DIR


def convert_to_gguf(dry_run: bool) -> None:
    _step(2, f"GGUF dönüşümü ({QUANT_TYPE.upper()})")
    print(f"  Input  : {MERGED_DIR}")
    print(f"  Output : {GGUF_PATH}")

    if dry_run:
        print("  [dry-run] Skipping.")
        return

    if not MERGED_DIR.exists():
        sys.exit(f"ERROR: merged model not found: {MERGED_DIR}\n"
                 "  Run without --skip-merge first.")

    llama_cpp = _find_or_clone_llama_cpp()
    script = llama_cpp / "convert_hf_to_gguf.py"

    cmd = [
        sys.executable, str(script),
        str(MERGED_DIR),
        "--outfile", str(GGUF_PATH),
        "--outtype", QUANT_TYPE,
    ]
    print(f"  Cmd: {' '.join(str(c) for c in cmd)}")
    subprocess.run(cmd, check=True)

    size_gb = GGUF_PATH.stat().st_size / 1e9
    print(f"  ✓ GGUF saved: {GGUF_PATH}  ({size_gb:.2f} GB)")


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — write Modelfile
# ─────────────────────────────────────────────────────────────────────────────
def write_modelfile() -> None:
    _step(3, "Ollama Modelfile oluştur")
    # The evaluation sends num_ctx / stop / num_predict with every request
    # (identical for the base model); the Modelfile only sets the same values
    # as defaults for interactive use.  FROM is relative to the Modelfile so
    # the file carries no machine-specific path.
    stops = "".join(
        f"PARAMETER stop {json.dumps(s, ensure_ascii=False)}\n" for s in config.LLM_STOP
    )
    content = (
        f"FROM ./{GGUF_PATH.name}\n"
        f"PARAMETER temperature {config.LLM_TEMPERATURE}\n"
        f"PARAMETER num_predict {config.LLM_MAX_TOKENS}\n"
        f"PARAMETER num_ctx {config.LLM_NUM_CTX}\n"
        f"PARAMETER num_gpu 999\n"
        + stops
        + 'SYSTEM "Sen Türk hukuku alanında uzman bir hukuki asistansın. '
        'Soruları yalnızca verilen bağlama dayanarak Türkçe yanıtla."\n'
    )
    MODELFILE_PATH.write_text(content, encoding="utf-8")
    print(f"  Written: {MODELFILE_PATH}")
    print(f"\n{content}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — register with Ollama
# ─────────────────────────────────────────────────────────────────────────────
def register_with_ollama(dry_run: bool) -> None:
    _step(4, f"Ollama'ya kaydet → {OLLAMA_NAME}")

    ollama_cmd = _ollama_cmd()

    if dry_run:
        print(f"  [dry-run] Would run: {ollama_cmd} create {OLLAMA_NAME} -f {MODELFILE_PATH}")
        return

    if not GGUF_PATH.exists():
        sys.exit(f"ERROR: GGUF file missing: {GGUF_PATH}")

    if not MODELFILE_PATH.exists():
        write_modelfile()

    subprocess.run(
        [ollama_cmd, "create", OLLAMA_NAME, "-f", str(MODELFILE_PATH)],
        check=True,
    )
    print(f"\n  ✓ Model registered: {OLLAMA_NAME}")
    print(f"  Test: ollama run {OLLAMA_NAME} \"Mülkiyet hakkı nedir?\"")


# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser(
        description="Export fine-tuned LoRA adapter to Ollama (GGUF format)."
    )
    p.add_argument("--skip-merge",   action="store_true",
                   help="Skip merge step — merged model already on disk.")
    p.add_argument("--skip-convert", action="store_true",
                   help="Skip GGUF conversion — GGUF file already on disk.")
    p.add_argument("--dry-run",      action="store_true",
                   help="Print plan without running heavy operations.")
    args = p.parse_args()

    print("\n🚀  LoRA → Ollama Export Pipeline")
    print(f"   Adapter : {ADAPTER_DIR}")
    print(f"   Merged  : {MERGED_DIR}")
    print(f"   GGUF    : {GGUF_PATH}")
    print(f"   Ollama  : {OLLAMA_NAME}")

    if not args.skip_merge:
        merge_adapter(args.dry_run)
    else:
        print("\n  [--skip-merge] Using existing merged model.")

    if not args.skip_convert:
        convert_to_gguf(args.dry_run)
    else:
        print("\n  [--skip-convert] Using existing GGUF file.")

    write_modelfile()
    register_with_ollama(args.dry_run)

    print("\n✅  Export complete!")
    print(f"   Model      : {OLLAMA_NAME}")
    print(f"   Config key : LLM_FINETUNED_MODEL = \"{OLLAMA_NAME}\"")
    print(f"   Next step  : python scripts/14_eval_all_stages.py --stages llm_ft")


if __name__ == "__main__":
    main()
