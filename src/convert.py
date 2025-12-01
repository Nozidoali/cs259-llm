from __future__ import annotations

import importlib
import sys
from pathlib import Path

from config import LLAMA_CPP_DIR


def convert_to_gguf(model_path: Path, output_file: Path, quantize_level: str) -> None:
    print(f"Converting: {model_path}")
    print(f"Output: {output_file}")
    print(f"Quantization: {quantize_level}")
    
    sys.path.append(str(LLAMA_CPP_DIR))
    convert_module = importlib.import_module("convert_hf_to_gguf")
    sys.argv = [
        "convert_hf_to_gguf.py",
        str(model_path),
        "--outfile",
        str(output_file),
        "--outtype",
        quantize_level,
    ]
    convert_module.main()
    print(f"\n✓ Complete: {output_file}")
