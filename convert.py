from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

from config import LLAMA_CPP_DIR, GGUF_OUTPUT_DIR, QUANTIZE_LEVEL


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Hugging Face model to GGUF")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--quantize", type=str, default=QUANTIZE_LEVEL)
    
    args = parser.parse_args()
    
    if not args.model:
        raise ValueError("--model is required")
    model_path = Path(args.model)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    if args.output:
        output_file = Path(args.output)
    else:
        output_file = GGUF_OUTPUT_DIR / f"{model_path.name}.gguf"
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    convert_to_gguf(model_path, output_file, args.quantize)


if __name__ == "__main__":
    main()
