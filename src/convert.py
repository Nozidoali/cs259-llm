from __future__ import annotations

import importlib
import sys
from pathlib import Path

from config import LLAMA_CPP_DIR


def convert_to_gguf(model_path: Path, output_file: Path, quantize_level: str = "f16") -> None:
    """
    Convert a HuggingFace model to GGUF format.
    
    Args:
        model_path: Path to the model directory
        output_file: Path to the output GGUF file
        quantize_level: Quantization level (f16, f32, bf16, q8_0, etc.)
                       Defaults to f16 for MoE models compatibility
    """
    # Ensure paths are absolute
    model_path = Path(model_path).resolve()
    output_file = Path(output_file).resolve()
    
    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Map Q4_0 and similar to supported formats (convert_hf_to_gguf doesn't support Q4_0 directly)
    quantize_map = {
        "Q4_0": "f16",  # Map to f16 as default for unsupported quantization
        "Q8_0": "q8_0",
    }
    quantize_level = quantize_map.get(quantize_level, quantize_level)
    
    print(f"Converting: {model_path}")
    print(f"Output: {output_file}")
    print(f"Quantization: {quantize_level}")
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    
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
