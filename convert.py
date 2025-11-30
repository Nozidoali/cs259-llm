from __future__ import annotations

import importlib
import sys

from config import LLAMA_CPP_DIR, OUTPUT_DIR, GGUF_MODEL_FILE


def convert_to_gguf() -> None:
    sys.path.append(str(LLAMA_CPP_DIR))
    convert_module = importlib.import_module("convert_hf_to_gguf")
    sys.argv = [
        "convert_hf_to_gguf.py",
        str(OUTPUT_DIR),
        "--outfile",
        str(GGUF_MODEL_FILE),
        "--outtype",
        "f16",
    ]
    convert_module.main()


def main() -> None:
    convert_to_gguf()


if __name__ == "__main__":
    main()

