#!/usr/bin/env python3
"""
Prepare comprehensive text data for llama-imatrix from multiple datasets.
Downloads TruthfulQA (817 questions) and combines with 50 qmsum prompts.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List

def collect_qmsum_texts(prompt_files_dir: Path, max_samples: int = 50) -> List[str]:
    """Collect qmsum prompt texts from prompt_files directory."""
    texts = []
    
    if not prompt_files_dir.exists():
        print(f"Warning: prompt_files directory not found: {prompt_files_dir}")
        return texts
    
    # Find all .prompt.txt files in prompt_files
    txt_files = sorted(prompt_files_dir.glob("qmsum_test_*.prompt.txt"))
    
    print(f"Found {len(txt_files)} qmsum prompt files")
    
    # Limit to max_samples
    txt_files = txt_files[:max_samples]
    print(f"Using {len(txt_files)} qmsum samples")
    
    for txt_file in txt_files:
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:  # Only add non-empty files
                    texts.append(content)
                    print(f"  ✓ {txt_file.name}: {len(content)} chars")
        except Exception as e:
            print(f"  ✗ Error reading {txt_file.name}: {e}")
    
    return texts


def download_truthfulqa() -> List[str]:
    """Download TruthfulQA dataset from HuggingFace (817 questions)."""
    texts = []
    
    try:
        print("Downloading TruthfulQA dataset from HuggingFace...")
        from datasets import load_dataset
        
        # Load the TruthfulQA dataset
        dataset = load_dataset("truthful_qa", "generation")
        
        # Use the validation split which has 817 questions
        validation_data = dataset['validation']
        
        print(f"  ✓ Downloaded {len(validation_data)} questions")
        
        for item in validation_data:
            text_parts = []
            
            # Extract question
            if 'question' in item:
                text_parts.append(f"Question: {item['question']}")
            
            # Extract best answer
            if 'best_answer' in item:
                text_parts.append(f"Answer: {item['best_answer']}")
            elif 'correct_answers' in item and item['correct_answers']:
                # Use first correct answer if available
                text_parts.append(f"Answer: {item['correct_answers'][0]}")
            
            if text_parts:
                texts.append("\n".join(text_parts))
        
        print(f"  ✓ Extracted {len(texts)} question-answer pairs")
        
    except ImportError:
        print("  ✗ Error: 'datasets' library not installed")
        print("  Install with: pip install datasets")
    except Exception as e:
        print(f"  ✗ Error downloading TruthfulQA: {e}")
    
    return texts


def create_imatrix_input(output_file: Path, texts: List[str], max_chars: int = 500000):
    """Create input file for llama-imatrix."""
    print(f"\n📝 Creating imatrix input file: {output_file}")
    
    if not texts:
        print("Error: No text collected!")
        return False
    
    # Combine all texts with separators
    combined = []
    total_chars = 0
    
    for i, text in enumerate(texts):
        if total_chars + len(text) > max_chars:
            print(f"  Reached max character limit ({max_chars:,})")
            break
        
        combined.append(text)
        total_chars += len(text)
    
    # Join with double newlines
    final_text = "\n\n".join(combined)
    
    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(final_text)
    
    print(f"  ✓ Wrote {len(combined)} text samples")
    print(f"  ✓ Total characters: {len(final_text):,}")
    print(f"  ✓ Total words (approx): {len(final_text.split()):,}")
    print(f"  ✓ Total tokens (approx): {len(final_text) // 4:,}")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Prepare imatrix input data from TruthfulQA (817 questions) and 50 qmsum prompts"
    )
    parser.add_argument(
        "--prompt-files-dir",
        type=Path,
        default=Path("prompt_files"),
        help="Directory containing qmsum prompt files"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("imatrix_input.txt"),
        help="Output file for imatrix input"
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=2000000,
        help="Maximum characters to include (default: 2M)"
    )
    parser.add_argument(
        "--qmsum-samples",
        type=int,
        default=50,
        help="Number of qmsum samples to include (default: 50)"
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("Preparing imatrix input data")
    print("="*70)
    
    all_texts = []
    
    # Download and collect TruthfulQA
    print("\n📁 Downloading TruthfulQA dataset (817 questions)...")
    truthfulqa_texts = download_truthfulqa()
    all_texts.extend(truthfulqa_texts)
    print(f"  Total: {len(truthfulqa_texts)} samples")
    
    # Collect from qmsum prompt files
    print(f"\n📁 Collecting {args.qmsum_samples} qmsum prompts...")
    qmsum_texts = collect_qmsum_texts(args.prompt_files_dir, args.qmsum_samples)
    all_texts.extend(qmsum_texts)
    print(f"  Total: {len(qmsum_texts)} samples")
    
    # Create output file
    if not all_texts:
        print("\n❌ No text data collected! Check your data directories.")
        return 1
    
    print(f"\n📊 Total samples collected: {len(all_texts)}")
    print(f"  - TruthfulQA: {len(truthfulqa_texts)} questions")
    print(f"  - QMSum: {len(qmsum_texts)} prompts")
    
    success = create_imatrix_input(args.output, all_texts, args.max_chars)
    
    if not success:
        return 1
    
    print("\n" + "="*70)
    print("✅ imatrix input file ready!")
    print("="*70)
    print(f"\nNext steps:")
    print(f"  1. Run llama-imatrix:")
    print(f"     llama-imatrix -m ./rmoe_model_moe_f16.gguf \\")
    print(f"                   -f {args.output} \\")
    print(f"                   -o rmoe_comprehensive.imatrix")
    print(f"\n  2. Use the imatrix for better quantization:")
    print(f"     llama-quantize --imatrix rmoe_comprehensive.imatrix \\")
    print(f"                    ./rmoe_model_moe_f16.gguf \\")
    print(f"                    ./rmoe_model_moe_Q4_K_M.gguf Q4_K_M")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
