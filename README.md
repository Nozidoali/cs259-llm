# CS259 - GPT-2 Fine-tuning on TruthfulQA

Fine-tune GPT-2 Small on TruthfulQA dataset and convert to GGUF format for edge deployment.

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure paths:**
   Edit `config.py` to adjust paths and training parameters.

## Usage

### Fine-tune GPT-2 on TruthfulQA

```bash
python main.py
```

This will:
- Download GPT-2 Small from Hugging Face
- Load TruthfulQA dataset
- Fine-tune the model
- Save to `models/gpt2-truthfulqa-finetuned/`

### Convert to GGUF

After fine-tuning, convert to GGUF format:

```bash
# From llama.cpp directory
python3 convert_hf_to_gguf.py /path/to/cs259/models/gpt2-truthfulqa-finetuned \
    --outfile /path/to/cs259/models/gguf/gpt2-truthfulqa.gguf \
    --outtype f16
```

### Quantize

```bash
# From llama.cpp build directory
./bin/llama-quantize \
    /path/to/cs259/models/gguf/gpt2-truthfulqa.gguf \
    /path/to/cs259/models/gguf/gpt2-truthfulqa-Q4_0.gguf \
    q4_0
```

### Push to Device

```bash
adb push models/gguf/gpt2-truthfulqa-Q4_0.gguf /data/local/tmp/gguf/
```

## Configuration

Edit `config.py` to adjust:
- Model paths
- Training hyperparameters (epochs, batch size, learning rate)
- Data paths

## Directory Structure

```
cs259/
├── config.py                          # Configuration file
├── main.py                           # Main fine-tuning script
├── requirements.txt                    # Python dependencies
├── README.md                          # This file
├── .gitignore                         # Git ignore rules
├── models/                            # Model outputs (gitignored)
│   ├── gpt2-truthfulqa-finetuned/    # Fine-tuned model
│   └── gguf/                          # GGUF models
├── data/                              # Data cache (gitignored)
└── logs/                              # Training logs (gitignored)
```

## Training Configuration

Default settings in `config.py`:
- **Epochs**: 3
- **Batch size**: 4
- **Learning rate**: 5e-5
- **Max length**: 512 tokens
- **Eval split**: 20%

Adjust these in `config.py` based on your hardware and requirements.

