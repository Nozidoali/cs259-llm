# MoE Model with Gating Network

This directory contains the implementation of a Mixture of Experts (MoE) model that combines two finetuned Llama-3.2-1B-Instruct models using a gating network. The gating network routes between the FFN layers of the two expert models based on input characteristics.

## Overview

The system consists of:
1. **Gating Network**: A neural network that outputs probabilities for routing between two expert models
2. **MoE Model**: Combines two finetuned models, using the gating network to route FFN layers
3. **Inference Interface**: Easy-to-use interface for running inference with the combined model

## Training the Gating Network

The gating network is trained to classify inputs and output probabilities for each expert model.

### Basic Usage

```bash
cd cs259-llm/src/gating
python train_gating.py --base_model meta-llama/Llama-3.2-1B-Instruct --output_dir models/gating-network/my_gating_model
```

### Command-Line Arguments

- `--base_model`: Base model for extracting embeddings (default: from config)
- `--output_dir`: Output directory for the trained gating network
- `--config`: Path to JSON config file (optional)
- `--hidden_dims`: Hidden layer dimensions, comma-separated (e.g., "512,256")
- `--dropout`: Dropout rate (default: 0.1)
- `--learning_rate`: Learning rate (default: 1e-4)
- `--batch_size`: Batch size (default: 32)
- `--num_epochs`: Number of training epochs (default: 10)
- `--weight_decay`: Weight decay (default: 0.01)
- `--train_split`, `--val_split`, `--test_split`: Dataset splits (must sum to 1.0)
- `--seed`: Random seed (default: 42)

### Example

```bash
python train_gating.py \
    --base_model qwen2-0.5b \
    --output_dir models/gating-network/qwen2-0.5b \
    --hidden_dims "512,256" \
    --dropout 0.1 \
    --learning_rate 1e-4 \
    --batch_size 32 \
    --num_epochs 10
```

### Output

The training script saves:
- `best_model.pt`: Best model based on validation F1 score
- `final_model.pt`: Final model after all epochs
- `training_info.json`: Training configuration and metrics

## Running MoE Inference

After training the gating network and having two finetuned models, you can run inference with the combined MoE model.

### Option 1: Using Saved Model (Recommended)

If you have already saved a MoE model using `moe_model.py`, you can load it directly:

```bash
cd cs259-llm/src
python moe_inference.py \
    --saved_model_path models/moe-merged \
    --prompt "Your input prompt here"
```

### Option 2: Loading from Components

You can also load from individual components:

```bash
cd cs259-llm/src
python moe_inference.py \
    --model1_path path/to/model1 \
    --model2_path path/to/model2 \
    --gating_model_path path/to/gating_network \
    --prompt "Your input prompt here"
```

### Command-Line Arguments

- `--saved_model_path`: Path to saved MoE model directory (if provided, loads from saved model)
- `--model1_path`: Path to first finetuned model (required if saved_model_path not provided)
- `--model2_path`: Path to second finetuned model (required if saved_model_path not provided)
- `--gating_model_path`: Path to trained gating network (required if saved_model_path not provided)
- `--base_model_path`: Path to base model for embeddings (optional, uses training info if not provided)
- `--routing_mode`: Routing strategy - `weighted_sum` or `select_one` (default: `weighted_sum`)
- `--prompt`: Input prompt for generation (required)
- `--max_new_tokens`: Maximum tokens to generate (default: 50)
- `--temperature`: Sampling temperature (default: 0.7)
- `--top_p`: Nucleus sampling parameter (default: 0.9)
- `--no_sample`: Use greedy decoding instead of sampling
- `--config`: Path to JSON config file (optional)

### Examples

**Using saved model:**
```bash
python moe_inference.py \
    --saved_model_path models/moe-merged \
    --prompt "Question: What is the capital of France?" \
    --max_new_tokens 100 \
    --temperature 0.7
```

**Loading from components:**
```bash
python moe_inference.py \
    --model1_path models/llama-3.2-1b-truthfulqa-ffn-finetuned \
    --model2_path models/llama-3.2-1b-qmsum-ffn-finetuned \
    --gating_model_path models/gating-network/qwen2-0.5b \
    --routing_mode weighted_sum \
    --prompt "Question: What is the capital of France?" \
    --max_new_tokens 100 \
    --temperature 0.7
```

### Output

The inference script outputs:
- Generated text
- Gating probabilities for each expert model:
  - `model1_prob`: Probability for model 1 (class 0)
  - `model2_prob`: Probability for model 2 (class 1)

### Using Configuration File

You can also use a JSON config file:

```json
{
    "model1_path": "models/llama-3.2-1b-truthfulqa-ffn-finetuned",
    "model2_path": "models/llama-3.2-1b-qmsum-ffn-finetuned",
    "gating_model_path": "models/gating-network/qwen2-0.5b",
    "routing_mode": "weighted_sum"
}
```

Then run:
```bash
python moe_inference.py --config config.json --prompt "Your prompt"
```

## Routing Modes

- **weighted_sum**: Combines FFN outputs from both models using gating probabilities as weights
- **select_one**: Selects the FFN output from the model with the highest gating probability

## Python API

You can also use the MoE model programmatically:

**Using saved model:**
```python
from moe_inference import MoEInference

# Load from saved model
inference = MoEInference(
    saved_model_path="models/moe-merged"
)
```

**Loading from components:**
```python
from moe_inference import MoEInference

# Initialize from components
inference = MoEInference(
    model1_path="path/to/model1",
    model2_path="path/to/model2",
    gating_model_path="path/to/gating_network",
    routing_mode="weighted_sum"
)
```

**Or directly using MoEModel:**
```python
from moe_model import MoEModel

# Load from saved model
model = MoEModel.from_pretrained("models/moe-merged")
```

# Generate text
result = inference.generate(
    prompt="Your prompt here",
    max_new_tokens=100,
    temperature=0.7
)

print(result["text"])
print(f"Model 1 prob: {result['gating_probs']['model1_prob']:.4f}")
print(f"Model 2 prob: {result['gating_probs']['model2_prob']:.4f}")

# Get gating probabilities only
probs = inference.get_gating_probs("Your text here")
print(probs)
```

## Requirements

- Two finetuned models (finetuned on different datasets, FFN layers only)
- Trained gating network
- Base model for embeddings (used by gating network)

## Notes

- Both expert models must have the same architecture (same number of layers, etc.)
- The gating network outputs 2 probabilities that sum to 1.0 (softmax)
- Model paths can be configured via command-line arguments or config files
- The system uses the first model as the base for embeddings and attention layers, routing only the FFN layers

