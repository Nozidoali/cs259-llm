# Expert Reuse Optimization

## Problem
When training a MoE model with duplicate datasets in the list, the original code would:
1. Load the same dataset 32 times
2. Potentially retrain the same expert 32 times (if cache miss)
3. Waste time and GPU memory

**Example:** 
```json
"datasets": [
  "truthfulqa", "truthfulqa", "truthfulqa", ... // 32 times
]
```

Would load truthfulqa dataset 32 times and check for expert existence 32 times.

## Solution: Smart Expert Reuse

### What Changed
The pipeline now:
1. **Detects unique datasets** - Extracts unique dataset names from the list
2. **Trains once** - Trains only ONE expert per unique dataset
3. **Reuses paths** - Maps all 32 expert slots to the same trained model

### Code Flow

```python
# Original datasets list (with duplicates)
datasets = ["truthfulqa", "truthfulqa", ..., "truthfulqa"]  # 32x

# Extract unique datasets
unique_datasets = ["truthfulqa"]  # Only 1 unique

# Train only unique experts
for dataset_name in unique_datasets:  # Loops only ONCE
    train_expert(dataset_name)  # Train truthfulqa expert
    
# Map all slots to the trained expert
expert_paths = [expert_truthfulqa_path] * 32  # All 32 slots point to same expert
```

## Benefits

| Metric          | Before         | After         | Improvement   |
| --------------- | -------------- | ------------- | ------------- |
| Experts trained | 32             | 1             | 32x faster!   |
| Dataset loads   | 32             | 1             | 32x less I/O  |
| Disk space      | 32x model size | 1x model size | 32x less disk |
| Training time   | ~32 hours      | ~1 hour       | 32x faster!   |

## Example Output

### Scenario: 32 truthfulqa experts

```bash
INFO - Total experts requested: 32
INFO - Unique datasets: 1 - ['truthfulqa']
INFO - Optimization: Training only 1 unique expert(s), will reuse for 32 slots
INFO - ================================================================
INFO - Training expert for dataset: truthfulqa
INFO - Output directory: workspace/.../experts/truthfulqa
[Training proceeds for ONE expert]
INFO - Expert paths created: 32 experts (reusing 1 unique model(s))
```

### Scenario: 16 truthfulqa + 16 qmsum

```bash
INFO - Total experts requested: 32
INFO - Unique datasets: 2 - ['truthfulqa', 'qmsum']
INFO - Optimization: Training only 2 unique expert(s), will reuse for 32 slots
INFO - ================================================================
INFO - Training expert for dataset: truthfulqa
[Training proceeds]
INFO - ================================================================
INFO - Training expert for dataset: qmsum
[Training proceeds]
INFO - Expert paths created: 32 experts (reusing 2 unique model(s))
```

## How It Works

### 1. Unique Dataset Extraction
```python
unique_datasets = list(dict.fromkeys(datasets))
```
- Uses dict to remove duplicates while preserving order
- If `datasets = ["a", "b", "a", "c", "b"]`
- Then `unique_datasets = ["a", "b", "c"]`

### 2. Train Unique Experts
```python
trained_experts = {}
for dataset_name in unique_datasets:
    expert_path = train_expert(dataset_name)
    trained_experts[dataset_name] = expert_path
```
- Trains only unique datasets
- Stores mapping: dataset_name → expert_path

### 3. Create Expert Paths List
```python
expert_paths = [trained_experts[dataset_name] for dataset_name in datasets]
```
- Maps each dataset in original list to its trained expert
- If all 32 are "truthfulqa", all 32 paths point to same expert

## Configuration Examples

### Example 1: Homogeneous Experts (32 identical)
```json
{
  "datasets": [
    "truthfulqa", "truthfulqa", ..., "truthfulqa"  // 32x
  ]
}
```
**Result:** Trains 1 expert, reuses for all 32 slots

### Example 2: Heterogeneous Experts (16 + 16)
```json
{
  "datasets": [
    "truthfulqa", "truthfulqa", ..., "truthfulqa",  // 16x
    "qmsum", "qmsum", ..., "qmsum"                  // 16x
  ]
}
```
**Result:** Trains 2 experts, reuses each 16 times

### Example 3: Mixed Experts
```json
{
  "datasets": [
    "truthfulqa", "qmsum", "truthfulqa", "qmsum", ...
  ]
}
```
**Result:** Trains 2 experts, alternates them in MoE slots

## Why This Works for MoE

In a Mixture of Experts model:
- Multiple expert slots can share the same model weights
- The **gating network** learns to route inputs to appropriate experts
- Even with identical expert models, the gating network provides diversity
- This is useful for:
  - **Ensemble effects**: Multiple routes to same expert
  - **Load balancing**: Distribute compute across expert slots
  - **Redundancy**: Backup routes if primary expert overloads

## Disk Usage

### Before Optimization:
```
experts/
  truthfulqa/         # 1.0 GB
  truthfulqa_1/       # 1.0 GB (duplicate)
  truthfulqa_2/       # 1.0 GB (duplicate)
  ...
  truthfulqa_31/      # 1.0 GB (duplicate)
Total: 32 GB
```

### After Optimization:
```
experts/
  truthfulqa/         # 1.0 GB
Total: 1.0 GB
```

All 32 MoE slots point to the same `experts/truthfulqa/` directory.

## Cache Behavior

The optimization works with the existing cache mechanism:
1. First run: Trains unique experts
2. Subsequent runs: Finds existing experts, skips training
3. All runs: Creates expert_paths list pointing to cached models

## Performance Impact

For 32 truthfulqa experts with 6 epochs each:

| Phase           | Time Before      | Time After      | Savings        |
| --------------- | ---------------- | --------------- | -------------- |
| Expert Training | ~32 hours        | ~1 hour         | 31 hours       |
| Gating Training | ~10 min          | ~10 min         | No change      |
| Merging         | ~5 min           | ~5 min          | No change      |
| **Total**       | **~32.25 hours** | **~1.25 hours** | **~31 hours!** |

## Notes

- **This is not a bug**: Having duplicate datasets is a valid MoE configuration
- **Gating still learns**: The gating network will learn different routing patterns
- **Flexibility**: Can still mix unique and duplicate experts in any configuration
- **No config changes needed**: Works automatically based on dataset list

## Related Files

- `train.py` - Expert reuse logic
- `src/rmoe/merge.py` - MoE merging (uses expert_paths list)
- `src/rmoe/gating.py` - Gating network (works with any expert configuration)
