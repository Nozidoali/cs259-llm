# Modifications to llama.cpp for Zero Shared Expert Intermediate Size

This document describes the modifications needed in llama.cpp to:
1. Set metadata for `shared_expert_intermediate_size = 0`
2. Skip the corresponding computation for the shared expert (shexp)

## Overview

When `shared_expert_intermediate_size = 0`, the shared expert should be completely skipped during inference to save computation. This requires changes in both the conversion script and the inference code.

## Part 1: Conversion Script (`convert_hf_to_gguf.py`)

### 1.1 Reading and Setting Metadata

In the conversion script, you need to:

1. **Read `shared_expert_intermediate_size` from config.json:**
   ```python
   # In the function that reads model config
   shared_expert_intermediate_size = getattr(hf_config, 'shared_expert_intermediate_size', None)
   
   # If you want to force it to 0, or if it's already 0:
   if shared_expert_intermediate_size == 0 or force_zero_shexp:
       shared_expert_intermediate_size = 0
   ```

2. **Write to GGUF metadata:**
   ```python
   # When writing GGUF metadata
   if shared_expert_intermediate_size is not None:
       # Write as UINT32
       gguf_writer.add_key_value("llama.moe.shared_expert_intermediate_size", 
                                  shared_expert_intermediate_size)
   ```

3. **Skip writing shared expert weights if size is 0:**
   ```python
   # When iterating through model layers
   if shared_expert_intermediate_size == 0:
       # Skip all shared expert weight tensors
       # Look for keys like:
       # - "model.layers.{i}.mlp.shared_expert_gate_proj.weight"
       # - "model.layers.{i}.mlp.shared_expert_up_proj.weight"
       # - "model.layers.{i}.mlp.shared_expert_down_proj.weight"
       continue
   ```

### 1.2 Specific File Locations

The conversion script is typically located at:
- `convert_hf_to_gguf.py` (root of llama.cpp)

Key functions to modify:
- Function that reads HuggingFace config (usually `load_hf_model_config`)
- Function that writes GGUF metadata (usually `write_gguf_metadata`)
- Function that converts and writes tensors (usually `convert_and_write_tensors`)

## Part 2: Inference Code (C++)

### 2.1 Reading Metadata

In the model loading code (typically `llama.cpp` or `gguf.cpp`):

1. **Read the metadata:**
   ```cpp
   // In struct llama_model_params or similar
   uint32_t shared_expert_intermediate_size = 0;
   
   // When loading GGUF file
   shared_expert_intermediate_size = gguf_get_val_u32(ctx, 
       "llama.moe.shared_expert_intermediate_size");
   ```

2. **Store in model structure:**
   ```cpp
   // In struct llama_model or similar MoE model structure
   struct llama_moe_layer {
       // ... existing fields ...
       uint32_t shared_expert_intermediate_size;
       bool use_shared_expert;  // true if size > 0
   };
   ```

### 2.2 Skipping Computation

In the forward pass code (typically in `llama.cpp` or MoE-specific files):

1. **Check before shared expert computation:**
   ```cpp
   // In the MoE layer forward function
   void llama_moe_forward(...) {
       // ... regular expert computation ...
       
       // Shared expert computation
       if (layer->shared_expert_intermediate_size > 0) {
           // Existing shared expert computation
           // ... gate_proj, up_proj, activation, down_proj ...
       } else {
           // Skip shared expert - output zero contribution
           // Or simply don't add shared expert output
       }
   }
   ```

2. **Alternative: Early return or conditional:**
   ```cpp
   // More efficient: check at the beginning
   if (layer->shared_expert_intermediate_size == 0) {
       // Skip all shared expert allocations and computations
       // Only compute regular expert outputs
       return expert_outputs;
   }
   ```

### 2.3 Specific File Locations

Key C++ files to modify:
- `gguf.cpp` or `gguf.h` - for reading metadata
- `llama.cpp` - main inference code
- MoE-specific files (if separate, e.g., `llama-moe.cpp` or similar)

Look for:
- Functions with names like `llama_moe_forward`, `llama_build_moe_layer`
- Structures like `llama_moe_layer`, `llama_model`
- Shared expert computation blocks

## Part 3: Memory Optimization

When `shared_expert_intermediate_size = 0`, you can also:

1. **Skip weight loading:**
   ```cpp
   if (shared_expert_intermediate_size == 0) {
       // Don't allocate memory for shared expert weights
       layer->shared_expert_gate_proj = nullptr;
       layer->shared_expert_up_proj = nullptr;
       layer->shared_expert_down_proj = nullptr;
   }
   ```

2. **Skip intermediate buffers:**
   ```cpp
   // Don't allocate intermediate activation buffers for shared expert
   if (shared_expert_intermediate_size > 0) {
       allocate_shared_expert_buffer(...);
   }
   ```

## Part 4: Testing

After modifications:

1. **Verify metadata is written correctly:**
   ```bash
   # Use gguf-dump or similar tool to check metadata
   ./bin/gguf-dump model.gguf | grep shared_expert
   ```

2. **Test inference:**
   - Load a model with `shared_expert_intermediate_size = 0`
   - Verify no shared expert computation occurs
   - Check that outputs are correct (should match when shared expert is zero-initialized)

3. **Performance check:**
   - Compare inference speed with and without shared expert
   - Should see speedup when shared expert is skipped

## Part 5: Python Side (Optional)

You may also want to modify the Python code to set this to 0:

In `src/rmoe/moemodel.py`, around line 580-583:

**Current code:**
```python
if self.use_zero_shared_expert or self.shared_expert_model is None:
    # Use same intermediate size as regular experts
    shared_expert_intermediate_size = moe_intermediate_size
    base_config['shared_expert_intermediate_size'] = shared_expert_intermediate_size
    logger.info(f"Shared expert intermediate size (zero-initialized): {shared_expert_intermediate_size}")
```

**Modified code (to set to 0):**
```python
if self.use_zero_shared_expert or self.shared_expert_model is None:
    # Set to 0 to skip shared expert computation in llama.cpp
    shared_expert_intermediate_size = 0
    base_config['shared_expert_intermediate_size'] = shared_expert_intermediate_size
    logger.info(f"Shared expert intermediate size (disabled): {shared_expert_intermediate_size}")
```

This ensures the config.json already has `shared_expert_intermediate_size = 0` before conversion, which will be picked up by the llama.cpp conversion script.

**Note:** If you keep the Python code as-is (setting it to `moe_intermediate_size`), you can still force it to 0 in the llama.cpp conversion script by checking the value and overriding it.

## Part 6: Concrete Code Examples

### Example 1: Conversion Script (Python)

```python
# In convert_hf_to_gguf.py, find where config is read
def convert_moe_model(...):
    # Read config
    hf_config = model.config
    
    # Get shared expert intermediate size
    shared_expert_intermediate_size = getattr(
        hf_config, 
        'shared_expert_intermediate_size', 
        None
    )
    
    # Force to 0 if you want to skip it
    # Or check if it's already 0
    if shared_expert_intermediate_size == 0:
        # Write metadata
        gguf_writer.add_key_value(
            "llama.moe.shared_expert_intermediate_size",
            GGUFValueType.UINT32,
            0
        )
        
        # Skip writing shared expert weights
        # In the tensor writing loop:
        for name, tensor in state_dict.items():
            if "shared_expert" in name:
                continue  # Skip shared expert tensors
            # Write other tensors...
```

### Example 2: C++ Model Loading

```cpp
// In gguf.cpp or model loading code
struct llama_moe_layer {
    // ... existing fields ...
    uint32_t shared_expert_intermediate_size;
    bool has_shared_expert;
    
    // Weight pointers (nullptr if not used)
    struct ggml_tensor * shared_expert_gate_proj;
    struct ggml_tensor * shared_expert_up_proj;
    struct ggml_tensor * shared_expert_down_proj;
};

// When loading model
void load_moe_layer(...) {
    // Read metadata
    layer->shared_expert_intermediate_size = 
        gguf_get_val_u32(ctx, "llama.moe.shared_expert_intermediate_size");
    
    layer->has_shared_expert = (layer->shared_expert_intermediate_size > 0);
    
    if (layer->has_shared_expert) {
        // Load shared expert weights
        load_tensor("shared_expert_gate_proj", ...);
        load_tensor("shared_expert_up_proj", ...);
        load_tensor("shared_expert_down_proj", ...);
    } else {
        // Set to nullptr - no weights to load
        layer->shared_expert_gate_proj = nullptr;
        layer->shared_expert_up_proj = nullptr;
        layer->shared_expert_down_proj = nullptr;
    }
}
```

### Example 3: C++ Forward Pass

```cpp
// In the MoE forward function
void llama_moe_forward(
    struct ggml_context * ctx,
    struct ggml_tensor * hidden_states,
    struct llama_moe_layer * layer,
    // ... other params
) {
    // ... regular expert computation ...
    
    // Shared expert computation (only if enabled)
    struct ggml_tensor * shared_expert_out = nullptr;
    
    if (layer->has_shared_expert && layer->shared_expert_intermediate_size > 0) {
        // Compute shared expert
        auto gate = ggml_mul_mat(ctx, layer->shared_expert_gate_proj, hidden_states);
        auto up = ggml_mul_mat(ctx, layer->shared_expert_up_proj, hidden_states);
        // ... activation, down_proj, etc ...
        shared_expert_out = /* computed output */;
    }
    // If has_shared_expert is false, shared_expert_out remains nullptr
    
    // Combine outputs
    if (shared_expert_out != nullptr) {
        // Add shared expert contribution
        output = ggml_add(ctx, expert_output, shared_expert_out);
    } else {
        // Only expert output
        output = expert_output;
    }
    
    return output;
}
```

## Summary

The key changes are:
1. **Conversion**: Read config, write metadata as 0, skip weight writing
2. **Inference**: Read metadata, check if 0, skip computation
3. **Memory**: Don't allocate weights/buffers when size is 0

This will allow llama.cpp to efficiently skip the shared expert when it's not needed.

## Next Steps

1. Locate your llama.cpp repository (check `LLAMA_CPP_DIR` from config or environment)
2. Find the conversion script (`convert_hf_to_gguf.py`)
3. Find the MoE inference code (search for "shared_expert" or "moe" in C++ files)
4. Apply the modifications following the patterns above
5. Test with a model that has `shared_expert_intermediate_size = 0`

