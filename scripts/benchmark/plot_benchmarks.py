#!/usr/bin/env python3
"""
Script to plot benchmark results from benchmarks.csv
X-axis: Geometric mean of prefill and decode tokens per second
Y-axis: Model size in MB
Reference: Llama 3.2 1B Q8 (dashed lines)
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

# Set font to Lato
plt.rcParams['font.family'] = 'Lato'

# Read the CSV file
df = pd.read_csv('benchmarks.csv')

# Calculate geometric mean of prefill_tokens_per_sec and decode_tokens_per_sec
df['tokens_per_sec_geomean'] = np.sqrt(
    df['prefill_tokens_per_sec'] * df['decode_tokens_per_sec']
)

# Find the reference point: Llama 3.2 1B Q8
reference_row = df[
    (df['model_name'] == 'llama-3.2-1b') & 
    (df['quantization'] == 'Q8_0')
]

if reference_row.empty:
    print("Warning: Reference model (Llama 3.2 1B Q8) not found!")
    ref_x = None
    ref_y = None
else:
    ref_x = reference_row['tokens_per_sec_geomean'].values[0]
    ref_y = reference_row['model_size_mb'].values[0]
    print(f"Reference point (Llama 3.2 1B Q8): x={ref_x:.2f}, y={ref_y:.2f}")

# Create the plot
plt.figure(figsize=(20, 8))

# Get unique model names and assign colors
unique_models = df['model_name'].unique()
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_models)))
model_colors = dict(zip(unique_models, colors))

# Congestion detection algorithm
def detect_congestion(df, distance_threshold=0.08):
    """
    Detect points that are too close together and should skip annotations.
    Returns a set of indices to skip.
    Excludes special models (baseline and Qwen 0.5B) from congestion detection.
    """
    # Normalize coordinates to [0, 1] range for distance calculation
    x_vals = df['tokens_per_sec_geomean'].values
    y_vals = df['model_size_mb'].values
    
    x_norm = (x_vals - x_vals.min()) / (x_vals.max() - x_vals.min())
    y_norm = (y_vals - y_vals.min()) / (y_vals.max() - y_vals.min())
    
    skip_indices = set()
    annotated_indices = set()
    
    # First, add all special models to annotated_indices (they're always shown)
    for i in range(len(df)):
        model_name = df.iloc[i]['model_name']
        quantization = df.iloc[i]['quantization']
        is_baseline = (model_name == 'llama-3.2-1b' and quantization == 'Q8_0')
        is_qwen_0_5b = (model_name == 'qwen2-0.5b')
        if is_baseline or is_qwen_0_5b:
            annotated_indices.add(i)
    
    for i in range(len(df)):
        if i in skip_indices or i in annotated_indices:
            continue
            
        # Check distance to all annotated points
        is_congested = False
        for j in annotated_indices:
            dist = np.sqrt((x_norm[i] - x_norm[j])**2 + (y_norm[i] - y_norm[j])**2)
            if dist < distance_threshold:
                is_congested = True
                break
        
        if is_congested:
            skip_indices.add(i)
        else:
            annotated_indices.add(i)
    
    return skip_indices

# Detect congested points
skip_indices = detect_congestion(df, distance_threshold=0.08)
print(f"Detected {len(skip_indices)} congested points, skipping their annotations")

# Plot all models grouped by model name
plotted_models = set()
for idx, row in df.iterrows():
    model_name = row['model_name']
    quantization = row['quantization']
    param_size = row['param_size']
    label_full = f"{row['model_name']} {row['quantization']}"
    
    # Check if this is Qwen 0.5B
    is_qwen_0_5b = (model_name == 'qwen2-0.5b')
    
    # Only add to legend if this model hasn't been plotted yet
    if model_name not in plotted_models:
        label = f"{model_name} ({param_size})"
        plotted_models.add(model_name)
    else:
        label = None
    
    # Override color for Qwen 0.5B
    point_color = 'red' if is_qwen_0_5b else model_colors[model_name]
    
    # Plot with square markers
    plt.scatter(
        row['tokens_per_sec_geomean'], 
        row['model_size_mb'],
        s=200,  # Larger markers
        alpha=0.7,
        color=point_color,
        marker='s',  # Square markers
        label=label
    )
    
    # Check if this is a special model that should always be annotated
    is_baseline = (model_name == 'llama-3.2-1b' and quantization == 'Q8_0')
    is_qwen_0_5b = (model_name == 'qwen2-0.5b')
    
    # Annotate if not in congested area OR if it's a special model (baseline or Qwen 0.5B)
    if idx not in skip_indices or is_baseline or is_qwen_0_5b:
        # Set annotation color: blue for baseline, red for Qwen 0.5B, black for others
        if is_baseline:
            text_color = 'blue'
            font_weight = 'normal'
            bbox_props = None
            text_offset = (5, 5)
        elif is_qwen_0_5b:
            text_color = 'red'
            font_weight = 'bold'
            bbox_props = dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='red', linewidth=1.5)
            text_offset = (15, 15)  # Upper right positioning
        else:
            text_color = 'black'
            font_weight = 'normal'
            bbox_props = None
            text_offset = (5, 5)
        
        plt.annotate(
            label_full,
            (row['tokens_per_sec_geomean'], row['model_size_mb']),
            xytext=text_offset,
            textcoords='offset points',
            fontsize=18,
            alpha=0.7,
            color=text_color,
            weight=font_weight,
            bbox=bbox_props
        )

# Add reference lines if reference point exists
if ref_x is not None and ref_y is not None:
    plt.axhline(y=ref_y, color='blue', linestyle='--', linewidth=2, 
                label='Llama 3.2 1B Q8 reference', alpha=0.6)
    plt.axvline(x=ref_x, color='blue', linestyle='--', linewidth=2, alpha=0.6)

# Labels (no title)
plt.xlabel('Tokens per Second (Geometric Mean of Prefill & Decode)', fontsize=18)
plt.ylabel('Model Size (MB)', fontsize=18)
plt.grid(True, alpha=0.3)

# Increase tick label sizes
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Add legend outside on the right side, grouped by model name
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), 
           bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=18, framealpha=0.9)

plt.tight_layout()

# Save the plot
output_file = 'benchmark_plot.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\nPlot saved to: {output_file}")

# Also display the plot
plt.show()

# Print summary statistics
print("\n=== Summary Statistics ===")
print(df[['model_name', 'quantization', 'model_size_mb', 
          'prefill_tokens_per_sec', 'decode_tokens_per_sec', 
          'tokens_per_sec_geomean']].to_string(index=False))
