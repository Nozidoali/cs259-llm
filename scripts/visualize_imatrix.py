#!/usr/bin/env python3
"""
Visualize importance matrix (imatrix) from GGUF files.
Shows which weights are most important for quantization.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Add llama.cpp gguf-py to path
gguf_py_path = Path(__file__).parent.parent / "external" / "llama.cpp" / "gguf-py"
sys.path.insert(0, str(gguf_py_path))

try:
    from gguf import GGUFReader
except ImportError:
    print("Error: Could not import gguf module from llama.cpp")
    print(f"Tried to import from: {gguf_py_path}")
    print("Please ensure llama.cpp submodule is initialized:")
    print("  git submodule update --init --recursive")
    sys.exit(1)


def parse_imatrix(imatrix_file: Path):
    """Parse GGUF imatrix file and extract importance values."""
    print(f"Reading imatrix from: {imatrix_file}")
    
    reader = GGUFReader(str(imatrix_file))
    
    # Extract metadata
    metadata = {}
    for field in reader.fields.values():
        metadata[field.name] = field.parts[field.data[0]]
    
    # Extract importance data
    importance_data = {}
    
    for tensor in reader.tensors:
        name = tensor.name
        data = tensor.data
        
        # Parse tensor names to extract layer and component info
        if ".in_sum2" in name or ".counts" in name:
            base_name = name.replace(".in_sum2", "").replace(".counts", "")
            if base_name not in importance_data:
                importance_data[base_name] = {}
            
            if ".in_sum2" in name:
                importance_data[base_name]['sum2'] = data
            elif ".counts" in name:
                importance_data[base_name]['counts'] = data
    
    # Calculate importance scores (average squared value)
    importance_scores = {}
    for name, data in importance_data.items():
        if 'sum2' in data and 'counts' in data:
            sum2 = np.array(data['sum2'], dtype=np.float32)
            counts = np.array(data['counts'], dtype=np.float32)
            
            # Avoid division by zero
            counts = np.maximum(counts, 1.0)
            
            # Mean squared value
            importance = np.mean(sum2 / counts)
            importance_scores[name] = importance
    
    return importance_scores, metadata


def categorize_weights(importance_scores):
    """Categorize weights by layer and component type."""
    categories = defaultdict(lambda: defaultdict(list))
    
    for name, score in importance_scores.items():
        # Extract layer number
        if "blk." in name:
            parts = name.split(".")
            try:
                # Format: blk.N.component.weight
                if parts[0] == "blk" and len(parts) >= 2:
                    layer_num = int(parts[1])
                else:
                    continue
            except ValueError:
                continue  # Skip if can't parse layer number
            component = ".".join(parts[2:])
            
            # Categorize by component type
            if "attn" in component:
                if "attn_q" in component:
                    cat = "attention_q"
                elif "attn_k" in component:
                    cat = "attention_k"
                elif "attn_v" in component:
                    cat = "attention_v"
                elif "attn_output" in component:
                    cat = "attention_output"
                else:
                    cat = "attention_other"
            elif "ffn" in component:
                if "ffn_gate" in component:
                    cat = "ffn_gate"
                elif "ffn_up" in component:
                    cat = "ffn_up"
                elif "ffn_down" in component:
                    cat = "ffn_down"
                else:
                    cat = "ffn_other"
            else:
                cat = "other"
            
            categories[cat][layer_num].append(score)
    
    return categories


def plot_layer_importance(importance_scores, output_dir: Path):
    """Plot importance by layer."""
    # Group by layer
    layer_scores = defaultdict(list)
    
    for name, score in importance_scores.items():
        if "blk." in name:
            try:
                parts = name.split(".")
                # Format: blk.N.component.weight
                if parts[0] == "blk" and len(parts) >= 2:
                    layer_num = int(parts[1])
                    layer_scores[layer_num].append(score)
            except ValueError:
                continue  # Skip if can't parse layer number
    
    if not layer_scores:
        print("  ⚠️  No layer-specific data found, skipping layer importance plot")
        return
    
    # Calculate average importance per layer
    layers = sorted(layer_scores.keys())
    avg_importance = [np.mean(layer_scores[l]) for l in layers]
    max_importance = [np.max(layer_scores[l]) for l in layers]
    min_importance = [np.min(layer_scores[l]) for l in layers]
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Average importance by layer
    ax1.plot(layers, avg_importance, 'b-o', linewidth=2, markersize=6, label='Average')
    ax1.fill_between(layers, min_importance, max_importance, alpha=0.3, label='Min-Max Range')
    ax1.set_xlabel('Layer Number', fontsize=12)
    ax1.set_ylabel('Importance Score', fontsize=12)
    ax1.set_title('Weight Importance by Layer', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Heatmap of all weights
    all_layers = []
    for layer in layers:
        all_layers.append(layer_scores[layer])
    
    # Normalize for visualization
    max_len = max(len(scores) for scores in all_layers) if all_layers else 0
    if max_len > 0:
        heatmap_data = []
        for scores in all_layers:
            padded = list(scores) + [0] * (max_len - len(scores))
            heatmap_data.append(padded[:50])  # Limit to 50 for visibility
        
    sns.heatmap(heatmap_data, cmap='YlOrRd', ax=ax2, cbar_kws={'label': 'Importance Score'})
    ax2.set_xlabel('Weight Index (within layer)', fontsize=12)
    ax2.set_ylabel('Layer Number', fontsize=12)
    ax2.set_title('Weight Importance Heatmap', fontsize=14, fontweight='bold')
    # Only set labels if the number matches
    if len(ax2.get_yticklabels()) == len(layers):
        ax2.set_yticklabels(layers, rotation=0)
    
    plt.tight_layout()
    output_file = output_dir / "layer_importance.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file}")
    plt.close()


def plot_component_importance(categories, output_dir: Path):
    """Plot importance by component type."""
    if not categories:
        print("  ⚠️  No component-specific data found, skipping component importance plot")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Calculate average importance per component type
    component_avg = {}
    for comp_type, layers in categories.items():
        all_scores = []
        for scores in layers.values():
            all_scores.extend(scores)
        component_avg[comp_type] = np.mean(all_scores) if all_scores else 0
    
    if not component_avg:
        print("  ⚠️  No component data to plot")
        return
    
    # Sort by importance
    sorted_components = sorted(component_avg.items(), key=lambda x: x[1], reverse=True)
    comp_names, comp_scores = zip(*sorted_components)
    
    # Plot 1: Bar chart of component types
    colors = plt.cm.Set3(np.linspace(0, 1, len(comp_names)))
    bars = ax1.bar(range(len(comp_names)), comp_scores, color=colors)
    ax1.set_xticks(range(len(comp_names)))
    ax1.set_xticklabels(comp_names, rotation=45, ha='right')
    ax1.set_ylabel('Average Importance Score', fontsize=12)
    ax1.set_title('Importance by Component Type', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2e}', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Box plot showing distribution
    box_data = []
    box_labels = []
    for comp_type in comp_names:
        all_scores = []
        for scores in categories[comp_type].values():
            all_scores.extend(scores)
        if all_scores:
            box_data.append(all_scores)
            box_labels.append(comp_type)
    
    bp = ax2.boxplot(box_data, labels=box_labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax2.set_xticklabels(box_labels, rotation=45, ha='right')
    ax2.set_ylabel('Importance Score', fontsize=12)
    ax2.set_title('Importance Distribution by Component', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_yscale('log')
    
    plt.tight_layout()
    output_file = output_dir / "component_importance.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file}")
    plt.close()


def plot_top_weights(importance_scores, output_dir: Path, top_n=30):
    """Plot the most important weights."""
    # Sort by importance
    sorted_weights = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
    top_weights = sorted_weights[:top_n]
    
    names, scores = zip(*top_weights)
    
    # Shorten names for better display
    short_names = []
    for name in names:
        if len(name) > 40:
            short_names.append(name[:37] + "...")
        else:
            short_names.append(name)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    y_pos = np.arange(len(short_names))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(scores)))
    
    bars = ax.barh(y_pos, scores, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(short_names, fontsize=9)
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_title(f'Top {top_n} Most Important Weights', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, score) in enumerate(zip(bars, scores)):
        ax.text(score, i, f' {score:.2e}', va='center', fontsize=8)
    
    plt.tight_layout()
    output_file = output_dir / "top_weights.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file}")
    plt.close()


def plot_all_weights_heatmap(importance_scores, output_dir: Path):
    """Create a simple heatmap of all weights."""
    # Sort by score
    sorted_weights = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Take top 100 for visualization
    top_n = min(100, len(sorted_weights))
    names, scores = zip(*sorted_weights[:top_n])
    
    # Shorten names
    short_names = [name[:50] if len(name) <= 50 else name[:47] + "..." for name in names]
    
    # Reshape into a matrix for better visualization
    n_cols = 10
    n_rows = (top_n + n_cols - 1) // n_cols
    
    matrix = np.zeros((n_rows, n_cols))
    for i, score in enumerate(scores):
        row = i // n_cols
        col = i % n_cols
        matrix[row, col] = score
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(16, 14))
    
    sns.heatmap(
        matrix,
        cmap='YlOrRd',
        ax=ax,
        cbar_kws={'label': 'Importance Score'},
        linewidths=0.5,
        linecolor='white',
        annot=False,
        square=False
    )
    
    ax.set_xlabel('Weight Index (groups of 10)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Weight Group', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {top_n} Weight Importance Heatmap', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    output_file = output_dir / "all_weights_heatmap.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file}")
    plt.close()
    
    # Create a more detailed version with weight names
    # Organize into matrix by weight type
    weight_types = {}
    for name, score in importance_scores.items():
        # Extract weight type
        if ".attn_" in name:
            wtype = "attention"
        elif ".ffn_" in name:
            wtype = "ffn"
        else:
            wtype = "other"
        
        if wtype not in weight_types:
            weight_types[wtype] = []
        weight_types[wtype].append((name, score))
    
    # Create separate heatmaps for each weight type
    for wtype, weights in weight_types.items():
        if not weights:
            continue
        
        weights_sorted = sorted(weights, key=lambda x: x[1], reverse=True)
        n_show = min(50, len(weights_sorted))
        names, scores = zip(*weights_sorted[:n_show])
        
        # Create matrix
        n_cols = min(5, n_show)
        n_rows = (n_show + n_cols - 1) // n_cols
        matrix = np.zeros((n_rows, n_cols))
        
        for i, score in enumerate(scores):
            row = i // n_cols
            col = i % n_cols
            matrix[row, col] = score
        
        fig, ax = plt.subplots(figsize=(12, max(8, n_rows * 0.4)))
        
        sns.heatmap(
            matrix,
            cmap='RdYlGn_r',
            ax=ax,
            cbar_kws={'label': 'Importance Score'},
            linewidths=1,
            linecolor='white',
            annot=True,
            fmt='.1f',
            square=False
        )
        
        ax.set_title(f'{wtype.upper()} Weights Importance (Top {n_show})', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        output_file = output_dir / f"{wtype}_weights_heatmap.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_file}")
        plt.close()


def plot_full_heatmap(importance_scores, output_dir: Path):
    """Create comprehensive heatmap visualizations with layers on Y-axis."""
    # Organize data by layer and component
    layer_component_data = defaultdict(lambda: defaultdict(list))
    
    for name, score in importance_scores.items():
        # Parse layer and component
        if "blk." in name:
            parts = name.split(".")
            try:
                # The format is: blk.N.component.weight
                # parts[0] = 'blk', parts[1] = 'N' (layer number)
                if parts[0] == "blk" and len(parts) >= 2:
                    layer_num = int(parts[1])
                else:
                    continue
            except (ValueError, IndexError):
                continue
            
            # Determine component type with more detail
            # Skip parts[0] (blk) and parts[1] (layer number)
            component_name = ".".join(parts[2:])
            
            # Parse component names to separate experts
            if "attn_q" in component_name:
                comp = "Attn_Q"
            elif "attn_k" in component_name:
                comp = "Attn_K"
            elif "attn_v" in component_name:
                comp = "Attn_V"
            elif "attn_output" in component_name:
                comp = "Attn_Out"
            elif "ffn_gate_inp" in component_name:
                comp = "FFN_Gate"
            elif "ffn_gate_exps" in component_name:
                # This is for MoE models with multiple experts
                comp = "FFN_Gate_Exp"
            elif "ffn_up_exps" in component_name:
                comp = "FFN_Up_Exp"
            elif "ffn_down_exps" in component_name:
                comp = "FFN_Down_Exp"
            elif "ffn_gate" in component_name:
                comp = "FFN_Gate"
            elif "ffn_up" in component_name:
                comp = "FFN_Up"
            elif "ffn_down" in component_name:
                comp = "FFN_Down"
            else:
                comp = "Other"
            
            layer_component_data[layer_num][comp].append(score)
    
    # Also create a simple all-weights heatmap regardless
    plot_all_weights_heatmap(importance_scores, output_dir)
    
    if not layer_component_data:
        print("  ⚠️  No layer-structured data for detailed heatmap")
        return
    
    # Convert to matrix
    layers = sorted(layer_component_data.keys())
    all_components = set()
    for layer_data in layer_component_data.values():
        all_components.update(layer_data.keys())
    
    # Order components logically: Attention first, then FFN
    component_order = []
    for comp in ['Attn_Q', 'Attn_K', 'Attn_V', 'Attn_Out']:
        if comp in all_components:
            component_order.append(comp)
    for comp in ['FFN_Gate', 'FFN_Up', 'FFN_Down', 'FFN_Gate_Exp', 'FFN_Up_Exp', 'FFN_Down_Exp']:
        if comp in all_components:
            component_order.append(comp)
    for comp in sorted(all_components):
        if comp not in component_order:
            component_order.append(comp)
    
    components = component_order
    
    # Build heatmap matrix (average multiple scores per component)
    matrix = np.zeros((len(layers), len(components)))
    for i, layer in enumerate(layers):
        for j, comp in enumerate(components):
            scores = layer_component_data[layer].get(comp, [])
            if scores:
                matrix[i, j] = np.mean(scores)
    
    # Create main heatmap: Layers (Y) x Components (X)
    fig, ax = plt.subplots(figsize=(max(12, len(components) * 0.8), max(10, len(layers) * 0.35)))
    
    # Use log scale for better visualization
    matrix_log = np.log10(matrix + 1e-10)  # Add small value to avoid log(0)
    
    sns.heatmap(
        matrix_log,
        xticklabels=components,
        yticklabels=layers,
        cmap='YlOrRd',
        ax=ax,
        cbar_kws={'label': 'log10(Importance Score)'},
        linewidths=0.5,
        linecolor='gray',
        annot=False
    )
    
    ax.set_xlabel('Tensor Type (Q, K, V, FFN Experts)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Layer Index', fontsize=13, fontweight='bold')
    ax.set_title('Weight Importance: Layer × Tensor Heatmap (Log Scale)', fontsize=15, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    output_file = output_dir / "layer_x_tensor_heatmap_log.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file}")
    plt.close()
    
    # Create a linear scale version with annotations
    fig, ax = plt.subplots(figsize=(max(12, len(components) * 0.8), max(10, len(layers) * 0.35)))
    
    # Show annotations for smaller matrices
    show_annot = len(layers) * len(components) < 200
    
    sns.heatmap(
        matrix,
        xticklabels=components,
        yticklabels=layers,
        cmap='RdYlGn_r',
        ax=ax,
        cbar_kws={'label': 'Importance Score'},
        linewidths=0.5,
        linecolor='white',
        annot=show_annot,
        fmt='.1f' if show_annot else None
    )
    
    ax.set_xlabel('Tensor Type (Q, K, V, FFN Experts)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Layer Index', fontsize=13, fontweight='bold')
    ax.set_title('Weight Importance: Layer × Tensor Heatmap (Linear Scale)', fontsize=15, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    output_file = output_dir / "layer_x_tensor_heatmap.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file}")
    plt.close()
    
    # Create attention-only heatmap if present
    attn_components = [c for c in components if 'Attn' in c]
    if attn_components:
        attn_matrix = np.zeros((len(layers), len(attn_components)))
        for i, layer in enumerate(layers):
            for j, comp in enumerate(attn_components):
                scores = layer_component_data[layer].get(comp, [])
                attn_matrix[i, j] = np.mean(scores) if scores else 0
        
        fig, ax = plt.subplots(figsize=(8, max(10, len(layers) * 0.35)))
        sns.heatmap(
            attn_matrix,
            xticklabels=attn_components,
            yticklabels=layers,
            cmap='RdYlBu_r',
            ax=ax,
            cbar_kws={'label': 'Importance Score'},
            annot=True,
            fmt='.1f',
            linewidths=1,
            linecolor='white'
        )
        
        ax.set_xlabel('Attention Tensor', fontsize=12, fontweight='bold')
        ax.set_ylabel('Layer Index', fontsize=12, fontweight='bold')
        ax.set_title('Attention Weight Importance by Layer', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        output_file = output_dir / "attention_by_layer_heatmap.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_file}")
        plt.close()
    
    # Create FFN-only heatmap if present
    ffn_components = [c for c in components if 'FFN' in c]
    if ffn_components:
        ffn_matrix = np.zeros((len(layers), len(ffn_components)))
        for i, layer in enumerate(layers):
            for j, comp in enumerate(ffn_components):
                scores = layer_component_data[layer].get(comp, [])
                ffn_matrix[i, j] = np.mean(scores) if scores else 0
        
        fig, ax = plt.subplots(figsize=(max(10, len(ffn_components) * 1.2), max(10, len(layers) * 0.35)))
        sns.heatmap(
            ffn_matrix,
            xticklabels=ffn_components,
            yticklabels=layers,
            cmap='YlGnBu',
            ax=ax,
            cbar_kws={'label': 'Importance Score'},
            annot=len(ffn_components) <= 10,
            fmt='.1f' if len(ffn_components) <= 10 else None,
            linewidths=0.5,
            linecolor='white'
        )
        
        ax.set_xlabel('FFN Tensor / Expert', fontsize=12, fontweight='bold')
        ax.set_ylabel('Layer Index', fontsize=12, fontweight='bold')
        ax.set_title('FFN Weight Importance by Layer', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        output_file = output_dir / "ffn_by_layer_heatmap.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_file}")
        plt.close()


def print_statistics(importance_scores, categories, metadata):
    """Print summary statistics."""
    print("\n" + "="*70)
    print("IMPORTANCE MATRIX STATISTICS")
    print("="*70)
    
    # Metadata
    print("\n📊 Metadata:")
    for key, value in metadata.items():
        if isinstance(value, (str, int, float)):
            print(f"  {key}: {value}")
    
    # Overall statistics
    all_scores = list(importance_scores.values())
    print(f"\n📈 Overall Statistics:")
    print(f"  Total weights: {len(all_scores)}")
    print(f"  Mean importance: {np.mean(all_scores):.4e}")
    print(f"  Std deviation: {np.std(all_scores):.4e}")
    print(f"  Min importance: {np.min(all_scores):.4e}")
    print(f"  Max importance: {np.max(all_scores):.4e}")
    print(f"  Median importance: {np.median(all_scores):.4e}")
    
    # Component statistics
    print(f"\n🔧 Importance by Component Type:")
    comp_stats = []
    for comp_type, layers in categories.items():
        all_comp_scores = []
        for scores in layers.values():
            all_comp_scores.extend(scores)
        if all_comp_scores:
            comp_stats.append((comp_type, np.mean(all_comp_scores), len(all_comp_scores)))
    
    comp_stats.sort(key=lambda x: x[1], reverse=True)
    for comp_type, avg_score, count in comp_stats:
        print(f"  {comp_type:20s}: {avg_score:.4e} (n={count})")
    
    # Top weights
    print(f"\n🏆 Top 10 Most Important Weights:")
    sorted_weights = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
    for i, (name, score) in enumerate(sorted_weights[:10], 1):
        print(f"  {i:2d}. {name:50s}: {score:.4e}")
    
    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize importance matrix (imatrix) from GGUF files"
    )
    parser.add_argument(
        "imatrix_file",
        type=Path,
        help="Path to the imatrix GGUF file"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=Path("imatrix"),
        help="Output directory for plots (default: ./imatrix)"
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=30,
        help="Number of top weights to display (default: 30)"
    )
    
    args = parser.parse_args()
    
    # Validate input
    if not args.imatrix_file.exists():
        print(f"Error: File not found: {args.imatrix_file}")
        return 1
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse imatrix
    importance_scores, metadata = parse_imatrix(args.imatrix_file)
    
    if not importance_scores:
        print("Error: No importance data found in the imatrix file")
        return 1
    
    # Categorize weights
    categories = categorize_weights(importance_scores)
    
    # Print statistics
    print_statistics(importance_scores, categories, metadata)
    
    # Generate visualizations
    print(f"\n📊 Generating visualizations...")
    plot_layer_importance(importance_scores, args.output_dir)
    plot_component_importance(categories, args.output_dir)
    plot_top_weights(importance_scores, args.output_dir, args.top_n)
    plot_full_heatmap(importance_scores, args.output_dir)
    
    print(f"\n✅ Done! Visualizations saved to: {args.output_dir}")
    print(f"\nGenerated files:")
    for plot_file in args.output_dir.glob("*.png"):
        print(f"  - {plot_file.name}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())







