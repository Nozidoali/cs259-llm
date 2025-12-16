import argparse
import torch
from transformers import AutoConfig, AutoModelForCausalLM
from graphviz import Digraph

# Pick reasonably small / standard checkpoints to avoid totally exploding RAM/VRAM.
MODEL_MAP = {
    # Dense models (same architecture)
    "dense": "Qwen/Qwen2-0.5B",          # Use Qwen2 as reference

    # MoE models (same architecture)
    "moe": "Qwen/Qwen1.5-MoE-A2.7B",     # Use Qwen2-MoE as reference
}

# Model size variants for display
MODEL_SIZES = {
    "dense": {
        "Qwen2": ["0.5B", "1.5B", "7B", "72B"],
        "Qwen3": ["0.6B", "1.8B", "3B", "8B", "14B", "32B", "70B"]
    },
    "moe": {
        "Qwen2-MoE": ["A2.7B (14B total)", "57B-A14B"],
        "Qwen3-MoE": ["A3B (30B total)"]
    }
}


def analyze_layer(layer):
    """Analyze a single transformer layer and extract component info."""
    info = {
        'type': layer.__class__.__name__,
        'self_attn': {},
        'mlp': {},
        'norms': []
    }

    # Analyze self-attention
    if hasattr(layer, 'self_attn'):
        attn = layer.self_attn
        info['self_attn']['type'] = attn.__class__.__name__

        # Check for Q, K, V, O projections
        for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
            if hasattr(attn, proj_name):
                proj = getattr(attn, proj_name)
                info['self_attn'][proj_name] = proj.__class__.__name__

        # Check for rotary embeddings
        if hasattr(attn, 'rotary_emb'):
            info['self_attn']['rotary_emb'] = attn.rotary_emb.__class__.__name__

    # Analyze MLP/FFN
    if hasattr(layer, 'mlp'):
        mlp = layer.mlp
        info['mlp']['type'] = mlp.__class__.__name__

        # Check if it's MoE
        if hasattr(mlp, 'gate') and hasattr(mlp, 'experts'):
            # MoE structure
            info['mlp']['is_moe'] = True
            info['mlp']['router'] = mlp.gate.__class__.__name__

            if hasattr(mlp, 'experts'):
                info['mlp']['num_experts'] = len(mlp.experts)
                # Analyze first expert structure
                if len(mlp.experts) > 0:
                    expert = mlp.experts[0]
                    info['mlp']['expert'] = {}
                    for name in ['gate_proj', 'up_proj', 'down_proj', 'act_fn']:
                        if hasattr(expert, name):
                            info['mlp']['expert'][name] = getattr(expert, name).__class__.__name__

            # Shared expert
            if hasattr(mlp, 'shared_expert'):
                shared = mlp.shared_expert
                info['mlp']['shared_expert'] = {}
                for name in ['gate_proj', 'up_proj', 'down_proj', 'act_fn']:
                    if hasattr(shared, name):
                        info['mlp']['shared_expert'][name] = getattr(shared, name).__class__.__name__

            if hasattr(mlp, 'shared_expert_gate'):
                info['mlp']['shared_expert_gate'] = mlp.shared_expert_gate.__class__.__name__
        else:
            # Dense FFN
            info['mlp']['is_moe'] = False
            for name in ['gate_proj', 'up_proj', 'down_proj', 'act_fn']:
                if hasattr(mlp, name):
                    info['mlp'][name] = getattr(mlp, name).__class__.__name__

    # Layer norms
    for norm_name in ['input_layernorm', 'post_attention_layernorm']:
        if hasattr(layer, norm_name):
            norm = getattr(layer, norm_name)
            info['norms'].append((norm_name, norm.__class__.__name__))

    return info


def create_architecture_diagram(model, config, model_key, model_sizes):
    """Create detailed architecture diagram with exact tensor names."""
    dot = Digraph(comment=f'{model_key.upper()} Architecture', format='pdf')
    dot.attr(rankdir='LR', splines='ortho', nodesep='0.2', ranksep='0.4', compound='true')
    dot.attr('node', shape='box', style='filled,rounded', fontname='Courier', fontsize='8', height='0.25', margin='0.08')
    dot.attr('edge', fontname='Courier', fontsize='7', arrowsize='0.6')

    # Color scheme
    colors = {
        'embedding': '#E8F4F8',
        'norm': '#D5F4E6',
        'attn_proj': '#D4E6F1',
        'attn_compute': '#B8E6F0',
        'ffn_gate': '#F9E79F',
        'ffn_up': '#F5CBA7',
        'ffn_down': '#E59866',
        'router': '#D7BDE2',
        'experts': '#C39BD3',
        'shared_expert': '#F8C8DC',
        'output': '#A9DFBF',
        'activation': '#FAD7A0'
    }

    # Get layer info
    layer0 = model.model.layers[0]
    layer_info = analyze_layer(layer0)
    num_layers = config.num_hidden_layers

    # === FFN STRUCTURE ===
    with dot.subgraph(name='cluster_ffn_structure') as layer:
        is_moe = layer_info['mlp']['is_moe']
        title = 'Sparse MoE FFN Structure' if is_moe else 'Dense FFN Structure'
        layer.attr(label=title, style='filled', color='#F5F5F5')

        # Input to FFN
        layer.node('ffn_input', 'Input\n[h]',
                  fillcolor=colors['norm'], shape='oval')

        prev = 'ffn_input'

        # === FFN / MoE ===
        if layer_info['mlp']['is_moe']:
            # === MoE Architecture ===
            with layer.subgraph(name='cluster_moe') as moe:
                moe.attr(label='Sparse MoE', style='filled', color='#FEF9E7')

                # Router
                moe.node('router', 'gate (Router)\n[h → num_experts]',
                        fillcolor=colors['router'])
                dot.edge(prev, 'router')

                # Experts block
                with moe.subgraph(name='cluster_experts') as exp:
                    exp.attr(label=f'Experts (sparse, {layer_info["mlp"].get("num_experts", "?")} total)',
                            style='filled', color='#F4ECF7')

                    # Show expert structure
                    if 'expert' in layer_info['mlp']:
                        expert_info = layer_info['mlp']['expert']

                        exp.node('exp_input', 'Selected Experts\n(top-k routing)',
                                fillcolor=colors['experts'], shape='box3d')
                        dot.edge('router', 'exp_input')

                        if 'gate_proj' in expert_info:
                            exp.node('exp_gate', 'gate_proj\n[h → ffn_dim]',
                                    fillcolor=colors['ffn_gate'])
                            dot.edge('exp_input', 'exp_gate')

                        if 'up_proj' in expert_info:
                            exp.node('exp_up', 'up_proj\n[h → ffn_dim]',
                                    fillcolor=colors['ffn_up'])
                            dot.edge('exp_input', 'exp_up')

                        if 'act_fn' in expert_info:
                            exp.node('exp_act', 'SiLU(gate) * up',
                                    fillcolor=colors['activation'], shape='ellipse')
                            dot.edge('exp_gate', 'exp_act')
                            dot.edge('exp_up', 'exp_act')
                            exp_mid = 'exp_act'
                        else:
                            exp_mid = 'exp_up'

                        if 'down_proj' in expert_info:
                            exp.node('exp_down', 'down_proj\n[ffn_dim → h]',
                                    fillcolor=colors['ffn_down'])
                            dot.edge(exp_mid, 'exp_down')
                            expert_out = 'exp_down'
                        else:
                            expert_out = exp_mid

                # Shared Expert (if exists)
                if 'shared_expert' in layer_info['mlp']:
                    with moe.subgraph(name='cluster_shared') as shared:
                        shared.attr(label='Shared Expert (dense)', style='filled', color='#FCF3CF')

                        sh_info = layer_info['mlp']['shared_expert']

                        if 'shared_expert_gate' in layer_info['mlp']:
                            shared.node('sh_gate_in', 'shared_expert_gate\n(gating)',
                                       fillcolor=colors['router'])
                            dot.edge(prev, 'sh_gate_in')

                        if 'gate_proj' in sh_info:
                            shared.node('sh_gate', 'gate_proj\n[h → ffn_dim]',
                                       fillcolor=colors['ffn_gate'])
                            dot.edge(prev, 'sh_gate')

                        if 'up_proj' in sh_info:
                            shared.node('sh_up', 'up_proj\n[h → ffn_dim]',
                                       fillcolor=colors['ffn_up'])
                            dot.edge(prev, 'sh_up')

                        if 'act_fn' in sh_info:
                            shared.node('sh_act', 'SiLU(gate) * up',
                                       fillcolor=colors['activation'], shape='ellipse')
                            dot.edge('sh_gate', 'sh_act')
                            dot.edge('sh_up', 'sh_act')
                            sh_mid = 'sh_act'
                        else:
                            sh_mid = 'sh_up'

                        if 'down_proj' in sh_info:
                            shared.node('sh_down', 'down_proj\n[ffn_dim → h]',
                                       fillcolor=colors['ffn_down'])
                            dot.edge(sh_mid, 'sh_down')
                            shared_out = 'sh_down'
                        else:
                            shared_out = sh_mid

                    # Combine experts and shared expert
                    moe.node('moe_combine', 'Weighted Sum\n(experts + shared)',
                            fillcolor=colors['norm'], shape='oval')
                    dot.edge(expert_out, 'moe_combine')
                    dot.edge(shared_out, 'moe_combine')
                    if 'shared_expert_gate' in layer_info['mlp']:
                        dot.edge('sh_gate_in', 'moe_combine', style='dashed')
                    ffn_out = 'moe_combine'
                else:
                    ffn_out = expert_out

        else:
            # === Dense FFN ===
            with layer.subgraph(name='cluster_ffn') as ffn:
                ffn.attr(label='Feed-Forward Network (Dense)', style='filled', color='#FEF9E7')

                mlp_info = layer_info['mlp']

                if 'gate_proj' in mlp_info:
                    ffn.node('gate_proj', 'gate_proj\n[h → ffn_dim]',
                            fillcolor=colors['ffn_gate'])
                    dot.edge(prev, 'gate_proj')

                if 'up_proj' in mlp_info:
                    ffn.node('up_proj', 'up_proj\n[h → ffn_dim]',
                            fillcolor=colors['ffn_up'])
                    dot.edge(prev, 'up_proj')

                if 'act_fn' in mlp_info:
                    ffn.node('act_fn', 'SiLU(gate) * up',
                            fillcolor=colors['activation'], shape='ellipse')
                    dot.edge('gate_proj', 'act_fn')
                    dot.edge('up_proj', 'act_fn')
                    ffn_mid = 'act_fn'
                else:
                    ffn_mid = 'up_proj'

                if 'down_proj' in mlp_info:
                    ffn.node('down_proj', 'down_proj\n[ffn_dim → h]',
                            fillcolor=colors['ffn_down'])
                    dot.edge(ffn_mid, 'down_proj')
                    ffn_out = 'down_proj'
                else:
                    ffn_out = ffn_mid

        # FFN Output
        layer.node('ffn_output', 'Output\n[h]',
                  fillcolor=colors['norm'], shape='oval')
        dot.edge(ffn_out, 'ffn_output')

    # === MODEL SIZES ===
    with dot.subgraph(name='cluster_sizes') as sizes:
        sizes.attr(label='Available Model Sizes', style='filled', color='#F0F0F0')

        # Create a table for each model family
        for idx, (family, size_list) in enumerate(model_sizes.items()):
            size_text = f"{family}\\l"
            size_text += "=" * (len(family) + 2) + "\\l"
            for size in size_list:
                size_text += f"  • {size}\\l"

            node_id = f'sizes_{idx}'
            sizes.node(node_id, size_text,
                      shape='box',
                      fillcolor='#FFFACD',
                      style='filled',
                      fontname='Courier',
                      fontsize='8',
                      align='left')

    return dot


def build_and_draw(model_key: str, output_prefix: str):
    if model_key not in MODEL_MAP:
        raise ValueError(f"Unknown model_key {model_key}, "
                         f"choose from {list(MODEL_MAP.keys())}")

    model_id = MODEL_MAP[model_key]
    model_sizes = MODEL_SIZES[model_key]

    print(f"Loading config for {model_key} ({model_id}) ...")
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

    print(f"\n===== {model_key} FFN architecture =====")
    print(f"Reference model: {model_id}")
    print(f"hidden_size        : {config.hidden_size}")
    print(f"num_hidden_layers  : {config.num_hidden_layers}")

    if hasattr(config, 'num_experts'):
        print(f"num_experts        : {config.num_experts}")
    if hasattr(config, 'num_experts_per_tok'):
        print(f"num_experts_per_tok: {config.num_experts_per_tok}")

    print("\nAvailable model sizes:")
    for family, sizes in model_sizes.items():
        print(f"  {family}: {', '.join(sizes)}")

    print("=====================================\n")

    print(f"Loading model structure (no weights) ...")
    with torch.device('meta'):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

    print("Creating FFN architecture diagram...")
    dot = create_architecture_diagram(model, config, model_key, model_sizes)

    print(f"Rendering to {output_prefix}.pdf ...")
    dot.render(output_prefix, cleanup=True)
    print(f"Saved to {output_prefix}.pdf")

    dot.format = 'png'
    dot.render(output_prefix, cleanup=True)
    print(f"Saved to {output_prefix}.png")

    print("Done!")


def main():
    parser = argparse.ArgumentParser(
        description="Draw Qwen FFN architecture diagrams"
    )
    parser.add_argument(
        "model",
        choices=["dense", "moe"],
        help="Which architecture to draw: 'dense' (Qwen2/3) or 'moe' (Qwen2/3-MoE)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output filename prefix (default: qwen_dense or qwen_moe)",
    )
    args = parser.parse_args()

    prefix = args.out or f"qwen_{args.model}"
    build_and_draw(args.model, prefix)


if __name__ == "__main__":
    main()
