import logging
from pathlib import Path
from rmoe.moemodel import MoEModel

logger = logging.getLogger(__name__)

def merge_experts(
    expert_paths,
    gating_model_path,
    output_dir,
    base_model_path=None,
    routing_mode="weighted_sum",
    device=None,
    shared_expert_path=None,
    num_experts_per_tok=None,
    use_zero_shared_expert=True,
    forced_expert_idx=None,
    use_per_token_routing=None,
    shared_expert_intermediate_size=None,
):
    logger.info(f"Merging {len(expert_paths)} expert models into Qwen2MoE model")
    logger.info(f"Gating network: {gating_model_path}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Using shared expert: enabled (required for Qwen2MoE)")
    if use_zero_shared_expert:
        logger.info(f"Shared expert: zero-initialized (no model loaded)")
    elif shared_expert_path:
        logger.info(f"Shared expert path: {shared_expert_path}")
    else:
        logger.info(f"Shared expert path: using base model")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    moe_model = MoEModel(
        expert_paths=expert_paths,
        gating_model_path=gating_model_path,
        base_model_path=base_model_path,
        routing_mode=routing_mode,
        device=device,
        shared_expert_path=shared_expert_path,
        num_experts_per_tok=num_experts_per_tok,
        use_zero_shared_expert=use_zero_shared_expert,
        forced_expert_idx=forced_expert_idx,
        use_per_token_routing=use_per_token_routing,
        shared_expert_intermediate_size=shared_expert_intermediate_size,
    )
    logger.info("Saving merged MoE model...")
    moe_model.save_pretrained(str(output_dir))
    logger.info(f"Merged MoE model saved to: {output_dir}")
    return output_dir

