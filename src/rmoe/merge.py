import logging
from pathlib import Path
from rmoe.moemodel import MoEModel

logger = logging.getLogger(__name__)

def merge_experts(
    expert_paths,
    gating_model_path,
    output_dir,
    base_model_path=None,
    routing_mode="sparse_top1",
    device=None,
    router_trainable=True,
):
    """
    Merge expert models into a sparse MoE model with top-1 routing.
    
    Args:
        expert_paths: List of paths to expert models
        gating_model_path: Path to trained gating network
        output_dir: Output directory for merged MoE model
        base_model_path: Optional base model path (defaults to first expert)
        routing_mode: Routing mode - "sparse_top1" (default), "weighted_sum", or "select_one"
        device: Device to use for model loading
        router_trainable: Whether router should be trainable during fine-tuning
    
    Returns:
        Path to output directory
    """
    logger.info(f"Merging {len(expert_paths)} expert models into MoE model")
    logger.info(f"Gating network: {gating_model_path}")
    logger.info(f"Routing mode: {routing_mode}")
    logger.info(f"Router trainable: {router_trainable}")
    logger.info(f"Output directory: {output_dir}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    moe_model = MoEModel(
        expert_paths=expert_paths,
        gating_model_path=gating_model_path,
        base_model_path=base_model_path,
        routing_mode=routing_mode,
        device=device,
        router_trainable=router_trainable,
    )
    logger.info("Saving merged MoE model with sparse gating...")
    moe_model.save_pretrained(str(output_dir))
    logger.info(f"Merged MoE model saved to: {output_dir}")
    logger.info(f"Expert weights are kept separate (side-by-side) for GGUF compatibility")
    return output_dir

