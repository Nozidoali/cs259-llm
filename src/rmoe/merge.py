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
):
    logger.info(f"Merging {len(expert_paths)} expert models into MoE model")
    logger.info(f"Gating network: {gating_model_path}")
    logger.info(f"Output directory: {output_dir}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    moe_model = MoEModel(
        expert_paths=expert_paths,
        gating_model_path=gating_model_path,
        base_model_path=base_model_path,
        routing_mode=routing_mode,
        device=device,
    )
    logger.info("Saving merged MoE model...")
    moe_model.save_pretrained(str(output_dir))
    logger.info(f"Merged MoE model saved to: {output_dir}")
    return output_dir

