"""Meta-design package: gt_eval scoring and material resolver for MCE environments."""

from .gt_eval import evaluate_gt_eval, validate_gt_eval
from .materials.material_resolver import MaterialChoice, MaterialResolver, ResolvedMaterial

__all__ = [
    "evaluate_gt_eval",
    "validate_gt_eval",
    "MaterialChoice",
    "MaterialResolver",
    "ResolvedMaterial",
]
