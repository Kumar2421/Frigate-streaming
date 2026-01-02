"""Tracker configuration classes."""

from enum import Enum
from typing import Dict, Optional

from pydantic import BaseModel, Field, ConfigDict


class TrackerTypeEnum(str, Enum):
    """Available tracker types."""
    norfair = "norfair"
    centroid = "centroid"
    deepocsort = "deepocsort"


class DeepOCSORTConfig(BaseModel):
    """DeepOCSORT tracker configuration."""
    
    # DeepOCSORT parameters
    det_thresh: float = Field(default=0.3, title="Detection threshold")
    max_age: int = Field(default=30, title="Maximum age for tracks")
    min_hits: int = Field(default=3, title="Minimum hits for track initialization")
    iou_threshold: float = Field(default=0.3, title="IoU threshold for association")
    delta_t: int = Field(default=3, title="Time delta for track association")
    asso_func: str = Field(default="giou", title="Association function")
    inertia: float = Field(default=0.2, title="Inertia parameter")
    
    # Re-identification parameters
    w_association_emb: float = Field(default=0.75, title="Weight for embedding association")
    alpha_fixed_emb: float = Field(default=0.95, title="Alpha for fixed embedding")
    aw_param: float = Field(default=0.5, title="Appearance weight parameter")
    embedding_off: bool = Field(default=False, title="Disable embedding features")
    cmc_off: bool = Field(default=False, title="Disable camera motion compensation")
    aw_off: bool = Field(default=False, title="Disable appearance weighting")
    new_kf_off: bool = Field(default=False, title="Disable new Kalman filter")
    
    # ReID model parameters
    reid_model_path: str = Field(default="osnet_x1_0", title="ReID model name or path")
    reid_device: str = Field(default="cpu", title="Device for re-identification model")
    reid_threshold: float = Field(default=0.7, title="Re-identification similarity threshold")
    
    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class TrackerConfig(BaseModel):
    """Main tracker configuration."""
    
    type: TrackerTypeEnum = Field(default=TrackerTypeEnum.norfair, title="Tracker type")
    deepocsort: Optional[DeepOCSORTConfig] = Field(default=None, title="DeepOCSORT configuration")
    
    model_config = ConfigDict(extra="forbid", protected_namespaces=())
