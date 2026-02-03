"""Data models for poly-reducer."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class MeshAnalysis:
    """Analysis results for a mesh."""
    
    # Basic counts
    triangle_count: int
    vertex_count: int
    edge_count: int
    face_count: int
    
    # UV data
    has_uvs: bool
    uv_island_count: int = 0
    
    # Topology
    sharp_edge_count: int = 0
    boundary_edge_count: int = 0
    non_manifold_count: int = 0
    
    # Materials
    material_count: int = 0
    
    # Bounds
    bounds_min: tuple[float, float, float] = (0, 0, 0)
    bounds_max: tuple[float, float, float] = (0, 0, 0)
    
    # Computed metrics
    complexity_score: float = 0.0  # 0-1, higher = more complex
    detail_distribution: str = "uniform"  # uniform, concentrated, sparse
    
    # Suggestions
    suggested_lods: list[LODSpec] = field(default_factory=list)
    
    @property
    def bounds_size(self) -> tuple[float, float, float]:
        """Bounding box dimensions."""
        return (
            self.bounds_max[0] - self.bounds_min[0],
            self.bounds_max[1] - self.bounds_min[1],
            self.bounds_max[2] - self.bounds_min[2],
        )
    
    def compute_suggestions(self) -> None:
        """Compute suggested LOD targets based on analysis."""
        base = self.triangle_count
        
        # Simple geometric progression for now
        self.suggested_lods = [
            LODSpec(level=0, target_tris=base, ratio=1.0),
            LODSpec(level=1, target_tris=int(base * 0.5), ratio=0.5),
            LODSpec(level=2, target_tris=int(base * 0.25), ratio=0.25),
            LODSpec(level=3, target_tris=int(base * 0.1), ratio=0.1),
            LODSpec(level=4, target_tris=max(100, int(base * 0.01)), ratio=0.01),
        ]


@dataclass
class LODSpec:
    """Specification for a single LOD level."""
    level: int
    target_tris: int
    ratio: float
    output_path: Optional[Path] = None


@dataclass
class ReductionResult:
    """Result of a reduction operation."""
    
    success: bool
    source_path: Path
    output_path: Optional[Path] = None
    
    # Triangle counts
    original_tris: int = 0
    final_tris: int = 0
    
    # Timing
    analysis_time_ms: float = 0
    reduction_time_ms: float = 0
    total_time_ms: float = 0
    
    # Quality metrics
    uv_distortion: float = 0.0  # 0-1
    normal_deviation: float = 0.0  # degrees
    
    # Errors
    error_message: Optional[str] = None
    warnings: list[str] = field(default_factory=list)
    
    @property
    def reduction_ratio(self) -> float:
        """Actual reduction ratio achieved."""
        if self.original_tris == 0:
            return 0.0
        return self.final_tris / self.original_tris


@dataclass
class ReductionSettings:
    """Settings for reduction operation."""
    
    # Target (one of these should be set)
    target_tris: Optional[int] = None
    ratio: Optional[float] = None
    
    # Quality preservation
    preserve_uvs: bool = True
    preserve_normals: bool = True
    preserve_boundaries: bool = True
    preserve_vertex_groups: bool = False
    
    # Algorithm options
    symmetry: bool = False
    symmetry_axis: str = "X"
    
    # Output
    output_path: Optional[Path] = None
    output_format: Optional[str] = None
    overwrite: bool = False
    
    def validate(self) -> None:
        """Validate settings."""
        if self.target_tris is None and self.ratio is None:
            raise ValueError("Either target_tris or ratio must be specified")
        
        if self.ratio is not None and not (0 < self.ratio <= 1):
            raise ValueError("Ratio must be between 0 and 1")
