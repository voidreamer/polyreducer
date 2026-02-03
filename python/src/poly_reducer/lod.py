"""LOD chain generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union, Callable

from poly_reducer.models import LODSpec, ReductionResult
from poly_reducer.reducer import PolyReducer


@dataclass
class LODResult:
    """Result for a single LOD level."""
    level: int
    triangle_count: int
    output_path: Path
    reduction_ratio: float
    success: bool
    error_message: Optional[str] = None


class LODChain:
    """Generate a chain of LOD levels from a source mesh.
    
    Example:
        lods = LODChain(
            source="model.fbx",
            targets=[100000, 50000, 10000, 1000],
        ).generate()
    """
    
    def __init__(
        self,
        source: Union[str, Path],
        targets: Optional[list[int]] = None,
        ratios: Optional[list[float]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        output_pattern: str = "{stem}_lod{level}{ext}",
        preserve_uvs: bool = True,
        preserve_normals: bool = True,
        blender_path: Optional[str] = None,
    ) -> None:
        """Initialize LOD chain generator.
        
        Args:
            source: Path to source mesh
            targets: List of target triangle counts for each LOD
            ratios: Alternative: list of reduction ratios (0-1)
            output_dir: Output directory (default: same as source)
            output_pattern: Output filename pattern
            preserve_uvs: Preserve UV coordinates
            preserve_normals: Preserve custom normals
            blender_path: Path to Blender executable
        """
        self.source = Path(source)
        self.targets = targets
        self.ratios = ratios
        self.output_dir = Path(output_dir) if output_dir else self.source.parent
        self.output_pattern = output_pattern
        self.preserve_uvs = preserve_uvs
        self.preserve_normals = preserve_normals
        
        self.reducer = PolyReducer(blender_path=blender_path)
        
        if not targets and not ratios:
            # Use default LOD ratios
            self.ratios = [1.0, 0.5, 0.25, 0.1, 0.01]
    
    def generate(
        self,
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> list[LODResult]:
        """Generate all LOD levels.
        
        Args:
            on_progress: Optional callback (current_lod, total_lods)
            
        Returns:
            List of LODResult for each level
        """
        results = []
        
        # Analyze source first
        analysis = self.reducer.analyze(self.source)
        original_tris = analysis.triangle_count
        
        # Determine LOD specs
        specs = self._compute_specs(original_tris)
        
        for i, spec in enumerate(specs):
            if on_progress:
                on_progress(i, len(specs))
            
            # Generate output path
            output_path = self.output_dir / self.output_pattern.format(
                stem=self.source.stem,
                level=spec.level,
                ext=self.source.suffix,
            )
            
            # Skip LOD0 (original) - just copy or link
            if spec.ratio >= 1.0:
                import shutil
                self.output_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(self.source, output_path)
                
                results.append(LODResult(
                    level=spec.level,
                    triangle_count=original_tris,
                    output_path=output_path,
                    reduction_ratio=1.0,
                    success=True,
                ))
                continue
            
            # Reduce
            result = self.reducer.reduce(
                self.source,
                target_tris=spec.target_tris,
                output=output_path,
                preserve_uvs=self.preserve_uvs,
                preserve_normals=self.preserve_normals,
            )
            
            results.append(LODResult(
                level=spec.level,
                triangle_count=result.final_tris,
                output_path=output_path if result.success else Path(),
                reduction_ratio=result.reduction_ratio,
                success=result.success,
                error_message=result.error_message,
            ))
        
        if on_progress:
            on_progress(len(specs), len(specs))
        
        return results
    
    def _compute_specs(self, original_tris: int) -> list[LODSpec]:
        """Compute LOD specifications from targets or ratios."""
        specs = []
        
        if self.targets:
            for level, target in enumerate(self.targets):
                ratio = target / original_tris if original_tris > 0 else 1.0
                specs.append(LODSpec(
                    level=level,
                    target_tris=target,
                    ratio=min(1.0, ratio),
                ))
        elif self.ratios:
            for level, ratio in enumerate(self.ratios):
                specs.append(LODSpec(
                    level=level,
                    target_tris=int(original_tris * ratio),
                    ratio=ratio,
                ))
        
        return specs
