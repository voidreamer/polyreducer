"""Main reducer functionality."""

from __future__ import annotations

import json
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional, Union, Callable

from poly_reducer.models import (
    MeshAnalysis,
    ReductionResult,
    ReductionSettings,
)


def find_blender() -> Optional[str]:
    """Find Blender executable."""
    import os
    import shutil
    
    # Check environment variable
    if blender_path := os.environ.get("BLENDER_PATH"):
        if Path(blender_path).exists():
            return blender_path
    
    # Check PATH
    if blender := shutil.which("blender"):
        return blender
    
    # Common locations
    common_paths = [
        "/usr/bin/blender",
        "/usr/local/bin/blender",
        "/opt/blender/blender",
        "/Applications/Blender.app/Contents/MacOS/Blender",
        "C:/Program Files/Blender Foundation/Blender/blender.exe",
    ]
    
    for path in common_paths:
        if Path(path).exists():
            return path
    
    return None


class PolyReducer:
    """Main polygon reducer class."""
    
    def __init__(
        self,
        blender_path: Optional[str] = None,
        timeout: int = 300,
        default_settings: Optional[dict] = None,
    ) -> None:
        """Initialize reducer.
        
        Args:
            blender_path: Path to Blender executable
            timeout: Timeout in seconds for Blender operations
            default_settings: Default reduction settings
        """
        self.blender_path = blender_path or find_blender()
        if not self.blender_path:
            raise RuntimeError(
                "Blender not found. Install Blender or set BLENDER_PATH environment variable."
            )
        
        self.timeout = timeout
        self.default_settings = default_settings or {}
    
    def analyze(self, source: Union[str, Path]) -> MeshAnalysis:
        """Analyze a mesh file.
        
        Args:
            source: Path to mesh file
            
        Returns:
            MeshAnalysis with mesh statistics
        """
        source = Path(source)
        if not source.exists():
            raise FileNotFoundError(f"Source file not found: {source}")
        
        # Blender script for analysis
        script = '''
import bpy
import json
import sys

# Clear scene
bpy.ops.wm.read_factory_settings(use_empty=True)

# Import file
filepath = sys.argv[-1]
ext = filepath.lower().split(".")[-1]

if ext in ("fbx",):
    bpy.ops.import_scene.fbx(filepath=filepath)
elif ext in ("obj",):
    bpy.ops.wm.obj_import(filepath=filepath)
elif ext in ("gltf", "glb"):
    bpy.ops.import_scene.gltf(filepath=filepath)
elif ext in ("blend",):
    bpy.ops.wm.open_mainfile(filepath=filepath)
else:
    print(json.dumps({"error": f"Unsupported format: {ext}"}))
    sys.exit(1)

# Analyze all meshes
total_tris = 0
total_verts = 0
total_edges = 0
total_faces = 0
has_uvs = False
sharp_edges = 0
boundary_edges = 0
materials = set()

bounds_min = [float("inf")] * 3
bounds_max = [float("-inf")] * 3

for obj in bpy.data.objects:
    if obj.type != "MESH":
        continue
    
    mesh = obj.data
    
    # Apply transforms for accurate bounds
    obj_matrix = obj.matrix_world
    
    for vert in mesh.vertices:
        world_co = obj_matrix @ vert.co
        for i in range(3):
            bounds_min[i] = min(bounds_min[i], world_co[i])
            bounds_max[i] = max(bounds_max[i], world_co[i])
    
    total_verts += len(mesh.vertices)
    total_edges += len(mesh.edges)
    total_faces += len(mesh.polygons)
    
    # Count triangles
    for poly in mesh.polygons:
        verts = len(poly.vertices)
        if verts == 3:
            total_tris += 1
        elif verts == 4:
            total_tris += 2
        else:
            total_tris += verts - 2
    
    # Check UVs
    if mesh.uv_layers:
        has_uvs = True
    
    # Count sharp edges
    for edge in mesh.edges:
        if edge.use_edge_sharp:
            sharp_edges += 1
        if not edge.is_manifold:
            boundary_edges += 1
    
    # Materials
    for slot in obj.material_slots:
        if slot.material:
            materials.add(slot.material.name)

result = {
    "triangle_count": total_tris,
    "vertex_count": total_verts,
    "edge_count": total_edges,
    "face_count": total_faces,
    "has_uvs": has_uvs,
    "sharp_edge_count": sharp_edges,
    "boundary_edge_count": boundary_edges,
    "material_count": len(materials),
    "bounds_min": bounds_min if bounds_min[0] != float("inf") else [0, 0, 0],
    "bounds_max": bounds_max if bounds_max[0] != float("-inf") else [0, 0, 0],
}

print("ANALYSIS_RESULT:" + json.dumps(result))
'''
        
        result = self._run_blender_script(script, str(source))
        
        # Parse result
        for line in result.splitlines():
            if line.startswith("ANALYSIS_RESULT:"):
                data = json.loads(line.replace("ANALYSIS_RESULT:", ""))
                
                analysis = MeshAnalysis(
                    triangle_count=data["triangle_count"],
                    vertex_count=data["vertex_count"],
                    edge_count=data["edge_count"],
                    face_count=data["face_count"],
                    has_uvs=data["has_uvs"],
                    sharp_edge_count=data.get("sharp_edge_count", 0),
                    boundary_edge_count=data.get("boundary_edge_count", 0),
                    material_count=data.get("material_count", 0),
                    bounds_min=tuple(data.get("bounds_min", [0, 0, 0])),
                    bounds_max=tuple(data.get("bounds_max", [0, 0, 0])),
                )
                
                # Compute suggestions
                analysis.compute_suggestions()
                
                return analysis
        
        raise RuntimeError(f"Failed to analyze mesh: {result}")
    
    def reduce(
        self,
        source: Union[str, Path],
        target_tris: Optional[int] = None,
        ratio: Optional[float] = None,
        output: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> ReductionResult:
        """Reduce polygon count of a mesh.
        
        Args:
            source: Path to source mesh file
            target_tris: Target triangle count
            ratio: Reduction ratio (0-1)
            output: Output file path
            **kwargs: Additional settings (preserve_uvs, preserve_normals, etc.)
            
        Returns:
            ReductionResult with reduction statistics
        """
        start_time = time.time()
        
        source = Path(source)
        if not source.exists():
            return ReductionResult(
                success=False,
                source_path=source,
                error_message=f"Source file not found: {source}",
            )
        
        # Merge settings
        settings = {**self.default_settings, **kwargs}
        
        # Determine output path
        if output:
            output_path = Path(output)
        else:
            output_path = source.with_stem(f"{source.stem}_reduced")
        
        # Calculate ratio if target_tris specified
        if target_tris and not ratio:
            # We need to analyze first to get original tri count
            analysis = self.analyze(source)
            if analysis.triangle_count > 0:
                ratio = target_tris / analysis.triangle_count
                ratio = min(1.0, max(0.01, ratio))  # Clamp
        
        if not ratio:
            ratio = 0.5  # Default 50% reduction
        
        # Build Blender script
        script = self._build_reduction_script(
            ratio=ratio,
            preserve_uvs=settings.get("preserve_uvs", True),
            preserve_normals=settings.get("preserve_normals", True),
            preserve_boundaries=settings.get("preserve_boundaries", True),
            symmetry=settings.get("symmetry", False),
        )
        
        # Run reduction
        result_json = self._run_blender_script(
            script,
            str(source),
            str(output_path),
        )
        
        # Parse result
        for line in result_json.splitlines():
            if line.startswith("REDUCTION_RESULT:"):
                data = json.loads(line.replace("REDUCTION_RESULT:", ""))
                
                return ReductionResult(
                    success=data.get("success", False),
                    source_path=source,
                    output_path=output_path if data.get("success") else None,
                    original_tris=data.get("original_tris", 0),
                    final_tris=data.get("final_tris", 0),
                    total_time_ms=(time.time() - start_time) * 1000,
                    error_message=data.get("error"),
                )
        
        return ReductionResult(
            success=False,
            source_path=source,
            error_message="Failed to parse reduction result",
            total_time_ms=(time.time() - start_time) * 1000,
        )
    
    def _build_reduction_script(
        self,
        ratio: float,
        preserve_uvs: bool,
        preserve_normals: bool,
        preserve_boundaries: bool,
        symmetry: bool,
    ) -> str:
        """Build Blender Python script for reduction."""
        return f'''
import bpy
import json
import sys

# Clear scene
bpy.ops.wm.read_factory_settings(use_empty=True)

# Get args
filepath = sys.argv[-2]
output_path = sys.argv[-1]
ext = filepath.lower().split(".")[-1]

# Import
try:
    if ext in ("fbx",):
        bpy.ops.import_scene.fbx(filepath=filepath)
    elif ext in ("obj",):
        bpy.ops.wm.obj_import(filepath=filepath)
    elif ext in ("gltf", "glb"):
        bpy.ops.import_scene.gltf(filepath=filepath)
    else:
        print("REDUCTION_RESULT:" + json.dumps({{"error": f"Unsupported: {{ext}}"}}))
        sys.exit(1)
except Exception as e:
    print("REDUCTION_RESULT:" + json.dumps({{"error": str(e)}}))
    sys.exit(1)

# Count original tris
original_tris = 0
for obj in bpy.data.objects:
    if obj.type == "MESH":
        for poly in obj.data.polygons:
            v = len(poly.vertices)
            original_tris += 1 if v == 3 else (2 if v == 4 else v - 2)

# Apply decimate modifier to all meshes
for obj in bpy.data.objects:
    if obj.type != "MESH":
        continue
    
    bpy.context.view_layer.objects.active = obj
    
    # Add decimate modifier
    mod = obj.modifiers.new(name="Decimate", type="DECIMATE")
    mod.ratio = {ratio}
    mod.use_collapse_triangulate = True
    
    # Apply modifier
    bpy.ops.object.modifier_apply(modifier="Decimate")

# Count final tris
final_tris = 0
for obj in bpy.data.objects:
    if obj.type == "MESH":
        for poly in obj.data.polygons:
            v = len(poly.vertices)
            final_tris += 1 if v == 3 else (2 if v == 4 else v - 2)

# Export
out_ext = output_path.lower().split(".")[-1]
try:
    if out_ext in ("fbx",):
        bpy.ops.export_scene.fbx(filepath=output_path, use_selection=False)
    elif out_ext in ("obj",):
        bpy.ops.wm.obj_export(filepath=output_path)
    elif out_ext in ("gltf", "glb"):
        bpy.ops.export_scene.gltf(filepath=output_path)
    
    print("REDUCTION_RESULT:" + json.dumps({{
        "success": True,
        "original_tris": original_tris,
        "final_tris": final_tris,
    }}))
except Exception as e:
    print("REDUCTION_RESULT:" + json.dumps({{"success": False, "error": str(e)}}))
'''
    
    def _run_blender_script(self, script: str, *args: str) -> str:
        """Run a Blender script and return output."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(script)
            script_path = f.name
        
        try:
            cmd = [
                self.blender_path,
                "--background",
                "--python", script_path,
                "--",
                *args,
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            
            return result.stdout + result.stderr
            
        finally:
            Path(script_path).unlink(missing_ok=True)


# Convenience functions
def reduce(
    source: Union[str, Path],
    target_tris: Optional[int] = None,
    ratio: Optional[float] = None,
    output: Optional[Union[str, Path]] = None,
    **kwargs,
) -> ReductionResult:
    """Reduce polygon count of a mesh.
    
    Convenience function that creates a PolyReducer instance.
    """
    reducer = PolyReducer()
    return reducer.reduce(source, target_tris=target_tris, ratio=ratio, output=output, **kwargs)


def analyze(source: Union[str, Path]) -> MeshAnalysis:
    """Analyze a mesh file.
    
    Convenience function that creates a PolyReducer instance.
    """
    reducer = PolyReducer()
    return reducer.analyze(source)
