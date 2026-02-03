# Poly Reducer

Smart headless polygon reduction for VFX/Game pipelines. Uses Blender for decimation with intelligent analysis for optimal results.

[![Python CI](https://github.com/voidreamer/poly-reducer/actions/workflows/python-ci.yml/badge.svg)](https://github.com/voidreamer/poly-reducer/actions/workflows/python-ci.yml)

## Features

- **Headless operation** — No GUI, perfect for pipeline automation
- **Smart analysis** — Automatically determines optimal reduction targets
- **Quality preservation** — Protects UVs, normals, sharp edges, vertex colors
- **Batch processing** — Process entire directories of models
- **Multiple backends** — Blender (default), OpenMesh, or custom
- **Format support** — FBX, OBJ, GLTF, USD, Alembic

## Installation

```bash
pip install poly-reducer

# Blender must be installed and accessible in PATH
# Or set BLENDER_PATH environment variable
```

## Quick Start

### CLI

```bash
# Basic reduction to target triangle count
polyreduce model.fbx --target-tris 10000

# Reduction by percentage
polyreduce model.fbx --ratio 0.5 --output model_lod1.fbx

# Preserve features
polyreduce model.fbx --target-tris 5000 --preserve-uvs --preserve-normals

# Analyze without reducing
polyreduce model.fbx --analyze

# Batch processing
polyreduce ./models/*.fbx --target-tris 10000 --output-dir ./lods/

# Generate LOD chain
polyreduce model.fbx --lod-chain 100000,50000,10000,1000
```

### Python API

```python
from poly_reducer import reduce, analyze, LODChain

# Simple reduction
result = reduce(
    "model.fbx",
    target_tris=10000,
    preserve_uvs=True,
    output="model_reduced.fbx",
)
print(f"Reduced from {result.original_tris} to {result.final_tris}")

# Analyze mesh first
analysis = analyze("model.fbx")
print(f"Triangles: {analysis.triangle_count}")
print(f"Recommended LOD targets: {analysis.suggested_lods}")
print(f"Feature edges: {analysis.feature_edge_count}")

# Generate LOD chain
lods = LODChain(
    source="model.fbx",
    targets=[100000, 50000, 10000, 1000],
    preserve_uvs=True,
).generate()

for lod in lods:
    print(f"LOD{lod.level}: {lod.triangle_count} tris -> {lod.output_path}")
```

### Pipeline Integration

```python
from poly_reducer import PolyReducer
from poly_reducer.backends import BlenderBackend

# Configure reducer
reducer = PolyReducer(
    backend=BlenderBackend(
        blender_path="/usr/bin/blender",
        timeout=300,
    ),
    default_settings={
        "preserve_uvs": True,
        "preserve_normals": True,
        "symmetry": True,
    },
)

# Process file
result = reducer.reduce("hero_character.fbx", target_tris=50000)

# Batch process with progress callback
def on_progress(file, progress, total):
    print(f"[{progress}/{total}] Processing {file}")

results = reducer.batch_reduce(
    files=["char1.fbx", "char2.fbx", "char3.fbx"],
    target_tris=10000,
    on_progress=on_progress,
)
```

## Analysis

Poly Reducer analyzes meshes to provide intelligent recommendations:

```python
from poly_reducer import analyze

analysis = analyze("complex_model.fbx")

print(f"Triangle count: {analysis.triangle_count}")
print(f"Vertex count: {analysis.vertex_count}")
print(f"Has UVs: {analysis.has_uvs}")
print(f"UV islands: {analysis.uv_island_count}")
print(f"Sharp edges: {analysis.sharp_edge_count}")
print(f"Boundary edges: {analysis.boundary_edge_count}")
print(f"Non-manifold edges: {analysis.non_manifold_count}")
print(f"Suggested LODs: {analysis.suggested_lods}")
print(f"Reduction complexity: {analysis.complexity_score}")  # 0-1
```

### Suggested LODs

The analyzer provides smart LOD recommendations based on:
- Mesh complexity and detail distribution
- Feature edge density
- UV seam locations
- Overall triangle budget

```python
# Example output
analysis.suggested_lods = [
    {"level": 0, "tris": 100000, "ratio": 1.0},    # Original
    {"level": 1, "tris": 50000, "ratio": 0.5},     # Medium
    {"level": 2, "tris": 10000, "ratio": 0.1},     # Low
    {"level": 3, "tris": 1000, "ratio": 0.01},     # Distant
]
```

## Options

### Reduction Settings

| Option | CLI | Python | Description |
|--------|-----|--------|-------------|
| Target triangles | `--target-tris` | `target_tris` | Absolute triangle count |
| Ratio | `--ratio` | `ratio` | Reduction ratio (0-1) |
| Preserve UVs | `--preserve-uvs` | `preserve_uvs` | Keep UV coordinates |
| Preserve normals | `--preserve-normals` | `preserve_normals` | Keep custom normals |
| Preserve boundaries | `--preserve-boundaries` | `preserve_boundaries` | Keep mesh boundaries |
| Symmetry | `--symmetry` | `symmetry` | Use symmetry for reduction |
| Vertex groups | `--preserve-groups` | `preserve_vertex_groups` | Keep vertex group weights |

### Output Settings

| Option | CLI | Python | Description |
|--------|-----|--------|-------------|
| Output path | `--output` | `output` | Output file path |
| Output format | `--format` | `output_format` | Force output format |
| Overwrite | `--overwrite` | `overwrite` | Overwrite existing files |

## Backends

### Blender (Default)

Uses Blender's Decimate modifier. Best quality, supports all features.

```python
from poly_reducer.backends import BlenderBackend

backend = BlenderBackend(
    blender_path="/path/to/blender",  # Optional, auto-detected
    timeout=300,  # Seconds
)
```

### OpenMesh (Experimental)

Pure Python/C++ implementation. Faster but fewer features.

```python
from poly_reducer.backends import OpenMeshBackend

backend = OpenMeshBackend()
```

## Performance

Poly Reducer uses Rust for mesh analysis when available:

```bash
# Install with Rust acceleration
pip install poly-reducer[rust]
```

Benchmarks (1M triangle mesh):
- Analysis: ~50ms (Rust) vs ~500ms (Python)
- Reduction: Depends on Blender (typically 5-30 seconds)

## Development

```bash
git clone https://github.com/voidreamer/poly-reducer.git
cd poly-reducer
uv sync --dev

# Run tests
uv run pytest

# Run with Blender tests (requires Blender)
BLENDER_PATH=/path/to/blender uv run pytest -m blender
```

## License

MIT © Alejandro Cabrera
