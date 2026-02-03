"""Poly Reducer - Smart headless polygon reduction."""

__version__ = "0.1.0"

from poly_reducer.reducer import PolyReducer, reduce, analyze
from poly_reducer.models import ReductionResult, MeshAnalysis, LODSpec
from poly_reducer.lod import LODChain

__all__ = [
    "PolyReducer",
    "reduce",
    "analyze",
    "ReductionResult",
    "MeshAnalysis",
    "LODSpec",
    "LODChain",
]
