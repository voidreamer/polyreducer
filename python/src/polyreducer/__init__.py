"""Poly Reducer - Smart headless polygon reduction."""

__version__ = "0.1.0"

from polyreducer.reducer import PolyReducer, reduce, analyze
from polyreducer.models import ReductionResult, MeshAnalysis, LODSpec
from polyreducer.lod import LODChain

__all__ = [
    "PolyReducer",
    "reduce",
    "analyze",
    "ReductionResult",
    "MeshAnalysis",
    "LODSpec",
    "LODChain",
]
