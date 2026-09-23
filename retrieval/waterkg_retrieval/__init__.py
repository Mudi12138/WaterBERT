"""Hybrid (entity graph + dense) literature retrieval over the WaterKG graph."""
from .pipeline import HybridRetriever
from .store import GraphStore, build_store

__all__ = ["HybridRetriever", "GraphStore", "build_store"]
__version__ = "1.0.0"
