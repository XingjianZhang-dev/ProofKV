"""ProofKV evidence indexing, typed proof selection, and KV materialization.

The package contains the paper's query-conditioned evidence compiler, learned
relation scorer, physical compact-prefill path, and serving integrations.
"""

from .config import CompassConfig, Route
from .routing import RouteTable
from .controllers import CompassKVController

__all__ = ["CompassConfig", "Route", "RouteTable", "CompassKVController"]
