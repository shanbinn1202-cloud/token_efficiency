"""Private growing-history v2 composition.

The package is dormant unless it is constructed by the reviewed pre-B0 startup
path.  Importing it never enables a route or claims native readiness.
"""

from .readiness import NativeReadiness

__all__ = ["NativeReadiness"]
