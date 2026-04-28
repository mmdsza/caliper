"""Back-compat re-export of the SDK grader loader.

The canonical implementation lives in ``easytrain_sdk._loader`` so the
sandbox driver can import it without depending on the server package.
"""

from __future__ import annotations

from easytrain_sdk import GraderLoadError, LoadedGrader, load_grader_from_source

__all__ = ["GraderLoadError", "LoadedGrader", "load_grader_from_source"]
