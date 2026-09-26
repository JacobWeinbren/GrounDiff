"""GrounDiff-style DSM -> DTM and lasground_new -> hand-edited DTM models."""
import os as _os

# Must be set before torch is first imported: lets ops that Apple's MPS
# backend lacks fall back to the CPU instead of raising.
_os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
