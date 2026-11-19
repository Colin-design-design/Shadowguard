"""ShadowGuard — phonetic-shadow prompt-injection detection."""

from .guard import ShadowGuard
from .report import ScanResult, ThreatReport

__version__ = "0.1.0"
__all__ = ["ShadowGuard", "ScanResult", "ThreatReport"]
