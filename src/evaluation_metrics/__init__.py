"""evaluation_metrics — Member 4 evaluation & tracking utilities.

Public API
----------
    from evaluation_metrics import LeakDetector, WandbTracker
    from evaluation_metrics.tracking_utils import WandbTracker, log_scalar_dict
    from evaluation_metrics.leak_detector import LeakDetector, LeakResult
"""

from .leak_detector import LeakDetector, LeakResult, Severity
from .tracking_utils import WandbTracker, log_scalar_dict

__all__ = [
    "LeakDetector",
    "LeakResult",
    "Severity",
    "WandbTracker",
    "log_scalar_dict",
]
