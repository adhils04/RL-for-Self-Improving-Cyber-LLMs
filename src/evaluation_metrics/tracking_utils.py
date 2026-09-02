"""evaluation_metrics/tracking_utils.py — Member 4: W&B logger initialization.

Provides a thin, testable wrapper around wandb so that every script that
needs experiment tracking uses the same initialization logic and gracefully
degrades when wandb is unavailable (e.g. during unit tests or offline runs).

Design rules
------------
- Never call wandb.init() more than once per process; guard with is_initialized().
- All scalar logging goes through log_scalar_dict() so tests can mock a single
  point of entry.
- Artifact upload is gated behind a guard so dry-run scripts never push data.
- The config path defaults to configs/wandb_config.yaml; callers may override.

Usage
-----
    from evaluation_metrics.tracking_utils import WandbTracker, log_scalar_dict

    tracker = WandbTracker(config_path="configs/wandb_config.yaml", run_name="review1")
    tracker.init(tags=["review1", "baseline"])
    log_scalar_dict(tracker, {"attack_success_rate": 0.0, "benign_task_success_rate": 1.0})
    tracker.upload_artifact("reports/review1_baseline_metrics.json", artifact_type="metrics")
    tracker.finish()
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Attempt to import wandb; fall back gracefully so tests never hard-fail
# ---------------------------------------------------------------------------
try:
    import wandb as _wandb  # type: ignore[import-untyped]
    _WANDB_AVAILABLE = True
except ImportError:  # pragma: no cover
    _wandb = None  # type: ignore[assignment]
    _WANDB_AVAILABLE = False

try:
    import yaml  # type: ignore[import-untyped]
    _YAML_AVAILABLE = True
except ImportError:  # pragma: no cover
    _YAML_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_wandb_config(config_path: str | Path) -> Dict[str, Any]:
    """Load the wandb section of configs/wandb_config.yaml.

    Returns an empty dict if the file is missing or yaml is not installed
    so callers are never blocked.
    """
    path = Path(config_path)
    if not path.exists():
        logger.warning("wandb_config.yaml not found at %s — using defaults.", path)
        return {}
    if not _YAML_AVAILABLE:
        logger.warning("PyYAML not installed — cannot parse wandb_config.yaml.")
        return {}
    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return raw or {}


def log_scalar_dict(
    tracker: "WandbTracker",
    metrics: Dict[str, float],
    step: Optional[int] = None,
) -> None:
    """Log a flat dict of scalar metrics through *tracker*.

    No-ops if the tracker is not active (mode=disabled, wandb unavailable,
    or tracker not yet initialized).

    Parameters
    ----------
    tracker:
        A WandbTracker instance.
    metrics:
        Flat dict of ``{metric_name: float_value}``.
    step:
        Optional global step counter; omit to let wandb auto-increment.
    """
    if not tracker.active:
        return
    try:
        if step is not None:
            _wandb.log(metrics, step=step)
        else:
            _wandb.log(metrics)
    except Exception as exc:  # pragma: no cover
        logger.warning("wandb.log failed: %s", exc)


# ---------------------------------------------------------------------------
# WandbTracker
# ---------------------------------------------------------------------------

class WandbTracker:
    """Thin wrapper around wandb that enforces team-wide tracking conventions.

    The tracker is deliberately **not** a context manager so that it survives
    across multiple helper functions inside a single script.  Call ``finish()``
    explicitly at the end of each script.

    Parameters
    ----------
    config_path:
        Path to ``configs/wandb_config.yaml``.  Resolved relative to the
        project root if a relative path is given.
    run_name:
        Human-readable run name (e.g. ``"review1_baseline_seed42"``).
        Overrides the ``name`` field from the YAML config.
    mode:
        ``"online"``, ``"offline"``, or ``"disabled"``.  Overrides the YAML
        ``init.mode`` field and the ``WANDB_MODE`` environment variable.
    """

    def __init__(
        self,
        config_path: str | Path = "configs/wandb_config.yaml",
        run_name: Optional[str] = None,
        mode: Optional[str] = None,
    ) -> None:
        self._config_path = Path(config_path)
        self._run_name = run_name
        self._mode_override = mode
        self._run = None
        self._active = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def active(self) -> bool:
        """True if wandb is available and the run was successfully initialized."""
        return self._active

    @property
    def run(self):
        """The underlying wandb.Run object, or None."""
        return self._run

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def init(
        self,
        tags: Optional[List[str]] = None,
        extra_config: Optional[Dict[str, Any]] = None,
    ) -> "WandbTracker":
        """Initialize a wandb run.

        Safe to call even when wandb is not installed (logs a warning and
        sets ``active = False``).

        Parameters
        ----------
        tags:
            Additional per-run tags merged with the base tags from the YAML.
        extra_config:
            Additional hyperparameter dict logged to the W&B run config.

        Returns
        -------
        self — allows ``tracker = WandbTracker(...).init()``.
        """
        if not _WANDB_AVAILABLE:
            logger.warning(
                "wandb not installed — tracking disabled. "
                "Install with: pip install wandb"
            )
            return self

        raw = _load_wandb_config(self._config_path)
        init_kwargs: Dict[str, Any] = dict(raw.get("init", {}))

        # Apply overrides
        if self._run_name is not None:
            init_kwargs["name"] = self._run_name

        effective_mode = (
            self._mode_override
            or os.environ.get("WANDB_MODE")
            or init_kwargs.get("mode", "online")
        )
        init_kwargs["mode"] = effective_mode

        # Merge tags
        base_tags: List[str] = list(init_kwargs.get("tags", []))
        if tags:
            base_tags.extend(t for t in tags if t not in base_tags)
        init_kwargs["tags"] = base_tags

        # Merge extra config (logged as run-level hyperparams)
        if extra_config:
            init_kwargs["config"] = {**init_kwargs.get("config", {}), **extra_config}

        try:
            # Guard against double-init in the same process
            if _wandb.run is not None:
                logger.warning(
                    "wandb.init() called but a run is already active (%s). "
                    "Reusing existing run.",
                    _wandb.run.name,
                )
                self._run = _wandb.run
            else:
                self._run = _wandb.init(**init_kwargs)
            self._active = effective_mode != "disabled"
        except Exception as exc:  # pragma: no cover
            logger.error("wandb.init() failed: %s — tracking disabled.", exc)
            self._active = False

        return self

    def log(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log a dict of scalar metrics.  No-op if not active."""
        log_scalar_dict(self, metrics, step=step)

    def upload_artifact(
        self,
        path: str | Path,
        artifact_type: str = "metrics",
        name: Optional[str] = None,
    ) -> None:
        """Upload a local file as a W&B artifact.

        Parameters
        ----------
        path:
            Local path to the file to upload.
        artifact_type:
            W&B artifact type tag (e.g. ``"metrics"``, ``"dataset"``).
        name:
            Artifact name; defaults to the file stem.
        """
        if not self._active:
            return
        path = Path(path)
        if not path.exists():
            logger.warning("Artifact path does not exist: %s", path)
            return
        artifact_name = name or path.stem
        try:
            artifact = _wandb.Artifact(artifact_name, type=artifact_type)
            artifact.add_file(str(path))
            _wandb.log_artifact(artifact)
            logger.info("Uploaded artifact: %s (%s)", artifact_name, artifact_type)
        except Exception as exc:  # pragma: no cover
            logger.warning("Artifact upload failed for %s: %s", path, exc)

    def log_table(
        self,
        table_name: str,
        columns: List[str],
        rows: List[List[Any]],
    ) -> None:
        """Log a W&B Table (used for leak detection results).

        Parameters
        ----------
        table_name:
            Key under which the table appears in the W&B run UI.
        columns:
            Column headers.
        rows:
            List of row lists; each inner list must match *columns* length.
        """
        if not self._active:
            return
        try:
            table = _wandb.Table(columns=columns, data=rows)
            _wandb.log({table_name: table})
        except Exception as exc:  # pragma: no cover
            logger.warning("wandb.Table log failed: %s", exc)

    def log_json_artifact(
        self,
        data: Dict[str, Any],
        artifact_name: str,
        artifact_type: str = "metrics",
        tmp_dir: Optional[str | Path] = None,
    ) -> None:
        """Serialize *data* to a temp JSON file and upload as an artifact.

        Useful for logging in-memory report dicts without needing a pre-written
        file on disk.
        """
        if not self._active:
            return
        import tempfile

        tmp_dir = Path(tmp_dir) if tmp_dir else Path(tempfile.gettempdir())
        tmp_path = tmp_dir / f"{artifact_name}.json"
        try:
            tmp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            self.upload_artifact(tmp_path, artifact_type=artifact_type, name=artifact_name)
        except Exception as exc:  # pragma: no cover
            logger.warning("log_json_artifact failed for %s: %s", artifact_name, exc)
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    def finish(self) -> None:
        """Finalize the current wandb run.  Safe to call if not active."""
        if self._active and _WANDB_AVAILABLE:
            try:
                _wandb.finish()
            except Exception as exc:  # pragma: no cover
                logger.warning("wandb.finish() failed: %s", exc)
            finally:
                self._active = False

    def __repr__(self) -> str:
        status = "active" if self._active else "inactive"
        run_name = self._run.name if self._run else "—"
        return f"WandbTracker(status={status}, run={run_name!r})"
