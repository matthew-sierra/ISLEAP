"""The roles YAML file: what it may contain, its defaults, and its validation.

Rule for what goes here versus on the command line: anything that changes the
RESULT (which crops exist, what they look like, which are selected) lives in
this file, so a session can be reproduced from it and it can be stamped into the
output. Anything about WHERE or HOW the run happens (paths, cores, cache size)
is a command-line option.

The one exception is the `training:` section, which also holds the checkpoint to
resume from. `prepare` checks it but does not need it, and it is left out of the
crops' cache_id, so editing it never makes existing crops stale.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

CONFIG_VERSION = 1

# Every setting the file may contain, with its default. None marks a required
# setting with no sensible default (the roles depend on the skeleton).
DEFAULTS: dict[str, Any] = {
    "config_version": CONFIG_VERSION,
    "n_animals": None,
    # Keypoint roles, exactly v1's config.py names (lower-cased).
    "centroid_node": None,
    "top_node": None,
    "top_node_fallback": None,
    "bottom_node": None,
    "surrounding_keypoints": None,
    "bounded_keypoints": [],
    # Which SLEAP instances to use. "both" is v1's behaviour (every instance).
    "instance_source": "both",
    "color_mode": "grayscale",
    # Crop shape (v1: MAX_CROP_AREA, SIZE_STAT, MIN_HULL_KEYPOINTS).
    "max_crop_area": 6400,
    "size_stat": "median",
    "min_hull_keypoints": 3,
    # Training sample.
    "min_fragment_length": 4,
    "crops_per_fragment": None,
    "seed": 0,
    # Training (read by `train`; `prepare` only checks it). See TRAINING_DEFAULTS.
    "training": None,
}

# The `training:` section. None marks a setting `train` requires.
TRAINING_DEFAULTS: dict[str, Any] = {
    # Checkpoint of an earlier run to continue from (its latest.pt); null = start fresh.
    "resume_from": None,
    # Stop once the training loss has reached this level...
    "target_loss": None,
    # ...and has not improved for this many batches since (checked every 50
    # batches, so the default is 10 checks).
    "patience_batches": 500,
    # Halve the learning rate when the loss has not improved for this many
    # batches (whether or not target_loss has been reached).
    "lr_patience_batches": 250,
}
TRAINING_REQUIRED = ("target_loss",)

REQUIRED = ("n_animals", "centroid_node", "top_node", "bottom_node",
            "surrounding_keypoints")
SINGLE_ROLES = ("centroid_node", "top_node", "top_node_fallback", "bottom_node")
LIST_ROLES = ("surrounding_keypoints", "bounded_keypoints")
CHOICES = {
    "instance_source": ("predicted", "user", "both"),
    "color_mode": ("grayscale", "rgb"),
    "size_stat": ("median", "mean"),
}


class ConfigError(ValueError):
    """The roles file is missing, malformed, or does not fit the skeleton."""


def load(path: Path, node_names: list[str]) -> dict[str, Any]:
    """Read `path`, fill in defaults, and validate against the skeleton.

    Returns the RESOLVED config: every setting present, defaults included, so
    the copy stamped into the session records exactly what produced it.
    """
    try:
        raw = yaml.safe_load(Path(path).read_text()) or {}
    except yaml.YAMLError as e:
        raise ConfigError(f"{path} is not valid YAML: {e}") from e
    if not isinstance(raw, dict):
        raise ConfigError(f"{path} must be a mapping of setting: value")
    return resolve(raw, node_names, source=str(path))


def resolve(raw: dict, node_names: list[str], source: str = "config") -> dict:
    unknown = sorted(set(raw) - set(DEFAULTS))
    if unknown:
        raise ConfigError(f"{source}: unknown setting(s) {unknown}. "
                          f"Allowed: {sorted(DEFAULTS)}")
    cfg = copy.deepcopy(DEFAULTS)
    cfg.update(raw)
    errors = _training(cfg)
    errors += _validate(cfg, node_names)
    if errors:
        raise ConfigError(f"{source} has {len(errors)} problem(s):\n  - "
                          + "\n  - ".join(errors))
    if cfg["bounded_keypoints"] is None:
        cfg["bounded_keypoints"] = []
    return cfg


def require_training(cfg: dict, source: str = "config") -> dict:
    """The training section, with every setting `train` needs filled in."""
    tr = cfg["training"]
    missing = [k for k in TRAINING_REQUIRED if tr[k] is None]
    if missing:
        raise ConfigError(f"{source}: training setting(s) not filled in: {missing}")
    return tr


def _training(cfg: dict) -> list[str]:
    """Fill the training section's defaults in place; return its problems."""
    raw = cfg["training"]
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        cfg["training"] = copy.deepcopy(TRAINING_DEFAULTS)
        return ["training must be a section of setting: value lines"]
    errs = []
    unknown = sorted(set(raw) - set(TRAINING_DEFAULTS))
    if unknown:
        errs.append(f"training: unknown setting(s) {unknown}. "
                    f"Allowed: {sorted(TRAINING_DEFAULTS)}")
    tr = copy.deepcopy(TRAINING_DEFAULTS)
    tr.update({k: v for k, v in raw.items() if k in TRAINING_DEFAULTS})
    cfg["training"] = tr
    if tr["resume_from"] is not None and not isinstance(tr["resume_from"], str):
        errs.append(f"training.resume_from must be a path or null, got {tr['resume_from']!r}")
    t = tr["target_loss"]
    if t is not None and (not isinstance(t, (int, float)) or isinstance(t, bool) or t < 0):
        errs.append(f"training.target_loss must be a number >= 0, got {t!r}")
    for k in ("patience_batches", "lr_patience_batches"):
        p = tr[k]
        if not isinstance(p, int) or isinstance(p, bool) or p < 1:
            errs.append(f"training.{k} must be a positive integer, got {p!r}")
    return errs


def _validate(cfg: dict, node_names: list[str]) -> list[str]:
    errs: list[str] = []
    if cfg["config_version"] != CONFIG_VERSION:
        errs.append(f"config_version is {cfg['config_version']!r}; this version "
                    f"of sleap-id reads config_version {CONFIG_VERSION}")

    missing = [k for k in REQUIRED if cfg[k] in (None, [], "")]
    if missing:
        errs.append(f"required setting(s) not filled in: {missing}")

    valid = f"valid nodes: {', '.join(node_names)}"
    for k in SINGLE_ROLES:
        v = cfg[k]
        if v is not None and v not in node_names:
            errs.append(f"{k}: {v!r} is not a node in this skeleton ({valid})")
    for k in LIST_ROLES:
        v = cfg[k]
        if v is None:
            continue
        if not isinstance(v, list):
            errs.append(f"{k} must be a list of node names")
            continue
        bad = [n for n in v if n not in node_names]
        if bad:
            errs.append(f"{k}: {bad} not in this skeleton ({valid})")
        dup = sorted({n for n in v if v.count(n) > 1})
        if dup:
            errs.append(f"{k}: listed more than once: {dup}")

    sur, bnd = cfg["surrounding_keypoints"] or [], cfg["bounded_keypoints"] or []
    both = sorted(set(sur) & set(bnd))
    if both:
        errs.append(f"a keypoint cannot be both surrounding and bounded: {both}")
    if cfg["top_node"] is not None and cfg["top_node"] == cfg["centroid_node"]:
        errs.append("top_node and centroid_node must differ (the crop is rotated "
                    "so the top node sits above the centroid)")
    if cfg["top_node_fallback"] is not None and cfg["top_node_fallback"] == cfg["top_node"]:
        errs.append("top_node_fallback is the same node as top_node; a fallback "
                    "to itself can never help")

    for k, allowed in CHOICES.items():
        if cfg[k] not in allowed:
            errs.append(f"{k} must be one of {list(allowed)}, got {cfg[k]!r}")

    def positive_int(k, allow_none=False):
        v = cfg[k]
        if v is None and allow_none:
            return
        if not isinstance(v, int) or isinstance(v, bool) or v < 1:
            errs.append(f"{k} must be a positive integer"
                        + (" or null" if allow_none else "") + f", got {v!r}")

    if cfg["n_animals"] is not None:
        positive_int("n_animals")
    positive_int("max_crop_area", allow_none=True)
    positive_int("min_fragment_length")
    positive_int("crops_per_fragment", allow_none=True)
    if not isinstance(cfg["min_hull_keypoints"], int) or cfg["min_hull_keypoints"] < 3:
        errs.append("min_hull_keypoints must be an integer >= 3 (a polygon needs "
                    f"three corners), got {cfg['min_hull_keypoints']!r}")
    elif sur and len(sur) < cfg["min_hull_keypoints"]:
        errs.append(f"surrounding_keypoints has {len(sur)} node(s); the outline needs "
                    f"at least min_hull_keypoints={cfg['min_hull_keypoints']}")
    if not isinstance(cfg["seed"], int) or isinstance(cfg["seed"], bool):
        errs.append(f"seed must be an integer, got {cfg['seed']!r}")
    return errs


def template(node_names: list[str], slp_name: str) -> str:
    """Starter roles file for this skeleton: every setting, roles left blank."""
    nodes = "\n".join(f"#   - {n}" for n in node_names)
    return f"""\
# sleap-id roles file, generated from {slp_name}
# Fill in the settings marked REQUIRED, then run `sleap-id run` (everything)
# or `sleap-id prepare` (crops only; the training section is not needed for it).
#
# Nodes in this skeleton:
{nodes}

config_version: {CONFIG_VERSION}

n_animals:              # REQUIRED: number of animals in the video

# --- Keypoint roles --------------------------------------------------------
centroid_node:          # REQUIRED: the crop is rotated about this node
top_node:               # REQUIRED: rotated to sit directly above centroid_node
top_node_fallback:      # optional: used only when top_node is missing
bottom_node:            # REQUIRED: sets how high the centroid sits in the crop
surrounding_keypoints:  # REQUIRED: nodes ON the body outline (>= 3)
  # - NodeName
bounded_keypoints: []   # nodes INSIDE the outline; used only when a
                        # surrounding node is missing

# --- Inputs ----------------------------------------------------------------
instance_source: both   # predicted | user | both
color_mode: grayscale   # grayscale | rgb

# --- Crop shape ------------------------------------------------------------
max_crop_area: 6400     # pixels; larger crops are scaled down (null = no limit)
size_stat: median       # median | mean of body extent sets the crop size
min_hull_keypoints: 3   # fewer usable outline nodes -> no crop

# --- Training sample -------------------------------------------------------
min_fragment_length: 4  # shorter track runs are not trained on
crops_per_fragment:     # null = every frame; N = up to N evenly spaced per fragment
seed: 0                 # for the random steps of training

# --- Training --------------------------------------------------------------
# The loss is checked every 50 batches (lower is better). Training stops, and
# saves its final weights, once the loss has reached target_loss AND has not
# improved for patience_batches batches since. Whenever the loss has not
# improved for lr_patience_batches batches, the learning rate is halved.
training:
  resume_from:            # optional: checkpoint of an earlier run to start from,
                          # e.g. old_run/checkpoints/latest.pt (null = start fresh)
  target_loss:            # REQUIRED for training: loss level to reach
  patience_batches: 500   # batches without improvement before stopping
  lr_patience_batches: 250  # batches without improvement before halving the learning rate
"""
