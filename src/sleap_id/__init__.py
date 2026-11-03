"""sleap-id: identity assignment for SLEAP tracks."""

__version__ = "0.1.0"

# Version of the session files this package writes (selection.h5, crops.h5).
# Bumped whenever their layout changes, so a reader can refuse old sessions.
FORMAT_VERSION = 1


def run(*args, **kwargs):
    """The whole pipeline in one call; see sleap_id.pipeline.run."""
    from sleap_id.pipeline import run as _run  # imported here: torch is heavy
    return _run(*args, **kwargs)
