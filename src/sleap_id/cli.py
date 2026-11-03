"""Terminal commands: `sleap-id ...` (later `sleap id ...`).

`sleap-id run` does everything in one call (preprocess, train, predict, assign,
export). The group is chained, so several subcommands can run in one call:

    sleap-id init-config --slp_file preds.slp --config roles.yaml \\
             run --frames 0:108000

A later command reuses --slp_file / --config from an earlier one in the same
call when it doesn't pass its own.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click

from sleap_id import __version__

CONTEXT = {"help_option_names": ["-h", "--help"]}
log = logging.getLogger("sleap_id")

existing_file = click.Path(exists=True, dir_okay=False, path_type=Path)


def _setup_logging(verbose: bool, quiet: bool) -> None:
    log.setLevel(logging.DEBUG)
    if not any(isinstance(h, logging.StreamHandler) and h.stream is sys.stderr
               for h in log.handlers):
        h = logging.StreamHandler(sys.stderr)
        h.setFormatter(logging.Formatter("%(message)s"))
        log.addHandler(h)
    level = logging.DEBUG if verbose else logging.WARNING if quiet else logging.INFO
    for h in log.handlers:
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
            h.setLevel(level)


def _from_chain(ctx: click.Context, name: str, value, flag: str):
    """Use this command's own value, else one given earlier in the chain."""
    if value is not None:
        ctx.obj[name] = value
        return value
    if name in ctx.obj:
        return ctx.obj[name]
    raise click.UsageError(f"missing {flag} (give it here or to an earlier command)")


@click.group(chain=True, context_settings=CONTEXT)
@click.version_option(__version__, prog_name="sleap-id")
@click.pass_context
def cli(ctx: click.Context) -> None:
    """Identity assignment for SLEAP tracks (idtracker.ai's contrastive method)."""
    ctx.ensure_object(dict)


@cli.command("init-config", context_settings=CONTEXT)
@click.option("--slp_file", "-i", type=existing_file, default=None,
              help="SLEAP predictions file whose skeleton the config describes.")
@click.option("--config", "config_path", type=click.Path(dir_okay=False, path_type=Path),
              default=None, help="Roles YAML to create.")
@click.option("--overwrite", is_flag=True, help="Replace the config if it exists.")
@click.pass_context
def init_config(ctx, slp_file, config_path, overwrite):
    """Write a starter roles YAML listing this skeleton's nodes.

    If the file already exists it is kept as it is (pass --overwrite to replace
    it), so a chained `init-config ... prepare ...` re-runs safely.
    """
    from sleap_id import config, slp

    slp_file = _from_chain(ctx, "slp_file", slp_file, "--slp_file")
    config_path = _from_chain(ctx, "config", config_path, "--config")
    if config_path.exists() and not overwrite:
        click.echo(f"{config_path} exists; keeping it (--overwrite to replace)", err=True)
        return
    try:
        names = slp.read_node_names(slp_file)
    except Exception as e:
        raise click.ClickException(str(e))
    config_path.write_text(config.template(names, slp_file.name))
    click.echo(f"wrote {config_path}: fill in the REQUIRED settings, then run "
               "`sleap-id run`", err=True)


@cli.command("prepare", context_settings=CONTEXT)
@click.option("--config", "config_path", type=existing_file, default=None,
              help="Roles YAML (see `sleap-id init-config`).")
@click.option("--slp_file", "-i", type=existing_file, default=None,
              help="SLEAP predictions file (.slp) with tracks.")
@click.option("--video", type=existing_file, default=None,
              help="Video file; overrides the path stored in the .slp.")
@click.option("--video_index", type=int, default=0, show_default=True,
              help="Which video of the .slp to process.")
@click.option("--output_path", "-o", type=click.Path(file_okay=False, path_type=Path),
              default=None, help="Session directory [default: <slp stem>.sleap_id/].")
@click.option("--overwrite", is_flag=True, help="Replace an existing session.")
@click.option("--workers", default="auto", show_default=True,
              help="Decode processes, or 'auto'.")
@click.option("--cache_gb", type=float, default=2.0, show_default=True,
              help="Disk budget for the training-crop cache.")
@click.option("--frames", "frames_spec", default=None, metavar="START:END",
              help="Frame range, END exclusive (e.g. 0:1000; '500:' = to the end).")
@click.option("--dry_run", is_flag=True,
              help="Check everything and print the plan; write nothing.")
@click.option("--verbose", "-v", is_flag=True, help="More output.")
@click.option("--quiet", "-q", is_flag=True, help="Only warnings and errors.")
@click.pass_context
def prepare_cmd(ctx, config_path, slp_file, video, video_index, output_path, overwrite,
                workers, cache_gb, frames_spec, dry_run, verbose, quiet):
    """Build the training-crop cache: selection.h5 + crops.h5."""
    from sleap_id import prepare
    from sleap_id.config import ConfigError
    from sleap_id.geometry import CropGeometryError
    from sleap_id.slp import SlpError
    from sleap_id.stream import StreamError
    from sleap_id.video import VideoError

    _setup_logging(verbose, quiet)
    slp_file = _from_chain(ctx, "slp_file", slp_file, "--slp_file")
    config_path = _from_chain(ctx, "config", config_path, "--config")
    if not config_path.exists():
        raise click.UsageError(f"--config {config_path} does not exist")
    try:
        prepare.run(slp_file, config_path, video, video_index, output_path, overwrite,
                    workers, cache_gb, frames_spec, dry_run)
    except (prepare.PrepareError, ConfigError, CropGeometryError, SlpError,
            StreamError, VideoError, ValueError) as e:
        log.debug("details", exc_info=True)
        raise click.ClickException(str(e))  # exit code 1, one clean message


@cli.command("run", context_settings=CONTEXT)
@click.option("--config", "config_path", type=existing_file, default=None,
              help="Roles YAML, with its training section filled in.")
@click.option("--slp_file", "-i", type=existing_file, default=None,
              help="SLEAP predictions file (.slp) with tracks.")
@click.option("--video", type=existing_file, default=None,
              help="Video file; overrides the path stored in the .slp.")
@click.option("--video_index", type=int, default=0, show_default=True,
              help="Which video of the .slp to process.")
@click.option("--output_path", "-o", type=click.Path(file_okay=False, path_type=Path),
              default=None, help="Session directory [default: <slp stem>.sleap_id/].")
@click.option("--overwrite", is_flag=True,
              help="Replace a session that was made from different inputs or settings.")
@click.option("--workers", default="auto", show_default=True,
              help="Decode processes, or 'auto'.")
@click.option("--cache_gb", type=float, default=2.0, show_default=True,
              help="Disk budget for the training-crop cache.")
@click.option("--frames", "frames_spec", default=None, metavar="START:END",
              help="Frame range, END exclusive (e.g. 0:1000; '500:' = to the end).")
@click.option("--verbose", "-v", is_flag=True, help="More output.")
@click.option("--quiet", "-q", is_flag=True, help="Only warnings and errors.")
@click.pass_context
def run_cmd(ctx, config_path, slp_file, video, video_index, output_path, overwrite,
            workers, cache_gb, frames_spec, verbose, quiet):
    """Everything: preprocess, train, predict, assign identities, export a .slp.

    Run the same command again to continue an unfinished run: finished stages
    are skipped and training resumes from its last checkpoint.
    """
    from sleap_id import pipeline, prepare
    from sleap_id.config import ConfigError
    from sleap_id.export import ExportError
    from sleap_id.geometry import CropGeometryError
    from sleap_id.predict import PredictError
    from sleap_id.slp import SlpError
    from sleap_id.stream import StreamError
    from sleap_id.train import TrainError
    from sleap_id.video import VideoError

    _setup_logging(verbose, quiet)
    slp_file = _from_chain(ctx, "slp_file", slp_file, "--slp_file")
    config_path = _from_chain(ctx, "config", config_path, "--config")
    if not config_path.exists():
        raise click.UsageError(f"--config {config_path} does not exist")
    try:
        pipeline.run(slp_file, config_path, video, video_index, output_path, overwrite,
                     workers, cache_gb, frames_spec)
    except (pipeline.RunError, prepare.PrepareError, ConfigError, CropGeometryError,
            SlpError, StreamError, VideoError, TrainError, PredictError, ExportError,
            ValueError) as e:
        log.debug("details", exc_info=True)
        raise click.ClickException(str(e))
