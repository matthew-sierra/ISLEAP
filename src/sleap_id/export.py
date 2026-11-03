"""Stage 8, export: identities -> a copy of the .slp whose tracks are the identities.

The output is the input .slp with the session's video re-tracked: its tracks
become "1".."n_animals" (the identities) and each instance's track is its
identity, or none when it has no identity. Every instance, point and score is
kept, and the video reference is unchanged. Instances on other videos of the
file keep their original tracks.

Written as <slp stem>.sleap_id_predictions.slp in the session folder, through a
temporary name and a rename, then reloaded and checked instance by instance.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import h5py
import numpy as np
import sleap_io as sio

from sleap_id import files

log = logging.getLogger("sleap_id")


class ExportError(RuntimeError):
    """The .slp does not match the session, or the written file failed its check."""


def _identities_on(labels, video, names) -> dict:
    """(frame_idx, position in frame) -> identity (0 = no track) for one video."""
    out = {}
    for lf in labels.labeled_frames:
        if lf.video is video:
            for i, inst in enumerate(lf.instances):
                t = inst.track
                out[(lf.frame_idx, i)] = int(t.name) if t is not None and t.name in names else 0
    return out


def write_slp(slp_in: Path, slp_out: Path, video_index: int, frame_idx, instance_idx,
              identity, n_animals: int) -> dict:
    """Copy slp_in with identities as tracks on video `video_index`; verify the copy."""
    labels = sio.load_slp(str(slp_in), open_videos=False)
    video = labels.videos[video_index]
    tracks = [sio.Track(name=str(k)) for k in range(1, n_animals + 1)]
    ident = {(int(f), int(i)): int(k) for f, i, k in zip(frame_idx, instance_idx, identity)}
    other_tracks, n_set, n_absent = [], 0, 0
    for lf in labels.labeled_frames:
        if lf.video is not video:
            other_tracks += [inst.track for inst in lf.instances if inst.track is not None]
            continue
        for i, inst in enumerate(lf.instances):
            k = ident.get((lf.frame_idx, i))
            if k is None:                    # not in the session (instance_source)
                n_absent += 1
                k = 0
            inst.track = tracks[k - 1] if k > 0 else None
            n_set += k > 0
    kept = []
    for t in other_tracks:
        if all(t is not u for u in kept):
            kept.append(t)
    labels.tracks = tracks + kept

    slp_out = Path(slp_out)
    tmp = slp_out.with_name(slp_out.stem + ".partial.slp")
    sio.save_slp(labels, str(tmp), verbose=False)
    reloaded = sio.load_slp(str(tmp), open_videos=False)
    got = _identities_on(reloaded, reloaded.videos[video_index], {t.name for t in tracks})
    want = {key: ident.get(key, 0) for key in got}
    if got != want or len(got) != sum(len(lf.instances) for lf in labels.labeled_frames
                                      if lf.video is video):
        tmp.unlink(missing_ok=True)
        raise ExportError(f"{tmp.name}: identities differ after reloading")
    os.replace(tmp, slp_out)
    return dict(path=str(slp_out), instances_with_identity=int(n_set),
                instances_not_in_session=int(n_absent))


def run(session: Path, slp_file: Path, identities: Path | None = None,
        output: Path | None = None) -> dict:
    session = Path(session)
    id_path = Path(identities) if identities else session / "identities.h5"
    with h5py.File(session / "selection.h5", "r") as f:
        g = f["instances"]
        frame, inst = g["frame_idx"][:], g["instance_idx"][:]
        video_index, slp_fp = int(f.attrs["video_index"]), json.loads(f.attrs["slp"])
        cache_id = f.attrs["cache_id"]
    fp = files.fingerprint(slp_file)
    if any(fp[k] != slp_fp[k] for k in ("size", "sha256_head")):
        raise ExportError(f"{slp_file} is not the .slp this session was prepared from")
    with h5py.File(id_path, "r") as f:
        if f.attrs["cache_id"] != cache_id:
            raise ExportError(f"{id_path} was made for another session (cache_id differs)")
        identity = f["instances/identity"][:]
        n_animals = int(f.attrs["n_animals"])
    out = Path(output) if output else session / f"{Path(slp_file).stem}.sleap_id_predictions.slp"
    res = write_slp(slp_file, out, video_index, frame, inst, identity, n_animals)
    log.info(f"wrote {out}: {res['instances_with_identity']:,} of {len(identity):,} "
             f"instances have an identity")
    return res
