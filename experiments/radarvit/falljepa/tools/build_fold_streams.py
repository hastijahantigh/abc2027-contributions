"""
Build the streams of the subject-held-out folds (configs/folds/subject_folds.yaml) from `segments/`,
with the organizers' own generator (`src/abc2027/streams.py`, `scripts/build_streams.py` of the
release), so every stream is stitched and labelled exactly as the official ones.

    uv run python tools/build_fold_streams.py --release /path/to/abc2027_uwb_train_v1 --check-official
    uv run python tools/build_fold_streams.py --release ... --out DATA/folds --dry-run
    uv run python tools/build_fold_streams.py --release ... --out DATA/folds [--folds A B] [--train-stitchings 2]

Each fold becomes a release-shaped folder that `--data.release` reads unchanged:

    <out>/segments.h5                           segments/ once, in HDF5: the source of every virtual stream
    <out>/<fold>/streams/train/stream_XX.h5     K stitchings of the fold's training clips (ADL used 3 times)
    <out>/<fold>/streams/val/stream_XX.h5       2 stitchings of its held-out clips: 10 streams, ~4.2 h (ADL used 2 times, as test)
    <out>/<fold>/streams_events.csv, streams_composition.csv, groups.csv (subject per source), fold.json

Storage. A stream is all copies of segment frames except the 0.2 s crossfades, so by default
(`--storage virtual`) its `cir` is an HDF5 virtual dataset: runs of segments.h5 plus the crossfaded
frames, the only samples the stream file stores (a few MB). h5py reads it like the official files, so
the data module is unchanged; the whole build costs segments.h5 (4.9 GB) instead of ~0.8 GB per
stream hour. Keep <out> together (sources are found relative to the stream files); a missing source
reads as NaN, which the loader rejects. Before building, the official train and val streams are
rebuilt virtually and compared with the official files bit for bit. `--storage full` writes ordinary
files with the organizers' writer.

A stitching uses every clip of its set once (falls with fresh margins, ADL cut at fresh points), at
the official 57.19 falls per hour, packed into 30-minute streams; stitchings differ only by seed.
For the official fold (D) the first stitching is the official streams themselves (symlinked), so
its held-out score includes exactly the official val.

The shipped generator does not reproduce the official streams from `segments/`: it takes a clip's
duration from its n frames as n / 50 s, while the official plan used shorter durations. With these
rules all 18 official streams rebuild bit for bit (verified 2026-10-22):

    ADL and Autre/Empty clips   floor((n - 1) / 50) whole seconds (the radar clocks are integer seconds)
    fall clips (t_last)         (n - 1) / 50 s, the last sample; Chute02_270 of Test_P3_8 one frame less

`--check-official` replans the official train and val with the public seed and compares them with
`streams_composition.csv` (reads only array shapes). After writing, every fall label is checked: the
labelled interval of the stream must be an exact copy of the fall in `segments/`.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import yaml

RATE_HZ = 50
FALL_T_LAST_FRAMES = {"falls/Dataset 3/Test_P3_8/Chute02_270.json": 774}
SEGMENTS_H5 = "segments.h5"
SPLIT_FOR = {"train": "train", "val": "val"}     # organizer split name -> ADL uses (3 train, 2 val = test)


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def organizer_modules(release: Path):
    """The release's own generator: streams.py (planning) and build_streams.py (writing). build_streams
    imports matplotlib only for its report figure, which is not drawn here: a stub stands in if it is absent."""
    sys.path.insert(0, str(release / "src"))
    sys.path.insert(0, str(release / "scripts"))
    stubbed = False
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        import types
        stub = types.ModuleType("matplotlib")
        stub.use = lambda *a, **k: None
        sys.modules["matplotlib"] = stub
        sys.modules["matplotlib.pyplot"] = types.ModuleType("matplotlib.pyplot")
        stubbed = True
    import build_streams
    if stubbed:                    # other imports (lightning) must still see matplotlib as absent
        del sys.modules["matplotlib"], sys.modules["matplotlib.pyplot"]
    from abc2027 import streams
    return streams, build_streams


def subject(kind: str, path: str) -> str:
    """'F3' for a fall-corpus recording of session suffix 3 (falls and Autre/Empty), 'A5' for an ADL file."""
    if kind == "adl":
        return "A" + re.search(r"_(\d+)\.json$", path).group(1)
    return "F" + re.search(r"Test_P\d+_(\d+)", path).group(1)


def segment_path(release: Path, path: str) -> Path:
    return release / "segments" / (path.replace("/", "__") + ".npy")


def ensure_segments_h5(release: Path, out: Path, paths=None) -> Path:
    """segments/*.npy as one HDF5 file, a contiguous float32 dataset per clip (named like the .npy).
    Written once; `paths` limits it to some clips (tests)."""
    target = out / SEGMENTS_H5
    if target.exists():
        return target
    out.mkdir(parents=True, exist_ok=True)
    npys = sorted((release / "segments").glob("*.npy")) if paths is None else [segment_path(release, p) for p in paths]
    part = target.with_name(SEGMENTS_H5 + ".part")
    with h5py.File(part, "w") as h:
        for npy in npys:
            h.create_dataset(npy.stem, data=np.load(npy))
    part.rename(target)
    print(f"wrote {target} ({target.stat().st_size / 1e9:.1f} GB, {len(npys)} clips)")
    return target


class _Piece:
    """Frames [a, b) of one clip, whose last frames may be replaced by a crossfade (`tail`)."""
    def __init__(self, key: str, frames: int, a: int, b: int):
        self.key, self.frames, self.a, self.b, self.tail = key, frames, a, b, None

    def last(self, seg_h5: h5py.File, k: int) -> np.ndarray:
        """Its current last k frames (crossfade included)."""
        m = max(k, 0 if self.tail is None else self.tail.shape[1])
        x = np.array(seg_h5[self.key][:, self.b - m:self.b], dtype=np.float32)
        if self.tail is not None:
            x[:, m - self.tail.shape[1]:] = self.tail
        return x[:, m - k:]

    def set_last(self, seg_h5: h5py.File, values: np.ndarray) -> None:
        k = values.shape[1]
        if self.tail is not None and self.tail.shape[1] > k:
            self.tail[:, -k:] = values
        else:
            self.tail = values


def write_virtual_stream(path: Path, split: str, index: int, segments, seg_h5_path: Path, bs):
    """The organizers' _write_stream (scripts/build_streams.py) with the same arithmetic and the same
    events and composition rows, but `cir` is a virtual dataset over segments.h5: only the crossfaded
    frames are stored, in this file's `crossfades`."""
    fade_n, step = bs.FADE_N, bs.STEP_S
    pieces: list[_Piece] = []
    events, composition = [], []
    cursor, last_event_end = 0, -1
    stream_id = f"{split}_{index:02d}"
    with h5py.File(seg_h5_path, "r") as seg_h5:
        for order, segment in enumerate(segments):
            key = segment.path.replace("/", "__")
            frames = seg_h5[key].shape[1]
            i0 = int(np.clip(np.round(segment.source_t0 / step), 0, frames - 1))
            i1 = int(np.clip(np.round(segment.source_t1 / step), i0 + 1, frames))
            length = i1 - i0
            blend = 0
            if pieces:
                protect = length
                if segment.event_t0 is not None:
                    protect = max(0, int(np.round(segment.event_t0 / step)) - i0)
                tail_room = cursor - last_event_end
                blend = min(fade_n, length, pieces[-1].b - pieces[-1].a, protect, max(tail_room, 0))
                if blend > 0:
                    weight = np.linspace(0.0, 1.0, blend, dtype=np.float32)[None, :, None]
                    head = pieces[-1].last(seg_h5, blend)
                    block = np.array(seg_h5[key][:, i0:i0 + blend], dtype=np.float32)
                    pieces[-1].set_last(seg_h5, head * (1.0 - weight) + block * weight)
            origin = cursor - blend
            if length > blend:
                pieces.append(_Piece(key, frames, i0 + blend, i1))
            cursor = origin + length
            composition.append({
                "stream_id": stream_id, "segment_id": f"{stream_id}_{order:04d}", "kind": segment.kind,
                "path": segment.path, "label": segment.label, "dataset": segment.dataset,
                "t_start": f"{origin * step:.3f}", "t_end": f"{cursor * step:.3f}",
                "source_t0": f"{segment.source_t0:.3f}", "source_t1": f"{segment.source_t1:.3f}",
            })
            if segment.kind == "fall":
                event_a = origin + int(np.round(segment.event_t0 / step)) - i0
                event_b = origin + int(np.round(segment.event_t1 / step)) - i0
                last_event_end = max(last_event_end, event_b)
                events.append({
                    "stream_id": stream_id, "event_id": f"{stream_id}_{order:04d}",
                    "t_start": f"{event_a * step:.3f}", "t_end": f"{event_b * step:.3f}",
                    "fall_type": segment.fall_type, "ignore": str(segment.ignore).lower(),
                    "quality": segment.quality, "needs_review": str(segment.needs_review).lower(),
                    "source_path": segment.path, "source_t_start": f"{segment.event_t0:.3f}",
                    "source_t_end": f"{segment.event_t1:.3f}", "dataset": segment.dataset,
                })
    total = sum(p.b - p.a for p in pieces)
    if total != cursor:
        raise AssertionError(f"{stream_id}: {total} frames mapped, {cursor} expected")

    path.parent.mkdir(parents=True, exist_ok=True)
    source = Path(os.path.relpath(seg_h5_path.resolve(), path.parent.resolve()))
    tails = [p.tail for p in pieces if p.tail is not None]
    with h5py.File(path, "w") as h:
        fades = np.concatenate(tails, axis=1) if tails else np.zeros((3, 1, 368), dtype=np.float32)
        h.create_dataset("crossfades", data=fades)
        layout = h5py.VirtualLayout(shape=(3, total, 368), dtype="float32")
        at = used = 0
        for p in pieces:
            n, t = p.b - p.a, 0 if p.tail is None else p.tail.shape[1]
            if n > t:
                layout[:, at:at + n - t, :] = h5py.VirtualSource(str(source), p.key, shape=(3, p.frames, 368))[
                    :, p.a:p.b - t, :]
            if t:
                layout[:, at + n - t:at + n, :] = h5py.VirtualSource(".", "crossfades", shape=fades.shape)[
                    :, used:used + t, :]
                used += t
            at += n
        h.create_virtual_dataset("cir", layout, fillvalue=np.nan)
        h.create_dataset("t", data=np.arange(total, dtype=np.float64) * step)
        h.create_dataset("radar_serials", data=np.array(bs.SERIALS, dtype=np.int32))
        h.attrs["split"] = split
        h.attrs["seed"] = bs.SEED
        h.attrs["stream_index"] = index
        h.attrs["fade_s"] = bs.FADE_S
    return events, composition


def check_virtual(release: Path, seg_h5_path: Path, scratch: Path, stream_ids=None) -> bool:
    """Rebuild official streams as virtual files from segments.h5 and compare them with the official
    files bit for bit (cir, t, radar_serials, attributes)."""
    streams, bs = organizer_modules(release)
    clips = load_clips(release)
    by_split = {s: groups_of([c for c in clips if c["official_split"] == s]) for s in ("train", "val")}
    _, plans = streams.plan_all(by_split)
    ok, checked = True, 0
    for split, plan in plans.items():
        for i, stream in enumerate(plan.streams):
            sid = f"{split}_{i:02d}"
            if stream_ids is not None and sid not in stream_ids:
                continue
            mine = scratch / "streams" / split / f"stream_{i:02d}.h5"
            write_virtual_stream(mine, split, i, stream.segments, seg_h5_path, bs)
            with h5py.File(mine, "r") as a, h5py.File(release / "streams" / split / f"stream_{i:02d}.h5", "r") as b:
                same = (a["cir"].shape == b["cir"].shape and np.array_equal(a["t"][:], b["t"][:])
                        and np.array_equal(a["radar_serials"][:], b["radar_serials"][:])
                        and all(np.array_equal(a.attrs[k], b.attrs[k]) for k in b.attrs)
                        and all(np.array_equal(a["cir"][:, s:s + 20000], b["cir"][:, s:s + 20000])
                                for s in range(0, b["cir"].shape[1], 20000)))
            mine.unlink()
            checked += 1
            ok &= same
            if not same:
                print(f"  virtual {sid} differs from the official file")
    print(f"virtual rebuild of {checked} official streams: {'bit for bit identical' if ok else 'MISMATCH'}")
    shutil.rmtree(scratch, ignore_errors=True)
    return ok


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def stream_sha256(path: Path, step: int = 20000) -> str:
    """sha256 of a stream's content, independent of how it is stored (virtual, ordinary or symlinked):
    cir (float32, C order, read in time blocks), then t, then radar_serials."""
    h = hashlib.sha256()
    with h5py.File(path, "r") as f:
        cir = f["cir"]
        h.update(repr((cir.shape, str(cir.dtype))).encode())
        for a in range(0, cir.shape[1], step):
            h.update(np.ascontiguousarray(cir[:, a:a + step, :], dtype=np.float32).tobytes())
        h.update(np.ascontiguousarray(f["t"][:], dtype=np.float64).tobytes())
        h.update(np.ascontiguousarray(f["radar_serials"][:], dtype=np.int32).tobytes())
    return h.hexdigest()


def fold_checksums(out: Path, names) -> list[dict]:
    """One row per stream and per label/metadata file of every fold: what must be equal for two builds
    to be the same data."""
    rows = []
    for name in names:
        folder = out / name
        for path in sorted(folder.glob("streams/*/stream_*.h5")):
            rows.append({"fold": name, "file": path.relative_to(folder).as_posix(), "sha256": stream_sha256(path)})
        for file in ("streams_events.csv", "streams_composition.csv", "groups.csv"):
            rows.append({"fold": name, "file": file, "sha256": sha256_file(folder / file)})
        print(f"[{name}] checksums of {sum(r['fold'] == name for r in rows)} files")
    return rows


def compare_checksums(rows: list[dict], reference: Path) -> bool:
    ref = {(r["fold"], r["file"]): r["sha256"] for r in read_csv(reference)}
    got = {(r["fold"], r["file"]): r["sha256"] for r in rows}
    folds = {f for f, _ in got}
    ref = {k: v for k, v in ref.items() if k[0] in folds}
    missing, extra = sorted(set(ref) - set(got)), sorted(set(got) - set(ref))
    differ = sorted(k for k in set(ref) & set(got) if ref[k] != got[k])
    for label, keys in (("missing", missing), ("not in the reference", extra), ("different", differ)):
        for fold, file in keys[:20]:
            print(f"  {label}: {fold}/{file}")
    ok = not (missing or extra or differ)
    print(f"checksums vs {reference}: {len(got)} files, "
          + ("all identical" if ok else f"{len(missing)} missing, {len(extra)} extra, {len(differ)} different"))
    return ok


def load_clips(release: Path) -> list[dict]:
    """Every source clip of train and val with the duration the official plan used, in the row format of
    build_streams._groups(): falls are event rows (t_start/t_end in source time, t_last), ADL and others
    carry duration_s. Each row gets its kind, official split and subject."""
    events = {e["source_path"]: e for e in read_csv(release / "streams_events.csv")}
    clips, seen = [], set()
    for row in read_csv(release / "streams_composition.csv"):
        path = row["path"]
        if path in seen:
            continue
        seen.add(path)
        n = np.load(segment_path(release, path), mmap_mode="r").shape[1]
        split = row["stream_id"].split("_")[0]
        if row["kind"] == "fall":
            e = dict(events[path])
            frames = FALL_T_LAST_FRAMES.get(path, n - 1)
            clip = {**e, "t_start": e["source_t_start"], "t_end": e["source_t_end"], "t_last": frames / RATE_HZ,
                    "path": path}
        else:
            clip = {**row, "duration_s": math.floor((n - 1) / RATE_HZ + 1e-9)}
            if row["kind"] == "other":
                clip["position"] = row["dataset"].replace("Dataset ", "")
        clips.append({**clip, "kind": row["kind"], "official_split": split, "subject": subject(row["kind"], path)})
    return clips


def groups_of(clips: list[dict]) -> dict:
    out = {"falls": [], "adl": [], "others": []}
    for c in clips:
        out[{"fall": "falls", "adl": "adl", "other": "others"}[c["kind"]]].append(c)
    return out


def official_density(release: Path) -> float:
    text = (release / "configs" / "streams.yaml").read_text(encoding="utf-8")
    return float(re.search(r"target_falls_per_hour:\s*([\d.]+)", text).group(1))


def plan_stitching(streams, clips: list[dict], split: str, density: float, seed: list[int]):
    bundle = streams.prepare_split(split, **groups_of(clips), rng=np.random.default_rng(seed))
    return streams.finish_plan(bundle, density)


def plan_fingerprint(rows) -> str:
    """sha256 (first 16 hex) of a stitching's plan: stream id, kind, clip and cut of every segment, in order.
    Equal fingerprints on two machines mean the same streams and labels."""
    text = "\n".join(f"{sid}\t{kind}\t{path}\t{float(t0):.3f}\t{float(t1):.3f}" for sid, kind, path, t0, t1 in rows)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def check_official(release: Path) -> bool:
    """Replan official train and val with the public seed; compare segment by segment with the CSV."""
    streams, _ = organizer_modules(release)
    clips = load_clips(release)
    by_split = {s: groups_of([c for c in clips if c["official_split"] == s]) for s in ("train", "val")}
    density, plans = streams.plan_all(by_split)
    comp = read_csv(release / "streams_composition.csv")
    ok = abs(density - official_density(release)) < 1e-3
    print(f"density {density:.4f} (official {official_density(release)})")
    for split, plan in plans.items():
        got = [(f"{split}_{i:02d}", s.kind, s.path, round(s.source_t0, 3), round(s.source_t1, 3))
               for i, st in enumerate(plan.streams) for s in st.segments]
        ref = [(r["stream_id"], r["kind"], r["path"], float(r["source_t0"]), float(r["source_t1"]))
               for r in comp if r["stream_id"].startswith(split)]
        same = len(got) == len(ref) and all(
            a[:3] == b[:3] and abs(a[3] - b[3]) < 2e-3 and abs(a[4] - b[4]) < 2e-3 for a, b in zip(got, ref))
        print(f"  {split}: {len(got)} segments vs {len(ref)} official, identical: {same}")
        ok &= same
    return ok


def fold_clips(clips: list[dict], fold: dict) -> tuple[list[dict], list[dict]]:
    held = {f"F{s}" for s in fold["fall"]} | {f"A{s}" for s in fold["adl"]}
    known = {c["subject"] for c in clips}
    if held - known:
        raise ValueError(f"unknown subjects {sorted(held - known)}")
    return [c for c in clips if c["subject"] not in held], [c for c in clips if c["subject"] in held]


def describe(name: str, clips: list[dict]) -> str:
    falls = [c for c in clips if c["kind"] == "fall"]
    clean = sum(not (c["quality"] == "low" or c["needs_review"].lower() == "true") for c in falls)
    subjects = sorted({c["subject"] for c in clips}, key=lambda s: (s[0], int(s[1:])))
    return f"{name}: {len(falls)} falls ({clean} clean), subjects {' '.join(subjects)}"


def build_fold(release: Path, out: Path, name: str, fold: dict, fold_index: int, k_train: int,
               seed: int, dry_run: bool, storage: str = "virtual", k_heldout: int = 2) -> None:
    streams, bs = organizer_modules(release)
    clips = load_clips(release)
    density = official_density(release)
    train, held = fold_clips(clips, fold)
    print(describe(f"[{name}] train", train))
    print(describe(f"[{name}] held-out", held))
    official = bool(fold.get("official"))
    if official:
        expect = {"train": {c["path"] for c in clips if c["official_split"] == "train"},
                  "val": {c["path"] for c in clips if c["official_split"] == "val"}}
        if {c["path"] for c in train} != expect["train"] or {c["path"] for c in held} != expect["val"]:
            raise ValueError(f"fold {name} is marked official but its clips are not the official split")

    folder = (out or Path(".")) / name
    if not dry_run:                                       # a rebuild must not keep streams of an older build
        for old in folder.glob("streams/*/stream_*.h5"):
            old.unlink()
    events, parts, plans = [], [], []
    for role, split_clips, count in (("train", train, k_train), ("val", held, k_heldout)):
        index = 0
        for k in range(count):
            if official and k == 0:                       # the official streams themselves
                ids = sorted({r["stream_id"] for r in read_csv(release / "streams_composition.csv")
                              if r["stream_id"].startswith(role)})
                rows = [(r["stream_id"], r["kind"], r["path"], r["source_t0"], r["source_t1"])
                        for r in read_csv(release / "streams_composition.csv") if r["stream_id"] in ids]
                plans.append({"role": role, "stitching": k, "seed": "official", "streams": ids,
                              "fingerprint": plan_fingerprint(rows)})
                print(f"[{name}] {role} stitching {k}: the official streams, fingerprint {plans[-1]['fingerprint']}")
                index += len(ids)
                continue
            plan = plan_stitching(streams, split_clips, SPLIT_FOR[role], density, [seed, fold_index, int(role == "val"), k])
            first = index
            for stream in plan.streams:
                if not dry_run:
                    target = folder / "streams" / role / f"stream_{index:02d}.h5"
                    if storage == "virtual":
                        ev, comp = write_virtual_stream(target, role, index, stream.segments, out / SEGMENTS_H5, bs)
                    else:
                        ev, comp = bs._write_stream(target, role, index, stream.segments)
                    events += ev
                    parts += comp
                index += 1
            hours = plan.duration / 3600
            rows = [(f"{role}_{i:02d}", s.kind, s.path, s.source_t0, s.source_t1)
                    for i, st in enumerate(plan.streams, start=first) for s in st.segments]
            plans.append({"role": role, "stitching": k, "seed": [seed, fold_index, int(role == "val"), k],
                          "streams": [f"{role}_{i:02d}" for i in range(first, index)], "hours": round(hours, 3),
                          "fingerprint": plan_fingerprint(rows)})
            print(f"[{name}] {role} stitching {k}: {len(plan.streams)} streams, {hours:.2f} h, "
                  f"{plan.density_per_hour:.2f} falls/h, fingerprint {plans[-1]['fingerprint']}")
    if dry_run:
        return

    if official:                                          # link the official files, copy their rows
        for p in plans:
            if p["seed"] != "official":
                continue
            for sid in p["streams"]:
                role, number = sid.rsplit("_", 1)
                link = folder / "streams" / role / f"stream_{number}.h5"
                link.parent.mkdir(parents=True, exist_ok=True)
                if not link.exists():
                    link.symlink_to((release / "streams" / role / f"stream_{number}.h5").resolve())
        official_ids = {sid for p in plans if p["seed"] == "official" for sid in p["streams"]}
        events = [r for r in read_csv(release / "streams_events.csv") if r["stream_id"] in official_ids] + events
        parts = [r for r in read_csv(release / "streams_composition.csv") if r["stream_id"] in official_ids] + parts
    key = {"train": 0, "val": 1}
    order = lambda r: (key[r["stream_id"].split("_")[0]], int(r["stream_id"].split("_")[1]))  # noqa: E731
    write_csv(folder / "streams_events.csv", sorted(events, key=order))
    write_csv(folder / "streams_composition.csv", sorted(parts, key=order))
    numbering = {s: i for i, s in enumerate(sorted({c["subject"] for c in clips}))}
    write_csv(folder / "groups.csv", [{"path": c["path"], "fold": numbering[c["subject"]], "subject": c["subject"]}
                                      for c in clips])
    (folder / "fold.json").write_text(json.dumps({"fold": name, **fold, "train_stitchings": k_train,
                                                  "heldout_stitchings": k_heldout, "seed": seed,
                                                  "density_per_hour": density, "stitchings": plans}, indent=2))
    verify_labels(release, folder)


def verify_labels(release: Path, folder: Path) -> None:
    """Every fall label: the stream under it is an exact copy of the source fall, and its bounds and flags
    are the official ones of that recording."""
    import h5py
    official = {e["source_path"]: e for e in read_csv(release / "streams_events.csv")}
    bad = []
    by_stream = defaultdict(list)
    for e in read_csv(folder / "streams_events.csv"):
        by_stream[e["stream_id"]].append(e)
    for sid, evs in by_stream.items():
        role, number = sid.rsplit("_", 1)
        with h5py.File(folder / "streams" / role / f"stream_{number}.h5", "r") as h:
            for e in evs:
                a, b = round(float(e["t_start"]) * RATE_HZ), round(float(e["t_end"]) * RATE_HZ)
                s0, s1 = round(float(e["source_t_start"]) * RATE_HZ), round(float(e["source_t_end"]) * RATE_HZ)
                seg = np.load(segment_path(release, e["source_path"]), mmap_mode="r")
                o = official[e["source_path"]]
                same = (np.array_equal(h["cir"][:, a:b], seg[:, s0:s1])
                        and all(e[k] == o[k] for k in ("source_t_start", "source_t_end", "fall_type", "quality",
                                                         "needs_review")))
                if not same:
                    bad.append(e["event_id"])
    n = sum(len(v) for v in by_stream.values())
    print(f"[{folder.name}] labels checked: {n - len(bad)}/{n} exact" + (f"; wrong: {bad[:10]}" if bad else ""))
    if bad:
        raise SystemExit(1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[1], formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--release", type=Path, required=True, help="the official package, with segments/")
    p.add_argument("--out", type=Path, help="where the fold folders go")
    p.add_argument("--folds-config", type=Path, default=Path(__file__).resolve().parents[1] / "configs/folds/subject_folds.yaml")
    p.add_argument("--folds", nargs="*", help="fold names to build (default: all)")
    p.add_argument("--train-stitchings", type=int, default=2, help="stitchings of the training clips per fold")
    p.add_argument("--heldout-stitchings", type=int, default=2,
                   help="stitchings of the held-out clips per fold (each 5 streams, ~2.1 h)")
    p.add_argument("--storage", choices=["virtual", "full"], default="virtual",
                   help="virtual: HDF5 virtual datasets over <out>/segments.h5 (default); full: ordinary files")
    p.add_argument("--seed", type=int, default=2027)
    p.add_argument("--check-official", action="store_true", help="only replan the official streams and compare")
    p.add_argument("--dry-run", action="store_true", help="plan and print the folds, write nothing")
    p.add_argument("--checksums-out", type=Path, help="after building, write the sha256 of every stream and label file")
    p.add_argument("--checksums-ref", type=Path,
                   help="after building, compare with these checksums and fail on any difference (default: "
                        "configs/folds/stream_checksums.csv when the fold file, seed and stitchings are the defaults)")
    p.add_argument("--no-checksums", action="store_true", help="skip the default checksum comparison")
    p.add_argument("--verify-only", action="store_true",
                   help="do not build: compute the checksums of the folds in --out (with --checksums-ref / --checksums-out)")
    args = p.parse_args()

    if args.check_official:
        raise SystemExit(0 if check_official(args.release) else 1)
    if args.out is None and not args.dry_run:
        p.error("--out is required unless --dry-run or --check-official")
    folds = yaml.safe_load(args.folds_config.read_text())["folds"]
    names = args.folds or list(folds)
    default_config = Path(__file__).resolve().parents[1] / "configs/folds/subject_folds.yaml"
    reference = default_config.with_name("stream_checksums.csv")
    if (args.checksums_ref is None and not args.no_checksums and reference.exists() and args.seed == 2027
            and args.train_stitchings == 2 and args.heldout_stitchings == 2
            and args.folds_config.resolve() == default_config):
        args.checksums_ref = reference           # the reference build: the same folds, streams and labels for everyone
    if args.verify_only:
        raise SystemExit(0 if checksums(args, names) else 1)
    if not check_official(args.release):
        raise SystemExit("the duration rules no longer reproduce the official plan; not building")
    if args.storage == "virtual" and not args.dry_run:
        seg_h5 = ensure_segments_h5(args.release, args.out)
        if not check_virtual(args.release, seg_h5, args.out / "_check"):
            raise SystemExit("virtual streams do not reproduce the official files; not building")
    for name in names:
        build_fold(args.release, args.out, name, folds[name], list(folds).index(name), args.train_stitchings,
                   args.seed, args.dry_run, args.storage, args.heldout_stitchings)
    if not args.dry_run and not checksums(args, names):
        raise SystemExit(1)


def checksums(args, names) -> bool:
    if not (args.checksums_out or args.checksums_ref):
        return True
    rows = fold_checksums(args.out, names)
    if args.checksums_out:
        write_csv(args.checksums_out, rows)
        print(f"wrote {args.checksums_out}")
    return compare_checksums(rows, args.checksums_ref) if args.checksums_ref else True


if __name__ == "__main__":
    main()
