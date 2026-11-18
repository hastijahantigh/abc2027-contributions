"""
Per-fall analysis on several splits at once: why some fall types (Chute04) and some labels (exactly
2.00 s) are missed, and whether the training streams show the same thing as val.

    uv run python tools/analyze_falls.py \
        --model train=runs/UWB/Frame/U0/<id>/predictions/train-best \
        --model val=runs/UWB/Frame/U0/<id>/predictions/val-best \
        [--signal runs/_analysis/fall_signal.npz] [--dataset runs/_dataset] \
        [--html report/falljepa_fall_types.html] [--csv runs/_analysis/falls.csv]

Each --model is SPLIT=<prefix>, the prefix of <prefix>.csv and <prefix>.probs.npz written by a
whole-stream evaluation (tools/infer_splits.sh, or run.py validate: val-validate). --signal is the
output of tools/fall_signal.py (needs the streams, so it is made on the server); without it the
signal columns and plots are left out.

Every fall of those splits gets one row: labels (type, dataset, subject, trial, flags, duration,
2.00 s), stream context (the fall recording's margins, the activity before it), the model (status at
tIoU 0.3 and 0.5 under --ignore-flagged, P(fall) inside the label and around it, the nearest
prediction) and the signal (when the motion peaks and settles, when the static profile switches,
how large the change is). The HTML report compares groups of falls between the splits.
"""
import argparse
import csv
import html
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from falljepa.official import evaluate as official  # noqa: E402

RATE = 50
PROFILE = (-4.0, 8.0)       # seconds around the annotated start, for the mean profiles


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def truthy(v) -> bool:
    return str(v).strip().lower() in {"1", "true", "yes"}


def nanmedian(values) -> float | None:
    a = np.asarray([v for v in values if v is not None], dtype=float)
    a = a[np.isfinite(a)]
    return float(np.median(a)) if len(a) else None


# ------------------------------------------------------------------ rows


def label_rows(events: list[dict], comp: list[dict]) -> list[dict]:
    """Labels and stream context of every fall."""
    segs = defaultdict(list)
    for s in comp:
        segs[s["stream_id"]].append(s)
    for v in segs.values():
        v.sort(key=lambda s: float(s["t_start"]))
    rows = []
    for e in events:
        t0, t1 = float(e["t_start"]), float(e["t_end"])
        parts = e["source_path"].split("/")
        name = Path(e["source_path"]).stem                         # Chute04_283
        ss = segs.get(e["stream_id"], [])
        clip = next((i for i, s in enumerate(ss) if s["kind"] == "fall" and float(s["t_start"]) <= t0 and t1 <= float(s["t_end"])), None)
        prev = ss[clip - 1] if clip else None
        rows.append({
            "split": e["stream_id"].split("_")[0], "stream_id": e["stream_id"], "event_id": e["event_id"],
            "t_start": t0, "t_end": t1, "duration": round(t1 - t0, 3), "is_2s": abs(t1 - t0 - 2.0) < 0.011,
            "fall_type": e["fall_type"], "dataset": e["dataset"].replace("Dataset ", "D"),
            "subject": parts[2] if len(parts) > 3 else "", "trial": name.split("_")[-1],
            "quality": e["quality"], "needs_review": truthy(e["needs_review"]),
            "flagged": e["quality"].lower() == "low" or truthy(e["needs_review"]) or truthy(e["ignore"]),
            "source_t_start": float(e["source_t_start"]) if e["source_t_start"] else None,
            "clip_before_s": t0 - float(ss[clip]["t_start"]) if clip is not None else None,
            "clip_after_s": float(ss[clip]["t_end"]) - t1 if clip is not None else None,
            "before_kind": (prev["label"] or prev["kind"]) if prev else ("stream start" if clip == 0 else None),
        })
    return rows


def model_columns(rows: list[dict], prefix: Path, ground: list[dict]) -> dict:
    """Adds the model's view of each fall of this split; returns the per-timestep P(fall) per stream."""
    npz = np.load(f"{prefix}.probs.npz")
    pfall = {n.split("/", 1)[1]: 1 - np.exp(npz[n].astype(np.float64)[:, 0]) for n in npz.files if n.startswith("logp/")}
    preds = [{"stream_id": r["stream_id"], "t_start": float(r["t_start"]), "t_end": float(r["t_end"]),
              "score": float(r["score"])} for r in read_csv(Path(f"{prefix}.csv"))]
    by_pred = defaultdict(list)
    for p in preds:
        by_pred[p["stream_id"]].append(p)
    status = {}
    for sid in pfall:
        gt = [e for e in ground if e["stream_id"] == sid]
        for theta in (0.3, 0.5):
            res = official._match_stream(gt, by_pred[sid], theta)
            hit = {gi: iou for _pi, gi, iou in res["pairs"]}
            for gi, e in enumerate(res["scored"]):
                status[(sid, round(e["t_start"], 3), theta)] = hit.get(gi)
    for r in rows:
        sid = r["stream_id"]
        if sid not in pfall:
            continue
        p = pfall[sid]
        a, b = int(round(r["t_start"] * RATE)), int(round(r["t_end"] * RATE))
        inside = p[a:b]
        around = p[max(a - 4 * RATE, 0):b + 4 * RATE]
        ps = by_pred[sid]
        ious = [official.temporal_iou(q["t_start"], q["t_end"], r["t_start"], r["t_end"]) for q in ps]
        mid = (r["t_start"] + r["t_end"]) / 2
        near = min(ps, key=lambda q: abs((q["t_start"] + q["t_end"]) / 2 - mid)) if ps else None
        key = (sid, round(r["t_start"], 3))
        h3, h5 = status.get((*key, 0.3)), status.get((*key, 0.5))
        r.update({
            "status": "flagged" if r["flagged"] else ("hit" if h3 is not None else "missed"),
            "hit_0.5": None if r["flagged"] else h5 is not None, "tiou": h3,
            "best_tiou": max(ious, default=0.0),
            "p_max_in": float(inside.max()) if len(inside) else None,
            "p_mean_in": float(inside.mean()) if len(inside) else None,
            "p_max_pm4": float(around.max()) if len(around) else None,
            "p_peak_rel_start": (int(np.argmax(around)) + max(a - 4 * RATE, 0)) / RATE - r["t_start"] if len(around) else None,
            "p_mean_before2": float(p[max(a - 2 * RATE, 0):a].mean()), "p_mean_after2": float(p[b:b + 2 * RATE].mean()),
            "near_pred_start": None if near is None else near["t_start"] - r["t_start"],
            "near_pred_end": None if near is None else near["t_end"] - r["t_start"],
            "near_pred_score": None if near is None else near["score"],
        })
    return pfall


def signal_columns(rows: list[dict], sig) -> None:
    """When the motion peaks and settles and when the static profile switches, relative to the label."""
    index = {e: i for i, e in enumerate(sig["event_id"])}
    t = sig["t"]
    for r in rows:
        i = index.get(r["event_id"])
        if i is None:
            continue
        change = sig["change"][i]
        if not np.isfinite(change).any():
            continue
        k = int(np.nanargmax(change))                               # the radar whose scene changes most
        m, s = sig["motion"][i, k], sig["state"][i, k]
        mf = np.nanmax(sig["motion"][i], axis=0)                     # the most active radar at each step
        d = r["duration"]
        rel_end = t - d
        win = (t >= -3) & (t <= d + 3)
        base = np.nanpercentile(m[(t >= -6) & (t <= -2)], 30) if np.isfinite(m[(t >= -6) & (t <= -2)]).any() else np.nan
        peak_i = int(np.nanargmax(np.where(win, m, -np.inf)))
        peak = m[peak_i]
        level = base + 0.15 * (peak - base)
        after = np.where((t > t[peak_i]) & (m < level))[0]
        settle = t[after[0]] if len(after) else np.nan
        before = np.where((t < t[peak_i]) & (m < level))[0]
        onset = t[before[-1]] if len(before) else np.nan

        def cross(x):
            j = np.where(win & (s >= x))[0]
            return t[j[0]] if len(j) else np.nan
        energy = np.clip(m - base, 0, None)
        e_all = np.nansum(energy[win])
        e_in = np.nansum(energy[(t >= 0) & (t <= d)])
        r.update({
            "anchor_radar": k, "change_max": float(change[k]), "change_sum": float(np.nansum(change)),
            "motion_base": float(base), "motion_peak": float(peak), "motion_peak_rel_start": float(t[peak_i]),
            "motion_onset_rel_start": float(onset), "motion_settle_rel_end": float(settle - d),
            "motion_at_end_frac": float((np.interp(d, t, m) - base) / max(peak - base, 1e-9)),
            "motion_in_label_frac": float(e_in / e_all) if e_all > 0 else None,
            "motion_pre": float(np.nanmean(mf[(t >= -3) & (t < -0.5)])),
            "motion_post": float(np.nanmean(mf[(rel_end > 0.5) & (rel_end <= 3)])),
            "state_20_rel_start": float(cross(0.2)), "state_50_rel_start": float(cross(0.5)),
            "state_80_rel_end": float(cross(0.8) - d),
        })


# --------------------------------------------------------------- groups


def summarize(rows: list[dict], key) -> list[dict]:
    groups = defaultdict(list)
    for r in rows:
        k = key(r)
        if k is not None:
            groups[(r["split"], k)].append(r)
    out = []
    for (split, k), rs in sorted(groups.items()):
        sc = [r for r in rs if not r["flagged"]]
        fl = [r for r in rs if r["flagged"]]
        has_model = any("status" in r for r in rs)
        g = {"split": split, "group": k, "n": len(rs), "scored": len(sc), "flagged": len(fl),
             "dur_median": nanmedian(r["duration"] for r in rs), "is_2s": sum(r["is_2s"] for r in rs)}
        if has_model and sc:
            g |= {"recall_0.3": float(np.mean([r["status"] == "hit" for r in sc])),
                  "recall_0.5": float(np.mean([bool(r["hit_0.5"]) for r in sc])),
                  "p_max_in": nanmedian(r.get("p_max_in") for r in sc),
                  "p_mean_in": nanmedian(r.get("p_mean_in") for r in sc),
                  "silent": int(sum((r.get("p_max_pm4") or 0) < 0.5 for r in sc))}
        if has_model and fl:
            g["p_max_in_flagged"] = nanmedian(r.get("p_max_in") for r in fl)
        for c in ("change_max", "motion_peak", "motion_peak_rel_start", "motion_onset_rel_start",
                  "motion_settle_rel_end", "motion_at_end_frac", "motion_in_label_frac", "state_50_rel_start",
                  "state_80_rel_end", "motion_pre", "motion_post"):
            v = nanmedian(r.get(c) for r in sc or rs)
            if v is not None:
                g[c] = v
        out.append(g)
    return out


def mean_profiles(rows, pfall, sig) -> dict:
    """Mean P(fall), motion and state aligned on the annotated start, per split x (Chute04 or not)."""
    lo, hi = PROFILE
    tt = np.arange(int(lo * RATE), int(hi * RATE), 5) / RATE           # 10 Hz
    out = {"t": tt.round(2).tolist(), "series": {}}
    index = {e: i for i, e in enumerate(sig["event_id"])} if sig is not None else {}
    for r in rows:
        if r["flagged"]:
            continue
        key = f"{r['split']}|{'Chute04' if r['fall_type'] == 'Chute04' else 'other types'}"
        acc = out["series"].setdefault(key, {"n": 0, "p": [], "motion": [], "state": [], "dur": []})
        acc["n"] += 1
        acc["dur"].append(r["duration"])
        p = pfall.get(r["split"], {}).get(r["stream_id"])
        if p is not None:
            idx = (np.round((r["t_start"] + tt) * RATE)).astype(int)
            ok = (idx >= 0) & (idx < len(p))
            acc["p"].append(np.where(ok, p[np.clip(idx, 0, len(p) - 1)], np.nan))
        i = index.get(r["event_id"])
        if i is not None and np.isfinite(sig["change"][i]).any():
            k = int(np.nanargmax(sig["change"][i]))
            acc["motion"].append(np.interp(tt, sig["t"], sig["motion"][i, k]))
            acc["state"].append(np.interp(tt, sig["t"], sig["state"][i, k]))
    for acc in out["series"].values():
        for c in ("p", "motion", "state"):
            acc[c] = np.nanmean(np.array(acc[c]), axis=0).round(4).tolist() if acc[c] else None
        acc["dur"] = float(np.median(acc["dur"]))
    return out


# ------------------------------------------------------------------ html


def fmt(v, digits=2):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    if isinstance(v, bool):
        return "yes" if v else ""
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return html.escape(str(v))


COLUMNS = [("split", "split"), ("group", "group"), ("n", "falls"), ("scored", "scored"), ("flagged", "flagged"),
           ("is_2s", "2.00 s"), ("dur_median", "median dur (s)"), ("recall_0.3", "recall θ0.3"),
           ("recall_0.5", "recall θ0.5"), ("silent", "silent (P<0.5 ±4 s)"), ("p_max_in", "max P in label"),
           ("p_mean_in", "mean P in label"), ("p_max_in_flagged", "max P, flagged"),
           ("change_max", "scene change"), ("motion_peak", "motion peak"),
           ("motion_onset_rel_start", "motion onset − start"), ("motion_peak_rel_start", "motion peak − start"),
           ("motion_settle_rel_end", "motion settles − end"), ("motion_at_end_frac", "motion left at end"),
           ("motion_in_label_frac", "motion inside label"), ("state_50_rel_start", "profile 50 % − start"),
           ("state_80_rel_end", "profile 80 % − end"), ("motion_pre", "motion before"), ("motion_post", "motion after")]


def table(groups: list[dict]) -> str:
    cols = [(k, h) for k, h in COLUMNS if any(k in g for g in groups)]
    head = "".join(f"<th>{h}</th>" for _k, h in cols)
    body = "".join("<tr>" + "".join(f"<td>{fmt(g.get(k))}</td>" for k, _h in cols) + "</tr>" for g in groups)
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


FALL_COLS = ["split", "stream_id", "t_start", "duration", "fall_type", "dataset", "subject", "trial", "status",
             "best_tiou", "p_max_in", "p_max_pm4", "p_peak_rel_start", "near_pred_start", "near_pred_end",
             "before_kind", "clip_before_s", "change_max", "motion_onset_rel_start", "motion_peak_rel_start",
             "motion_settle_rel_end", "motion_at_end_frac", "state_50_rel_start", "state_80_rel_end"]


def fall_table(rows: list[dict]) -> str:
    cols = [c for c in FALL_COLS if any(c in r for r in rows)]
    head = "".join(f"<th>{c}</th>" for c in cols)
    body = "".join("<tr>" + "".join(f"<td>{fmt(r.get(c))}</td>" for c in cols) + "</tr>" for r in rows)
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def write_html(path: Path, sections: list[tuple[str, str]], profiles: dict, notes: list[str]) -> None:
    body = "\n".join(f"<section><h2>{html.escape(t)}</h2>{c}</section>" for t, c in sections)
    page = HTML.replace("__NOTES__", "".join(f"<li>{n}</li>" for n in notes)) \
               .replace("__BODY__", body).replace("__PROFILES__", json.dumps(profiles))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page, encoding="utf-8")


HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Fall Types Analysis</title>
<style>
:root{--surface:#fcfcfb;--text:#0b0b0b;--text2:#52514e;--grid:#e4e3df;--line:#c9c8c2;--s1:#2a78d6;--s2:#eb6834;--hl:#fff5d6}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--surface:#1a1a19;--text:#fff;--text2:#c3c2b7;--grid:#2f2f2d;--line:#4a4a47;--s1:#3987e5;--s2:#d95926;--hl:#3a3320}}
:root[data-theme="dark"]{--surface:#1a1a19;--text:#fff;--text2:#c3c2b7;--grid:#2f2f2d;--line:#4a4a47;--s1:#3987e5;--s2:#d95926;--hl:#3a3320}
body{background:var(--surface);color:var(--text);font:14px/1.5 system-ui,sans-serif;margin:0 auto;max-width:1200px;padding:16px}
h1{font-size:22px}h2{font-size:17px;margin-top:32px}ul{color:var(--text2)}
.scroll{overflow-x:auto}table{border-collapse:collapse;font-size:12px;font-variant-numeric:tabular-nums}
th,td{border-bottom:1px solid var(--grid);padding:3px 8px;text-align:right;white-space:nowrap}th{color:var(--text2);font-weight:600}
td:nth-child(-n+2),th:nth-child(-n+2){text-align:left}
.charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:16px}
.chart h3{font-size:13px;margin:0 0 4px;font-weight:600}.chart svg{width:100%;height:auto;overflow:visible}
.legend{display:flex;gap:16px;font-size:12px;color:var(--text2);margin:8px 0}.legend i{display:inline-block;width:14px;height:2px;vertical-align:middle;margin-right:6px}
.tip{position:fixed;pointer-events:none;background:var(--surface);border:1px solid var(--line);padding:4px 8px;font-size:12px;border-radius:4px;display:none}
</style></head><body>
<h1>Fall types, 2.00 s labels: train vs val</h1>
<ul>__NOTES__</ul>
<section><h2>Mean profiles around the annotated start (scored falls)</h2>
<div class="legend"><span><i style="background:var(--s1)"></i>other types</span><span><i style="background:var(--s2)"></i>Chute04</span><span>dashed: label end (median duration)</span></div>
<div class="charts" id="charts"></div></section>
__BODY__
<div class="tip" id="tip"></div>
<script>
const P=__PROFILES__;const tip=document.getElementById('tip');
const panels=[["p","P(fall) from the model",0,1],["motion","motion, anchor radar (z-diff RMS)",null,null],["state","static profile: before (0) → after (1)",0,1]];
const splits=[...new Set(Object.keys(P.series).map(k=>k.split('|')[0]))];
const W=360,H=170,L=36,R=8,T=8,B=24,box=document.getElementById('charts');
for(const [key,title,y0,y1] of panels){for(const sp of splits){
 const ser=[['other types','var(--s1)'],['Chute04','var(--s2)']].map(([g,c])=>({g,c,d:P.series[sp+'|'+g]})).filter(s=>s.d&&s.d[key]);
 if(!ser.length)continue;
 let lo=y0,hi=y1;if(lo===null){const v=ser.flatMap(s=>s.d[key]).filter(Number.isFinite);lo=Math.min(...v);hi=Math.max(...v);}
 const t=P.t,x=v=>L+(v-t[0])/(t[t.length-1]-t[0])*(W-L-R),y=v=>T+(1-(v-lo)/(hi-lo||1))*(H-T-B);
 let s=`<svg viewBox="0 0 ${W} ${H}">`;
 for(let k=0;k<=4;k++){const v=lo+(hi-lo)*k/4;s+=`<line x1="${L}" x2="${W-R}" y1="${y(v)}" y2="${y(v)}" stroke="var(--grid)"/><text x="${L-4}" y="${y(v)+3}" font-size="9" text-anchor="end" fill="var(--text2)">${v.toFixed(hi-lo<3?2:1)}</text>`;}
 for(let v=Math.ceil(t[0]);v<=t[t.length-1];v+=2)s+=`<text x="${x(v)}" y="${H-8}" font-size="9" text-anchor="middle" fill="var(--text2)">${v}s</text>`;
 s+=`<line x1="${x(0)}" x2="${x(0)}" y1="${T}" y2="${H-B}" stroke="var(--line)"/>`;
 for(const q of ser){s+=`<line x1="${x(q.d.dur)}" x2="${x(q.d.dur)}" y1="${T}" y2="${H-B}" stroke="${q.c}" stroke-dasharray="3 3" opacity=".6"/>`;
  s+=`<polyline fill="none" stroke="${q.c}" stroke-width="2" points="${q.d[key].map((v,i)=>Number.isFinite(v)?x(t[i])+','+y(v):'').join(' ')}"/>`;}
 s+=`<line class="xh" y1="${T}" y2="${H-B}" stroke="var(--text2)" visibility="hidden"/><rect x="${L}" y="${T}" width="${W-L-R}" height="${H-T-B}" fill="transparent"/></svg>`;
 const div=document.createElement('div');div.className='chart';div.innerHTML=`<h3>${sp}: ${title} <span style="color:var(--text2);font-weight:400">(n = ${ser.map(q=>q.g+' '+q.d.n).join(', ')})</span></h3>`+s;box.appendChild(div);
 const svg=div.querySelector('svg'),xh=svg.querySelector('.xh');
 svg.addEventListener('mousemove',ev=>{const r=svg.getBoundingClientRect(),px=(ev.clientX-r.left)/r.width*W;const i=Math.max(0,Math.min(t.length-1,Math.round((px-L)/(W-L-R)*(t.length-1))));
  xh.setAttribute('x1',x(t[i]));xh.setAttribute('x2',x(t[i]));xh.setAttribute('visibility','visible');
  tip.innerHTML=`t = ${t[i].toFixed(1)} s<br>`+ser.map(q=>`${q.g}: ${Number.isFinite(q.d[key][i])?q.d[key][i].toFixed(3):'—'}`).join('<br>');
  tip.style.display='block';tip.style.left=(ev.clientX+12)+'px';tip.style.top=(ev.clientY+12)+'px';});
 svg.addEventListener('mouseleave',()=>{xh.setAttribute('visibility','hidden');tip.style.display='none';});
}}
</script></body></html>"""


# -------------------------------------------------------------------- main


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", action="append", default=[], help="SPLIT=<prefix of .csv and .probs.npz>")
    p.add_argument("--signal", type=Path)
    p.add_argument("--dataset", type=Path, default=Path("runs/_dataset"))
    p.add_argument("--html", type=Path, default=Path("report/falljepa_fall_types.html"))
    p.add_argument("--csv", type=Path, default=Path("runs/_analysis/falls.csv"))
    args = p.parse_args(argv)

    models = dict(m.split("=", 1) for m in args.model)
    events = read_csv(args.dataset / "streams_events.csv")
    splits = sorted(models) or sorted({e["stream_id"].split("_")[0] for e in events})
    events = [e for e in events if e["stream_id"].split("_")[0] in splits]
    rows = label_rows(events, read_csv(args.dataset / "streams_composition.csv"))
    ground = official.load_ground_truth(args.dataset / "streams_events.csv", ignore_flagged=True)
    pfall = {}
    for split, prefix in models.items():
        pfall[split] = model_columns([r for r in rows if r["split"] == split], Path(prefix), ground)
    sig = None
    if args.signal and args.signal.exists():
        sig = dict(np.load(args.signal))
        signal_columns(rows, sig)

    c04 = lambda r: "Chute04" if r["fall_type"] == "Chute04" else "other types"  # noqa: E731
    sections = [
        ("By fall type", table(summarize(rows, lambda r: r["fall_type"]))),
        ("By fall type and flag (flagged falls are ignore zones in scoring)",
         table(summarize(rows, lambda r: f"{r['fall_type']} {'flagged' if r['flagged'] else 'scored'}"))),
        ("Exactly 2.00 s labels, Chute04 or not", table(summarize(rows, lambda r: f"{c04(r)}, {'2.00 s' if r['is_2s'] else 'other duration'}"))),
        ("By duration", table(summarize(rows, lambda r: "< 1.5 s" if r["duration"] < 1.5 else "1.5-1.99 s" if r["duration"] < 1.99
                                        else "2.00 s" if r["is_2s"] else "2.01-3 s" if r["duration"] <= 3 else "> 3 s"))),
        ("By dataset and fall type", table(summarize(rows, lambda r: f"{r['dataset']} {r['fall_type']}"))),
        ("Chute04 by subject", table(summarize([r for r in rows if r["fall_type"] == "Chute04"], lambda r: r["subject"]))),
        ("By subject (all types)", table(summarize(rows, lambda r: r["subject"]))),
        ("By what precedes the fall recording", table(summarize(rows, lambda r: r["before_kind"]))),
    ]
    if models:
        missed = [r for r in rows if r.get("status") == "missed"]
        c4 = [r for r in rows if r["fall_type"] == "Chute04" and not r["flagged"]]
        sections += [("Every scored Chute04 fall", fall_table(sorted(c4, key=lambda r: (r["split"], r.get("status", ""), r["stream_id"], r["t_start"])))),
                     ("Every missed scored fall", fall_table(sorted(missed, key=lambda r: (r["split"], r["fall_type"], r["stream_id"], r["t_start"])))),
                     ("Every scored 2.00 s fall", fall_table(sorted([r for r in rows if r["is_2s"] and not r["flagged"]],
                                                                    key=lambda r: (r["split"], r["fall_type"]))))]
    notes = [f"splits: {', '.join(splits)}; models: {html.escape(json.dumps(models))}",
             f"signal: {html.escape(str(args.signal)) if sig is not None else 'not given (run tools/fall_signal.py on the server)'}",
             "status at tIoU 0.3 under --ignore-flagged; flagged falls are not scored. Times in seconds; '− start' and '− end' are relative to the label.",
             "motion: per-radar RMS of the z-scored temporal difference (≈1 typical); anchor radar = the one whose static profile changes most; "
             "'motion left at end' = (motion at label end − baseline) / (peak − baseline); 'profile 50 % − start' = when the scene is halfway from "
             "the before- to the after-fall profile."]
    profiles = mean_profiles(rows, pfall, sig)
    write_html(args.html, sections, profiles, notes)

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with args.csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    for g in summarize(rows, lambda r: r["fall_type"]):
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in g.items()})
    print(f"written {args.html} and {args.csv}")


if __name__ == "__main__":
    main()
