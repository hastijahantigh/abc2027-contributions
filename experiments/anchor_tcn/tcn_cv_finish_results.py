#!/usr/bin/env python3
"""Finish the saved CV report after the duplicate 'seed' summary error.

Python standard library only. No training, inference, GPU access, package
installation, fold modification or repeated postprocessing. Reads saved results,
checks the completed jobs, and writes a new finished_results_* directory.

The default run is the completed experiment from the reported traceback.
Use --run /another/run only to finish another run from this same CV script.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import sys
import zipfile

sys.dont_write_bytecode = True
FOLDS = ('A', 'B', 'C', 'D')
DEFAULT_RUN = Path('/home/abc2027/hasti/tcn_shared_cv_20261028_143412_233355_396364')


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('x', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def close(a, b, description):
    if not math.isclose(float(a), float(b), rel_tol=0, abs_tol=1e-12):
        raise ValueError(f'Saved results disagree: {description}')


def safe_child(parent, relative):
    path = (parent / relative).resolve()
    if not path.is_relative_to(parent.resolve()):
        raise ValueError(f'Saved artifact path leaves its job directory: {relative}')
    return path


def check_jobs(run, protocol, protocol_sha):
    jobs, costs = {}, []
    epochs = protocol['config']['epochs']
    for fold in FOLDS:
        jobs[fold] = {}
        for seed in protocol['seeds']:
            parent = run / 'folds' / fold / f'seed_{seed}'
            attempts = sorted(parent.glob('attempt_*'), reverse=True)
            candidates = [p for p in attempts if (p / 'COMPLETE.json').is_file()]
            if not candidates:
                raise ValueError(f'{fold}, seed {seed}: no completed training/inference job. Nothing was retrained.')
            job = candidates[0]
            marker = load(job / 'COMPLETE.json')
            if marker['protocol_sha256'] != protocol_sha:
                raise ValueError(f'{job}: protocol does not match')
            print(f'[check saved job] {fold}, seed {seed}', flush=True)
            for relative, expected in marker['artifact_sha256'].items():
                path = safe_child(job, relative)
                if not path.is_file() or sha(path) != expected:
                    raise ValueError(f'Saved artifact missing or changed: {path}')
            training = load(job / 'training_summary.json')
            inference = load(job / 'inference' / 'summary.json')
            if training['seed'] != seed or training['full_epochs'] != epochs:
                raise ValueError(f'{job}: unexpected seed or epoch budget')
            if inference['validation_ids'] != [f'val_{i:02d}' for i in range(10)]:
                raise ValueError(f'{job}: validation selection differs')
            coverage = load(job / 'epoch_coverage.json')
            if (len(coverage) != epochs or [r['epoch'] for r in coverage] != list(range(1, epochs + 1))
                    or not all(r['every_window_seen_exactly_once']
                               and r['samples_seen'] == r['expected_windows'] for r in coverage)):
                raise ValueError(f'{job}: full-epoch coverage audit failed')
            # Mapping expansion first prevents the original duplicate keyword error.
            costs.append(dict(training, fold=fold, seed=seed, job=str(job),
                inference_seconds=inference['seconds'], inference_frames=inference['frames'],
                inference_real_time_factor=inference['real_time_factor']))
            jobs[fold][seed] = job
    return jobs, costs


def inspect_evaluation(folder, protocol):
    results = []
    for seed in protocol['seeds']:
        seed_out = folder / f'seed_{seed}'
        methods = load(seed_out / 'method_summary.json')
        selections = load(seed_out / 'nested_selections.json')
        shared = load(seed_out / 'supplied_cv_scorer_report.json')
        # These are written only after all nested scoring and deployment tuning finish.
        load(seed_out / 'postprocess_audit.json')
        load(seed_out / 'all_folds_deployment_tuning' / 'selection.json')
        if not (seed_out / 'metrics_all.csv').is_file():
            raise FileNotFoundError(seed_out / 'metrics_all.csv')
        for method, summary in methods.items():
            pooled = load(seed_out / f'{method}_pooled_official_report.json')
            close(summary['pooled_mean_f1'], pooled['ranking_score']['f1_mean'], f'{seed}/{method}/pooled')
            for fold in FOLDS:
                report = load(seed_out / fold / f'{method}_official_report.json')
                close(summary['fold_scores'][fold], report['ranking_score']['f1_mean'], f'{seed}/{method}/{fold}')
        close(shared['pooled']['score'], methods['selected']['pooled_mean_f1'], f'{seed}/shared pooled scorer')
        for fold in FOLDS:
            choice = selections[fold]
            if set(choice['tuning_folds']) != set(FOLDS) - {fold}:
                raise ValueError(f'{seed}/{fold}: tuning folds do not exclude the target')
            selected = load(seed_out / fold / 'selected_official_report.json')
            chosen = load(seed_out / fold / f'{choice["method"]}_official_report.json')
            if selected != chosen:
                raise ValueError(f'{seed}/{fold}: selected report differs from the saved chosen method')
            close(shared['per_fold'][fold]['score'], selected['ranking_score']['f1_mean'], f'{seed}/{fold}/shared scorer')
        results.append(dict(seed=seed, methods=methods, nested_selections=selections))
    if any(set(r['methods']) != set(results[0]['methods']) for r in results):
        raise ValueError('Method names differ between seeds')
    return results


def summarize(results):
    rows = []
    for method in results[0]['methods']:
        scores = [r['methods'][method]['pooled_mean_f1'] for r in results]
        row = dict(method=method, seeds=len(scores), mean_pooled_f1=statistics.mean(scores),
            sd_pooled_f1=statistics.stdev(scores) if len(scores) > 1 else 0,
            min_pooled_f1=min(scores), max_pooled_f1=max(scores))
        for fold in FOLDS:
            row[f'fold_{fold}_mean_f1'] = statistics.mean(r['methods'][method]['fold_scores'][fold] for r in results)
        row['worst_fold_mean_f1'] = min(row[f'fold_{f}_mean_f1'] for f in FOLDS)
        rows.append(row)
    return rows


def merge_metrics(folder, results, destination):
    fields, paths = [], []
    for result in results:
        path = folder / f'seed_{result["seed"]}' / 'metrics_all.csv'
        with path.open(encoding='utf-8-sig', newline='') as handle:
            reader = csv.DictReader(handle)
            for field in reader.fieldnames or []:
                if field not in fields:
                    fields.append(field)
        paths.append(path)
    count = 0
    with destination.open('x', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for path in paths:
            with path.open(encoding='utf-8-sig', newline='') as source:
                for row in csv.DictReader(source):
                    writer.writerow(row)
                    count += 1
    return count


def write_report(path, protocol, results, summaries, costs):
    lines = ['# TCN subject-held-out CV results', '',
        f'Four folds, seeds {protocol["seeds"]}, {protocol["config"]["epochs"]} full epochs each. '
        f'Checkpoint policy: {protocol["checkpoint_policy"]}.',
        'Each model trained on its fold\'s 26 training streams and predicted val_00 through val_09.',
        'The network, normalization and window settings were fixed before CV. '
        'For each scored fold, decoder and postprocessing selection used only the other three folds.', '',
        '| Seed | Pooled mean F1 | A | B | C | D |',
        '|---|---:|---:|---:|---:|---:|']
    for result in results:
        chosen = result['methods']['selected']
        lines.append(f'| {result["seed"]} | {chosen["pooled_mean_f1"]:.6f} | ' +
                     ' | '.join(f'{chosen["fold_scores"][f]:.6f}' for f in FOLDS) + ' |')
    chosen_summary = next(r for r in summaries if r['method'] == 'selected')
    lines += ['', f'Mean pooled F1 across seeds: **{chosen_summary["mean_pooled_f1"]:.6f}**; '
        f'sample SD: **{chosen_summary["sd_pooled_f1"]:.6f}**.', '',
        '| Processing stage | Mean pooled F1 across seeds | SD |',
        '|---|---:|---:|']
    for row in summaries:
        lines.append(f'| {row["method"]} | {row["mean_pooled_f1"]:.6f} | {row["sd_pooled_f1"]:.6f} |')
    lines += ['', 'The ranking mean uses event F1 at tIoU 0.1, 0.3 and 0.5; 0.7 is diagnostic.',
        'metrics_all.csv contains every saved official scalar metric at pooled, fold, stitching and stream level: '
        'precision, recall, F1, counts, false alarms/hour and boundary errors where available.',
        'Full official JSON reports, all selection settings, sweeps, training losses and window audits are included in the ZIP.',
        'The selected method can differ by target fold. The choice uses the other three folds, not the target result.',
        'All-fold deployment tuning is saved separately and is not an independent CV score.',
        'The earlier 0.9141 was obtained on one repeatedly tuned validation split. '
        'It is not directly comparable to the pooled subject-held-out CV score.', '',
        f'Total recorded training time across {len(costs)} runs: '
        f'{sum(c["training_seconds"] for c in costs)/60:.2f} minutes.',
        f'Total recorded inference time: {sum(c["inference_seconds"] for c in costs)/60:.2f} minutes.',
        'These totals exclude data verification, preprocessing, postprocessing, tuning and report writing.', '',
        'Recovery changed only report assembly after the duplicate seed keyword error. '
        'No models, predictions, thresholds or official scores were changed.',
        'This is a CV evaluation; no hidden-test score or measured alarm latency is available.', '']
    with path.open('x', encoding='utf-8') as handle:
        handle.write('\n'.join(lines))


def archive_results(run, evaluation, finished):
    archive_path = finished / 'RESULTS_TO_SHARE.zip'
    suffixes = {'.json', '.csv', '.md', '.txt', '.log', '.py'}
    with zipfile.ZipFile(archive_path, 'x', zipfile.ZIP_DEFLATED) as archive:
        # Fresh summary files at the top level, including the authoritative completion marker.
        for path in sorted(finished.iterdir()):
            if path.is_file() and path.suffix.lower() in suffixes:
                archive.write(path, path.name)
        for path in sorted(run.rglob('*')):
            relative = path.relative_to(run)
            if (not path.is_file() or path.suffix.lower() not in suffixes
                    or path.is_symlink() or not path.resolve().is_relative_to(run)):
                continue
            if any(p.startswith(('finished_results_', 'ema_cache_', 'selected_data_view_')) for p in relative.parts):
                continue
            if relative.parts[0].startswith('evaluation_') and relative.parts[0] != evaluation.name:
                continue
            archive.write(path, 'original_run/' + relative.as_posix())
    return archive_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', type=Path, default=DEFAULT_RUN)
    args = parser.parse_args(argv)
    run = args.run.expanduser().resolve()
    protocol = load(run / 'protocol.json')
    protocol_sha = sha(run / 'protocol.json')
    if (protocol.get('version') != 'shared-first10-1.0'
            or protocol.get('selected_validation_ids') != [f'val_{i:02d}' for i in range(10)]):
        raise ValueError('This helper is only for the first-10 shared-CV experiment')
    jobs, costs = check_jobs(run, protocol, protocol_sha)
    evaluation, results = None, None
    errors = []
    for folder in sorted(run.glob('evaluation_*'), reverse=True):
        try:
            results = inspect_evaluation(folder, protocol)
            evaluation = folder
            break
        except (OSError, ValueError, KeyError) as error:
            errors.append(f'{folder.name}: {error}')
    if evaluation is None:
        raise RuntimeError('No fully saved evaluation found. Nothing was retrained. Details: ' + '; '.join(errors))
    summaries = summarize(results)
    name = datetime.now(timezone.utc).strftime('finished_results_%Y%m%d_%H%M%S_%f') + f'_{os.getpid()}'
    finished = run / name
    finished.mkdir(exist_ok=False)
    write_csv(finished / 'seed_summary.csv', summaries)
    write_csv(finished / 'runtime_summary.csv', costs)
    save(finished / 'seed_results.json', results)
    metrics = merge_metrics(evaluation, results, finished / 'metrics_all.csv')
    write_report(finished / 'RESULTS.md', protocol, results, summaries, costs)
    save(finished / 'completion.json', dict(status='COMPLETE', recovery='Report assembly only',
        source_run=str(run), source_evaluation=str(evaluation), protocol_sha256=protocol_sha,
        training_jobs=len(costs), seeds=protocol['seeds'], folds=list(FOLDS), metric_rows=metrics,
        all_saved_artifact_hashes_verified=True, repeated_training=False, repeated_inference=False,
        repeated_postprocessing=False, finished_utc=datetime.now(timezone.utc).isoformat()))
    shutil.copyfile(Path(__file__), finished / Path(__file__).name)
    archive = archive_results(run, evaluation, finished)
    print('\nCOMPLETE. All saved training and evaluation results reused.', flush=True)
    for result in results:
        print(f'Seed {result["seed"]}: pooled mean F1 = {result["methods"]["selected"]["pooled_mean_f1"]:.6f}', flush=True)
    selected = next(r for r in summaries if r['method'] == 'selected')
    print(f'Mean across seeds: {selected["mean_pooled_f1"]:.6f}', flush=True)
    print(f'Report: {finished / "RESULTS.md"}\nSEND BACK: {archive}', flush=True)
    return finished


if __name__ == '__main__':
    main()
