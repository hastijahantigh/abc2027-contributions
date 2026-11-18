# AP3 ablation plan: from 0.91 to above 0.95, and stable

2026-10-16 · base: `configs/uwb/AP.yaml` + `rdrop` (early radar fusion kept) · metric: official ranking
score on val (mean F1 at tIoU 0.1 / 0.3 / 0.5, 67 scored falls, 1 fall ≈ 0.015), SWA model only.

## 1. Where the current models fail

All 16 AP / AP2 SWA runs, every fall and false alarm matched at each tIoU
(scratch analysis over `runs/UWB/AP/*/*/predictions/val-validate.csv`).

| error | per run (θ .3) | seen in | what it is |
|---|---|---|---|
| **2 misses + 2 false alarms on P1_9 Chute04_12/13** | 4 | **16 / 16 runs** | Label artefact. Each label sits 1.4–1.7 s into its recording. Every model, and the official labeller run on the stream crop, puts the fall 5–8 s later. P(fall) at the label is ≈ 0. **Fixed cost: about −0.03 in every run.** |
| Misses on Chute04 of P2_8 (862, 1485, 1606 s) and P2_9 (720 s) | 0–5 | depends on the seed | The model's real weakness. P(fall) is 0.1–0.7 and changes with the seed: JEPA-s0 finds them, rdrop-s1 finds none. Short lead-in (1.2–2 s after a join, or right after another fall recording), still moving after the label. |
| Other misses on easy falls | 0–13 | 1–6 runs each | Seed noise. noJEPA-s1/s2 drive most of it. |
| Localisation (hit at .1/.3, not at .5) | 0–1 | a few runs | Refinement on the wrong span, e.g. val_01 1436.86 sits behind a 1.0 s tea clip (tIoU .45). |
| ADL false alarms (Faire_the, Faire_menage, Enfiler_une_veste) | 0–9 | **no cluster in more than 6 / 16 runs** | Seed noise. rshift and jepaplus add many; rdrop and adlneg have the fewest. |
| Autre / Empty / join false alarms | 0–3 | 1–4 runs each | Seed noise. Gone in rdrop and adlneg. |

**What this means**

1. **Ceiling.** With the artefact pair, a perfect model scores 130/134 = **0.970**. To score above **0.95**, a model may make **at most about 2 other errors at each tIoU**. rdrop-s0, our best run (0.922), makes about 6. The 6-model SWA ensemble (0.926) makes 5, and only 1 of them is a false alarm.
2. **Most errors are variance, not bias.** Apart from the artefact, no false alarm and no easy-fall miss repeats across most seeds. 46 of the 65 fixable falls are hit by all 16 runs. Ensembles remove these errors (6 SWA models: 3 false alarms in total, 2 of them the artefact). **Stability work and score work are therefore the same work.**
3. **One systematic gap is left: recall on one subject's Chute04.** It is a subject shift. Train has the fall type (41 Chute04), but not this way of falling: short lead-in, motion after the label. Only data-side changes (context or lead-in diversity, single-radar evidence) can fix it. Ensembling cannot, because all members miss these falls together.
4. **Adding data to the sampler did not help** (adlneg 0.000). **Changes that make the model fire more easily** (rshift, jepaplus) add ADL false alarms faster than they add recall. Stacking all four changes was worse than any one of them.

## 2. What similar problems did

| Problem in the literature | Method | What we take |
|---|---|---|
| Domain generalisation to unseen domains, where ERM ends in sharp minima ([SWAD, Cha et al., NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/bcb41ccdc4363c6848a1d760f26c28a0-Abstract.html)) | Dense weight averaging (every iteration), starting before overfitting | Our test subjects are unseen domains, and the frame loss memorises by epoch 5: average earlier and more densely |
| Flat-minima optimisers ([Kaddour et al., NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/69b5534586d6c035a96b49c86dbeece8-Abstract.html)) | SAM and SWA find different flat regions. Averaging SAM iterates (WASAM) gave the flattest minima | SAM + our SWA, as one ablation |
| Speech separation on small corpora ([Wavesplit, Zeghidour & Grangier](https://arxiv.org/pdf/2002.08933); [MixIT](https://arxiv.org/pdf/2006.12701)) | **Dynamic mixing**: new training mixtures on the fly. Gains of about 1 dB on small sets such as WSJ0-2mix, less on large ones | Our streams are synthetic concatenations of segments. Re-synthesising training streams on the fly is the same idea and targets lead-in, context and join shortcuts directly |
| Sound event detection on synthetic soundscapes ([DCASE task 4 / DESED](https://hal.archives-ouvertes.fr/hal-02160855); [HODGEPODGE](https://arxiv.org/pdf/1907.07398)) | Strong labels come from generated soundscapes, and top systems are ensembles | Same setting as ours: event localisation in generated streams. Ensembles are standard practice |
| [FilterAugment](https://arxiv.org/pdf/2110.03282) (SED) | Random gains on random frequency bands. It beat plain frequency masking (+6.5 % PSDS against +2.1 %) | The range axis is our "frequency". Use band gains along range bins, not shifts (rshift moved the mat positions and hurt) |
| RF fall detection that has to generalise to new people ([Aryokee, Tian et al., IMWUT 2018](https://aryokee.csail.mit.edu/); [SiFall](https://arxiv.org/pdf/2301.03773)) | Many people and environments. Fall-like activities are the main confuser. Two-stage pipelines | Our main confusers are also fall-like ADL (bending while cooking or cleaning). The persistence gate is already our second stage |
| Cross-subject WiFi and radar HAR ([Hassan et al., PLoS ONE 2024](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11025916/); [DATTA](https://arxiv.org/abs/2411.13284v1)) | Domain-adversarial, subject-invariant features | Possible later. With 13 training subjects it is risky, so it is not in this round |
| Ensemble distillation ([Ensemble Distribution Distillation](https://arxiv.org/pdf/1905.00076); [Diversity matters, NeurIPS 2021](https://arxiv.org/abs/2110.14149v1)) | One network trained on the ensemble's soft targets, best on perturbed inputs where the members disagree | Stage 3: put the ensemble into one network, using training streams and remixed windows only (not transductive, unlike Amir's version) |
| Amir's pipeline (teammate_pipeline_20261013) | SWA, a 4-network output ensemble (0.951), physics-verified distillation (0.953) | Confirms that above 0.95 on this val set comes from removing variance |

## 3. The ablation

Every variant goes **on top of rdrop** and gets **3 seeds (0, 1, 2)**. Each run is compared with rdrop
of the same seed. Only stage 2 combines changes, and only in pairs.

### Stage 0: no new code (≈ 10 runs, ≈ 7 h on ABC27)

| spec | change | targets |
|---|---|---|
| `rdrop` seed 2 | completes the baseline to 3 seeds (s0/s1 are in tag 20261015-122938) | the reference |
| `rdrop30` | radar dropout 0.3 | P2_8 Chute04 recall: a fall changes mainly one radar (profile-step share 0.99, against 0.50 at a join) |
| `rdrop+swa5` | weight averaging over epochs 6–30 instead of 11–30 | variance (SWAD) |
| `rdrop+gain` | per-radar gain 0.1 and noise 0.05 | subject shift, the cheapest test |

The stage 0 and stage 1 runs go in one queue (section 3a).

Free (scoring only): ensembles of the existing SWA runs, including rdrop s0+s1+JEPA s0–s2, with
`tools/amir_postprocess.py --ensemble`. These tell us how much is left once variance is removed.

### Stage 1: new code (≈ 12 runs, ≈ 10 h; SAM costs about 1.6×)

| spec | change | targets | code |
|---|---|---|---|
| **`remix`** (main bet) | **On-the-fly stream re-synthesis.** In 50 % of training windows, rebuild the window from published training-stream segments (`streams_composition.csv`): a random previous segment cut at a random point (ADL 70 %, fall 15 %, Autre/Empty 15 %), then the fall or no-fall segment with a fresh lead-in of U[1, 8] s, then a random next segment, joined by a 0.2 s linear crossfade as the generator does. Labels move with their segment. Physics features are recomputed on the composed window. | short lead-in (the P2_8/P2_9 Chute04 misses), fall-after-fall contexts, join and ADL false alarms, memorisation (dynamic mixing) | `data/remix.py` + a hook in `WindowDataset`. Segments are cut inside their interior, so no old crossfade is reused |
| `rdrop+dswa` | Dense averaging: a snapshot every 50 steps from epoch 4 on | variance, compared with swa5 | `WeightAveraging(every_n_steps=50)` |
| `rdrop+sam` | SAM ρ 0.05, with the same SWA (WASAM) | flatter minima, which should mean less seed variance | `model.sam: {rho, clip}`, a manual-optimisation step in `module.py` |
| `rdrop+band` | 2–5 random bands of range bins per radar, each with a gain of ±3 dB (linear FilterAugment), positions unchanged | subject shift without moving the mat | `band_gain_db` / `band_count` in `_augment` |

Offline, no training: fix the refinement span when a clip shorter than 2 s sits in front of the fall
recording (val_01 1436.86). Decide it on the training streams first, then apply it once to val.

**Rule check for remix:** it reads only the published training streams and the composition CSV that ships
with them. It does not use the `segments/` arrays, `build_streams.py` or the LIARA sources.

### 3a. The stage 0 + 1 queue

Implemented on 2026-10-16. Every overlay is covered by `tests/test_falljepa.py` (it fits, averages and
validates on the synthetic release), and `test_remix_moves_labels_with_their_signal` checks that the signal
under every remixed fall label is that fall's signal in its training stream. Augment keys are now checked,
so a typo fails instead of being ignored. One queue on ABC27, one tag:

```bash
cd /home/abc2027/mehdi_ap
TAG=<tag> BASE=configs/uwb/AP.yaml GROUP=UWB/AP CKPT=swa GPUS=0 RELEASE=/home/abc2027/dataset/abc2027_uwb_train_v1 SEEDS="2" tools/run_ablations.sh rdrop
TAG=<tag> BASE=configs/uwb/AP.yaml GROUP=UWB/AP CKPT=swa GPUS=0 RELEASE=/home/abc2027/dataset/abc2027_uwb_train_v1 SEEDS="0 1 2" tools/run_ablations.sh rdrop+remix rdrop30 rdrop+swa5 rdrop+dswa rdrop+band rdrop+gain rdrop+sam
```

That is 22 runs. The two cheap seeds of rdrop already exist (tag 20261015-122938), so pair them by seed across the two tags.

### Stage 2: combine (≈ 10 runs)

Take the two best stage-0/1 changes that pass the rule below. Run them alone and as a pair,
**5 seeds each**, then score the **5-seed SWA output ensemble** of the winner. That ensemble is the
candidate for > 0.95.

### Stage 3 (optional): distil the ensemble (≈ 3 runs)

Train one network on the stage-2 ensemble's soft P(fall), on training windows and remixed windows, for 30 epochs
with SWA. It is not transductive (no val/test streams), so it needs no organiser approval.

## 4. How to decide

- **Keep a change** when its mean paired Δ SWA F1 over 3 seeds is ≥ +0.010, it is positive on at least 2 of 3 seeds,
  the mean of epochs 11–30 is not lower, and errors outside the artefact (FN + FP at θ .3) do not rise.
- **Stability**, reported for every variant:
  - seed SD of SWA F1 (now ≈ 0.025)
  - epoch SD over 11–30 (now 0.04–0.06)
  - **fall agreement**: falls hit by all seeds, out of 65 fixable
- For analysis only (never as the score), also report F1 without the two artefact falls, which removes a constant −0.03.
- Never select by best epoch, and never pick seeds.

## 5. What to expect

- Single SWA model: rdrop is 0.913 now. Realistically 0.92–0.94 after stages 0–1. Above 0.95 for one network
  needs the Chute04 recall fixed **and** at most 1–2 false alarms, which is unlikely without distillation.
- Above 0.95 most likely comes from the stage-2 **5-seed ensemble**. The 6-model ensemble already has only one false alarm outside the artefact.
  What it lacks is recall on P2_8 Chute04, and only remix / rdrop30 can supply that, because ensembling does not fix shared misses.
- Left out on purpose: rshift and jepaplus (they add ADL false alarms), stacking several changes at once (AP2: −0.016),
  per-radar encoders (early fusion stays), the stillness gate (loses the falls of subjects who move after falling), and best-epoch numbers.
