# An ML-ready EEG preprocessing pipeline for CNV step-type decoding

**Bottom line.** Your current ERP pipeline is well designed for grand-average CNV analysis but is suboptimal for within-subject single-trial classification on ~60–80 trials. The most important changes are: **(1) keep the 0.1 Hz high-pass — do not move to the ML-default 1 Hz, because that destroys the CNV** (Tanner et al., 2015; Widmann et al., 2015); (2) use the **dual-filter ICA trick** (fit ICA on a 1 Hz copy, apply unmixing back to the 0.1 Hz analysis copy) with **automated component classification via ICLabel**; (3) replace fixed 15% epoch dropping with **Autoreject (local)** so artifactual sensors are interpolated rather than whole trials discarded; (4) **drop the −0.1 to 0 s baseline** (it sits inside the discriminative CNV); (5) classify with **xDAWN-augmented covariance + tangent-space + shrinkage-LDA** as the primary model, which is repeatedly state-of-the-art for small-sample ERP/SCP BCI; and (6) evaluate with **repeated, block-aware nested CV** with permutation-based significance, since chance is not 50% at N≈80. Realistic balanced accuracy target: **60–75%**, per the published CNV/SCP single-trial literature.

The rest of this document expands each decision with citations, code, and a side-by-side comparison.

---

## Executive summary — proposed pipeline (numbered)

1. **Read** the .bdf, drop CMS/DRL, set the BioSemi64 montage, declare EEG channel types.
2. **Line-noise removal** with ZapLine or zero-phase notch at 50/60 Hz (and harmonics if visible) on the continuous 1024 Hz raw.
3. **Automated bad-channel detection and interpolation** with **PyPREP `PrepPipeline`** (RANSAC + deviation + correlation + HF-noise + flatline, inside a robust average-reference loop) — Bigdely-Shamlo et al. 2015 (doi:10.3389/fninf.2015.00016); Robbins et al. 2020 (doi:10.1109/TNSRE.2020.2980223). Log the interpolated channel list.
4. **Filter the analysis copy**: zero-phase FIR Hamming-windowed sinc, **HP 0.1 Hz** (transition 0.1 Hz) and **LP 40 Hz** (transition 10 Hz). This preserves the slow CNV (Tanner et al., 2015; Widmann et al., 2015; Acunzo et al., 2012).
5. **Filter a separate ICA-fit copy**: HP 1.0 Hz, LP 100 Hz, average-referenced (ICLabel training requirement). Optionally apply **ASR (cutoff k=20)** to the ICA copy to remove transient bursts (Mullen et al., 2015; Anders et al., 2020).
6. **Downsample to 256 Hz** after low-pass filtering (anti-alias already in place); resample again to 128 Hz only if you train EEGNet exactly per Lawhern et al. (2018).
7. **ICA** with **Picard (`extended=True, ortho=False`)** — equivalent to extended Infomax but faster and more reproducible (Ablin et al., 2018) — fit on the **full** ICA copy, with **`n_components = n_eeg − n_interpolated − 1`** (rank-aware, not fixed 20). `random_state=97`.
8. **Component classification with ICLabel** via `mne-icalabel` (Pion-Tonachini et al., 2019). Auto-exclude any IC where the argmax label is Muscle/Eye/Heart/LineNoise/ChannelNoise with probability >0.80. Apply the unmixing matrix back to the 0.1–40 Hz analysis copy (linear-operator commutativity; Winkler et al., 2015; Klug & Gramann, 2021).
9. **Epoch** −0.2 to +2.0 s relative to the warning cue; **no baseline correction** in the −0.1 to 0 s window. If you must baseline-correct, use **−200 to 0 ms relative to S1**, which is *outside* the CNV. Better: trial-masked robust detrending applied to continuous data (van Driel et al., 2021).
10. **Autoreject (local)** with `n_interpolate ∈ {1,2,4}`, `consensus` swept by random search, 5-fold inner CV (Jas et al., 2017). This interpolates briefly bad sensors instead of discarding entire trials — critical at N≈70.
11. **Surface Laplacian / CSD** (Kayser–Tenke spherical splines, m=4, λ=1e-5) as the analytic reference (McFarland, 1997; Kayser & Tenke, 2006, 2015). Optional second pipeline branch keeps CAR for the deep-net path.
12. **Channel set**: run two configurations in parallel — (a) **motor ROI** (Cz, FCz, CPz, C1, C2, FC1, FC2, CP1, CP2 ± C3/C4/FC3/FC4/CP3/CP4) for ROI-LDA and ERP-feature models; (b) **all 64 channels** for Riemannian and CNN models that learn spatial filters.
13. **Window**: primary = **late-CNV 1.0–2.0 s** post-S1; secondary = full 0–2 s with **cropped training** (1-s sub-windows, 250 ms stride; Schirrmeister et al., 2017).
14. **Primary classifier**: **xDAWN-covariances → tangent space → shrinkage-LDA** (`pyriemann`), with OAS shrinkage on covariances. Baselines: MDM, FBCSP+sLDA, ShallowConvNet, EEGNet.
15. **Augmentation** (deep-net path only): cropped training → time-shift → mixup → FT-surrogate (Rommel et al., 2022; Lashgari et al., 2020).
16. **CV**: **repeated stratified 5-fold (≥20 repeats) with `GroupKFold` on block ID**, **nested** for hyperparameters, plus a **chronological-fold sanity check** to expose temporal drift (Varoquaux et al., 2017).
17. **Metrics**: balanced accuracy + 95% bootstrap CI, AUC, Cohen's κ, confusion matrix; **permutation test (5,000 shuffles)** with full pipeline refit; report the binomial chance upper bound (≈60% at N=80; Müller-Putz et al., 2008; Combrisson & Jerbi, 2015).

---

## Side-by-side comparison

| # | Current step (ERP-optimised) | Proposed change (ML-optimised) | Reason |
|---|---|---|---|
| 1 | Load .bdf, set montage | Same | Fine. |
| 2 | Visually mark bad channels, spline-interpolate | **PyPREP `PrepPipeline` (RANSAC + 4 other detectors, robust ref)** | Reproducible; one of the only two preprocessing steps Delorme (2023) confirms always helps; removes researcher degrees of freedom (Robbins et al., 2020). |
| 3 | CAR with projection | **Provisional CAR for ICA; final CSD (spherical-spline Laplacian) for analysis** | CSD is reference-free, sharpens local sources, raises single-trial motor SNR (McFarland, 1997, 2015; Kayser & Tenke, 2006, 2015). |
| 4 | Notch 60/120/180/240/300/360 Hz | **ZapLine or single notch at line + harmonics ≤40 Hz** (most are above the 40 Hz LP anyway) | LP filter already removes harmonics; ZapLine avoids notch-filter ringing (de Cheveigné, 2020). |
| 5a | ICA on 50 s crop, 1 Hz HP, fastICA, n=20, manual rejection | **ICA on full data, 1 Hz HP copy, Picard `extended=True`, n=rank−1, ICLabel auto-rejection (>0.8)** | 50 s ≪ 30·channels² minimum (Klug & Gramann, 2021). Fixed n=20 truncates real sources. Picard is faster and ICLabel-compatible (Ablin et al., 2018). ICLabel reproducibility is far higher than manual review (Pion-Tonachini et al., 2019). |
| 5b | Apply ICA back to notch-only data | **Apply unmixing back to 0.1–40 Hz analysis copy** (linear-operator commutativity) | Winkler et al. (2015); Klug & Gramann (2021) validated this dual-filter approach. Preserves CNV while ICA sees a stationary ≥1 Hz signal. |
| 5c | (none) | **Add ASR with k=20 between bad-channel and ICA** | Removes transient bursts that would otherwise dominate ICA variance (Mullen et al., 2015; Chang et al., 2020; Anders et al., 2020 specifically for motor tasks). |
| 6 | HP 0.1 Hz / LP 40 Hz | **Keep HP 0.1 Hz / LP 40 Hz**, but as zero-phase FIR Hamming-windowed sinc with documented transition bands | Crucially **do NOT move to 1 Hz HP** — that destroys the CNV (Tanner et al., 2015; Acunzo et al., 2012). LP 40 Hz preserves mu/beta ERD; gamma at scalp is largely myogenic (Whitham et al., 2007). |
| 7 | (1024 Hz throughout) | **Downsample to 256 Hz after LP** (or 128 Hz for native EEGNet) | 4× memory and training-time saving with no information loss for ≤40 Hz signals (Picton et al., 2000); standard in EEGNet (Lawhern et al., 2018) and ShallowConvNet (Schirrmeister et al., 2017). |
| 8 | Find events: 96 preceded by 256/512 | Same | Fine. |
| 9 | Epoch −0.1 to +2.0 s, baseline (−0.1, 0) | **Epoch −0.2 to +2.0 s; no baseline correction (or pre-S1 baseline only)** | A −0.1 to 0 s baseline relative to S1 is fine; a −0.1 to 0 s baseline relative to **Go** sits *inside* the discriminative CNV and removes class-relevant drift (Alday, 2019; van Driel et al., 2021; Delorme, 2023). |
| 10 | Drop noisiest 15% by amplitude | **Autoreject (local), CV-tuned per channel, with epoch interpolation** | Interpolates instead of discards — critical at N≈70 (Jas et al., 2017). |
| 11 | (no z-scoring) | **OAS-shrinkage covariance for Riemannian path; per-channel exponential moving standardization for CNN path** | Sample covariance is rank-deficient at N=70, 64 ch (Blankertz et al., 2011); exponential standardization is the Braindecode standard (Schirrmeister et al., 2017). Fit on train fold only — no leakage. |
| 12 | Average ERP at Cz, 8 × 250 ms bins, eLORETA | **Multiple parallel ML pipelines: (i) xDAWN-cov + TS + sLDA (primary); (ii) MDM; (iii) FBCSP+sLDA; (iv) ShallowConvNet; (v) EEGNet** | xDAWN+TS+LDA is repeatedly SOTA for small-sample ERP/SCP BCI (Barachant & Congedo, 2014; Lotte et al., 2018); deep nets typically need hundreds–thousands of trials (Roy et al., 2019). |
| 13 | (single-window analysis) | **Late-CNV 1.0–2.0 s primary; cropped sub-windows for augmentation** | Late CNV carries motor preparation (Brunia & van Boxtel, 2001); cropped training boosts deep-net accuracy on small data (Schirrmeister et al., 2017). |
| 14 | (no CV reported) | **Repeated stratified `GroupKFold` (5-fold, ≥20 repeats), nested for HPs, plus chronological sanity fold** | Single LOO/k-fold severely underestimates variance (Varoquaux et al., 2017); blocks introduce temporal autocorrelation. |
| 15 | (no significance testing) | **Permutation tests (≥5,000) + binomial chance bounds** | Chance is ~60% at N=80 for a 2-class task (Müller-Putz et al., 2008; Combrisson & Jerbi, 2015). |

---

## Detailed discussion of the 15 questions

### 1. Filter bandpass — **keep 0.1 Hz HP, 40 Hz LP**, but as a properly specified zero-phase FIR

The temptation in ML pipelines is to move the HP to 1 Hz because it stabilizes ICA and removes drift. **For a CNV paradigm this is wrong.** Tanner, Morgan-Short & Luck (2015, *Psychophysiology* 52:997, doi:10.1111/psyp.12437) and Acunzo, MacKenzie & van Rossum (2012, *J Neurosci Methods* 209:212, doi:10.1016/j.jneumeth.2012.06.011) both showed that HP cutoffs ≥0.3 Hz inject **artifactual opposite-polarity deflections** into slow components — they create the very class differences the classifier is supposed to learn. Widmann, Schröger & Maess (2015, *J Neurosci Methods* 250:34, doi:10.1016/j.jneumeth.2014.08.002) give the implementation rule: zero-phase Hamming-windowed sinc FIR, transition bandwidth ≤ passband edge, order ≈ 3.3·fs/Δf. For movement-related cortical potentials specifically, Jochumsen et al. (2015, *J Neural Eng* 12:056003) and the broader MRCP literature converge on **0.05–5 Hz for the SCP stream**.

Low-pass: **40 Hz** is the right ceiling. It preserves mu (8–13 Hz) and beta (13–30 Hz) ERD that may carry supplementary direction information, while attenuating myogenic high-frequency activity that dominates "scalp gamma" (Whitham et al., 2007, *Clin Neurophysiol* 118:1877). 30 Hz is too aggressive; 80–100 Hz adds mostly muscle.

```python
raw.filter(l_freq=0.1, h_freq=40.0, method="fir", phase="zero",
           fir_design="firwin", l_trans_bandwidth=0.1, h_trans_bandwidth=10.0)
```

### 2. Re-referencing — **CSD (surface Laplacian) for the analytic path, CAR for the ICA path**

McFarland, McCane, David & Wolpaw (1997, doi:10.1016/S0013-4694(97)00022-2) showed the surface Laplacian outperformed CAR and ear references for sensorimotor BCI; Kayser & Tenke (2015, *Int J Psychophysiol* 97:189, doi:10.1016/j.ijpsycho.2015.04.012) recommend spherical-spline CSD as a default, with m=4 and λ=1e-5. CSD is reference-free, sharpens local sources, and is particularly beneficial when sources are focal — exactly the case for foot M1 around Cz (Pfurtscheller et al., 2006). Mastoid reference is deprecated for slow potentials because it biases CNV amplitude (Joyce & Rossion, 2005). REST (Yao, 2001) is theoretically attractive but empirically equivalent to CAR with 64 channels (Liu et al., 2018). Use CAR provisionally for ICA fitting (ICLabel was trained on average-referenced data — Pion-Tonachini et al., 2019), then transform to CSD for downstream features.

### 3. Automated bad-channel detection — **PyPREP**

Visual inspection injects researcher subjectivity that is incompatible with a reproducible ML pipeline. **PyPREP** (a Python port of the PREP pipeline; Bigdely-Shamlo et al., 2015, *Front Neuroinform* 9:16, doi:10.3389/fninf.2015.00016) combines five detectors (flatline, deviation, HF-noise SNR, neighbor correlation, RANSAC) inside an iterative robust-reference loop — bad channels do not contaminate the reference used to detect them. Robbins et al. (2020, *IEEE TNSRE* 28:1081) and Delorme (2023, *Sci Rep* 13:2372) both confirm that bad-channel interpolation is one of only two preprocessing steps that *consistently* improves data quality. Expect 1–4 channels flagged per session on a clean BioSemi recording.

### 4. ICA — automated, on full data, rank-aware, applied across filter copies

Replace every weak link in the current ICA step:
- **Crop length**: a 50 s window is far below the ~30·n_chan² ≈ 120 s minimum for stable decomposition (Klug & Gramann, 2021, *Eur J Neurosci* 54:8406, doi:10.1111/ejn.14992). Fit on the **full continuous recording**.
- **HP for ICA**: keep the **1 Hz copy** — Winkler et al. (2015) and Klug & Gramann (2021) both confirm 1–2 Hz is the optimum for ICA SNR. Apply the unmixing back to the 0.1–40 Hz analysis copy; this works because filtering and the ICA unmixing matrix commute as linear operators.
- **Algorithm**: **Picard with `extended=True, ortho=False`** (Ablin et al., 2018, *IEEE TSP* 66:4040, doi:10.1109/TSP.2018.2844203) — mathematically equivalent to extended Infomax (so ICLabel-compatible) but 5–10× faster and more reproducible.
- **n_components**: **`n_eeg − n_interpolated − 1`**, not fixed at 20. After CAR (rank −1) and *k* interpolations (rank −*k*), the data rank is 64−*k*−1. Forcing n=20 truncates real brain sources.
- **Component classification**: replace manual review with **ICLabel** (Pion-Tonachini, Kreutz-Delgado & Makeig, 2019, *NeuroImage* 198:181, doi:10.1016/j.neuroimage.2019.05.026), accessed via the `mne-icalabel` package. ICLabel was trained on >6,000 expert-labeled ICs and outputs continuous probabilities for seven classes (Brain, Muscle, Eye, Heart, Line Noise, Channel Noise, Other). MARA, ADJUST and SASICA are older and less robust, especially for muscle separation in motor tasks.
- **ASR**: optionally insert ASR with a conservative cutoff `k=20` between bad-channel interpolation and ICA, so transient bursts do not dominate the ICA variance (Mullen et al., 2015, *IEEE TBME* 62:2553; Anders et al., 2020 specifically recommend k=20 for motor tasks).

```python
from mne.preprocessing import ICA
from mne_icalabel import label_components

raw_ica = raw_clean.copy().filter(1.0, 100.0, fir_design="firwin")
raw_ica.set_eeg_reference("average", projection=False)

n_comp = raw_ica.info["nchan"] - len(raw_ica.info["bads"]) - 1
ica = ICA(n_components=n_comp, method="picard",
          fit_params=dict(ortho=False, extended=True),
          max_iter="auto", random_state=97)
ica.fit(raw_ica)

ic = label_components(raw_ica, ica, method="iclabel")
exclude = [i for i, lbl in enumerate(ic["labels"])
           if lbl not in ("brain", "other") and ic["y_pred_proba"][i] > 0.80]
ica.exclude = exclude
ica.apply(raw_clean)   # raw_clean is bandpassed 0.1–40 Hz
```

### 5. Epoch-level rejection — **Autoreject (local)**

The fixed-15% rule is statistically inferior at small N. Jas, Engemann, Bekhti, Raimondo & Gramfort (2017, *NeuroImage* 159:417, doi:10.1016/j.neuroimage.2017.06.030) introduced **Autoreject**, which learns per-channel peak-to-peak thresholds by cross-validation and tunes two hyperparameters (ρ = max channels to interpolate before dropping, κ = consensus fraction) by Bayesian/random search. The **local** variant interpolates a small number of bad sensors *within* an epoch instead of discarding the trial — preserving data volume, which is decisive at 60–80 trials. Engemann et al. (2022) report ~3–5 percentage-point ML accuracy gains attributable to Autoreject on age decoding; no head-to-head benchmark exists for CNV decoding, so this is worth piloting empirically.

```python
from autoreject import AutoReject
ar = AutoReject(n_interpolate=[1,2,4], consensus=np.linspace(0.2,1.0,9),
                cv=5, random_state=97, n_jobs=-1, thresh_method="random_search")
epochs_clean, log = ar.fit_transform(epochs, return_log=True)
```

### 6. Baseline correction — **drop it (or move it pre-S1)**

This is the most counterintuitive change. The baseline-correction debate has converged on three points relevant to ML:
- Alday (2019, *Psychophysiology* 56:e13451) showed pre-stim baseline subtraction *reduces* SNR by injecting baseline noise into every post-stim sample.
- van Driel, Olivers & Fahrenfort (2021, *J Neurosci Methods* 352:109080) explicitly demonstrated that **HP filtering plus baseline correction creates spurious decoding artifacts in MVPA** — patterns leak into "activity-silent" windows. They recommend **trial-masked robust detrending** (mask events; fit and subtract a polynomial trend on the masked continuous data).
- Delorme (2023, *Sci Rep* 13:2372) found advanced baseline-removal methods were significantly *detrimental* across three open ERP datasets.

A −0.1 to 0 s baseline relative to **S1** is acceptable (it sits before the CNV). A baseline relative to **Go** sits *inside* the CNV and would subtract precisely the discriminative slow drift between conditions. The current pipeline as stated uses (-0.1, 0) relative to the warning cue — that's fine for ERP. For ML, **either drop baseline correction entirely and rely on robust detrending, or use a clearly pre-S1 window like (−0.2, 0)**. Per-trial z-scoring is a reasonable substitute for deep nets but should not be applied to the time-domain inputs of Riemannian classifiers (it distorts covariance scaling).

### 7. Downsampling — **1024 → 256 Hz** (or 128 Hz for native EEGNet)

After the 40 Hz LP, Nyquist is satisfied at any rate ≥80 Hz. 256 Hz gives 4× compute savings with no information loss; EEGNet was published at 128 Hz, ShallowConvNet at 250 Hz (Lawhern et al., 2018; Schirrmeister et al., 2017), and MOABB benchmarks (Jayaram & Barachant, 2018) show no accuracy loss when downsampling motor-imagery data from 1000→250 Hz. Use polyphase filtering (`raw.resample(256)` in MNE applies an internal anti-alias filter).

### 8. Channel selection — **run two configurations**

The foot motor area is medial, so for foot tasks the discriminative signal is concentrated at **Cz, FCz, CPz, C1, C2, FC1, FC2, CP1, CP2** (Pfurtscheller et al., 2006, doi:10.1016/j.neuroimage.2005.12.003; Solis-Escalante et al., 2012). Channel-selection methods (Lal et al., 2004, doi:10.1109/TBME.2004.827827; Arvaneh et al., 2011, doi:10.1109/TBME.2011.2131142) consistently recover this medial cluster and show 8–16 channels suffice for motor-task BCI without accuracy loss.

CSP and Riemannian methods generally benefit from more channels (more spatial degrees of freedom), but become unstable when channels approach trials — at N=70 with 64 channels you are in the regime where regularized CSP (Lotte & Guan, 2011, doi:10.1109/TBME.2010.2082539) or OAS-shrinkage covariance is essential. Deep nets benefit from all 64 channels but are underpowered at this N regardless.

**Recommendation**: report both (a) ROI = 13–17 medial channels for ROI-LDA and ERP-feature models, and (b) all 64 channels for Riemannian and CNN models. This is also informative for interpretation.

### 9. Epoch window — **late CNV 1.0–2.0 s**, with cropped training as augmentation

The CNV has two functional phases (Brunia & van Boxtel, 2001, doi:10.1016/S0167-8760(00)00181-5; Birbaumer et al., 1990, *Physiol Rev* 70:1): early CNV (~0–800 ms post-S1, orienting/attention, parieto-occipital and frontal) and late CNV (~1000 ms post-S1 to S2, motor preparation, focal at Cz/FCz). **The motor-preparation information lives in the late CNV.** Garipelli, Chavarriaga & Millán (2013, *J Neural Eng* 10:036014, doi:10.1088/1741-2560/10/3/036014) confirmed peak single-trial decoding in the late preparation window for SCP-based anticipation.

Use **1.0–2.0 s as the primary window**, and as secondary analyses (i) the full 0–2 s with **cropped training** (Schirrmeister et al., 2017 — slide 1-s sub-windows with 250 ms stride and average predictions at test time), and (ii) a **sliding-window AUC time-course** to localize when discriminative information emerges (interpretability figure).

### 10. Normalization — **type-specific, fit on train fold only**

For Riemannian classifiers: use **OAS-shrinkage covariance** (`pyriemann.estimation.Covariances("oas")` or `XdawnCovariances(estimator="oas")`). Sample covariance is rank-deficient at N=70 with 64 channels; OAS shrinkage is what Blankertz et al. (2011, doi:10.1016/j.neuroimage.2010.06.048) demonstrated is essential. Do *not* z-score time-domain channels into Riemannian pipelines — it alters covariance scaling.

For deep nets: per-channel **exponential moving standardization** (Schirrmeister et al., 2017; Braindecode default `init_block_size=1000, factor=1e-3`), or simpler per-recording per-channel z-score using statistics computed on the training fold only. Lawhern et al. (2018) used per-trial mean-centering. Critically: **fit any scaler / xDAWN filter / CSP filter / class-mean ONLY on the training fold of each CV split**. In sklearn, put them inside a `Pipeline` and pass the whole pipeline to `cross_val_score` — this is the single most common leakage bug in EEG-ML papers.

### 11. Data augmentation — **for the deep-net path only, ranked**

Two systematic studies anchor this. Lashgari, Maoz & Liang (2020, *J Neurosci Methods* 346:108885, doi:10.1016/j.jneumeth.2020.108885) reviewed augmentations for DL-EEG; Rommel, Paillard, Moreau & Gramfort (2022, *J Neural Eng* 19:066020, doi:10.1088/1741-2552/aca220) benchmarked 13 augmentations on EEGNet/ShallowConvNet and found that for motor BCI tasks the consistently helpful ones are:

1. **Cropped training** (Schirrmeister et al., 2017) — slide ~1.5 s windows with 100–250 ms stride; ~6–10× expansion. Best single technique for raw-signal CNNs. Native to Braindecode.
2. **Time-shift / temporal jitter (±100 ms)** — cheap, almost always positive.
3. **Mixup (α=0.2–0.4)** — Zhang et al. (2018, ICLR) — strong in low-data regimes.
4. **FT-surrogate / frequency masking** — phase-randomized surrogates preserve spectrum.
5. **Gaussian noise** at σ ≈ 0.1·channel_std — modest regularizer.
6. **Channel dropout** (10–20% of channels zeroed) — useful when there is hemispheric structure.
7. **Riemannian-domain augmentation** — class-conditional Gaussian sampling in the tangent space; recommended only if the classifier is itself Riemannian.

GAN/VAE-based augmentation is **not recommended** at N=60–80 — generators need more data to train than the classifier (Lashgari et al., 2020). SMOTE only applies to non-trivial imbalance (>60/40) and only in the tangent space, not raw signal space (Chawla et al., 2002). Apply augmentation **only to the training fold** — augmenting the validation/test fold inflates accuracy.

### 12. Feature extraction — **xDAWN-augmented covariance is the right primary feature for a CNV task**

Most motor-BCI features (CSP, FBCSP) extract band-power and **miss the slow CNV by construction** because they operate on bands that exclude DC. The Riemannian-geometry recipe for ERP/SCP classification uses **xDAWN spatial filtering** to estimate class-prototype responses, concatenates the prototype to each trial to form a "super-trial," computes the covariance, and projects to the tangent space (Rivet et al., 2009, doi:10.1109/TBME.2009.2012869; Barachant & Congedo, 2014, ESANN; Cecotti, 2017). This exposes time-locked slow-wave structure inside a Riemannian classifier. `pyriemann.estimation.XdawnCovariances` implements this directly. It has been used successfully on P300, ErrP, MRCP and VEP tasks.

A productive ensemble strategy is to concatenate three feature blocks: (i) tangent-space vectors from xDAWN-covariances (ERP), (ii) tangent-space vectors from broadband Covariances (oscillatory), and (iii) FBCSP log-variance features in mu (8–13 Hz) and beta (13–30 Hz) — and feed to shrinkage-LDA. As an interpretable benchmark, also report **CNV mean amplitude in eight 250 ms bins on the 9 motor channels** (= 72 features) classified by shrinkage-LDA — this is what a neurophysiologist would compute by hand.

### 13. Class imbalance — minor; handle with class weighting

After Autoreject the 40/40 nominal balance may drift to ~35/42. Handle with `class_weight="balanced"` in sklearn LDA/LR/SVM, `weight=` in PyTorch loss, or pyRiemann's `sample_weight`. Stratified CV is mandatory. SMOTE-in-tangent-space is reserved for >60/40 drift.

### 14. Cross-validation — **repeated stratified GroupKFold, nested, with chronological sanity check**

Varoquaux, Raamana, Engemann, Hoyos-Idrobo, Schwartz & Thirion (2017, *NeuroImage* 145:166, doi:10.1016/j.neuroimage.2016.10.038) is the foundational reference. Key conclusions: **leave-one-out is biased and high-variance** at small N; standard error across folds **strongly underestimates** true variance; default hyperparameters often beat tuned ones in low-data regimes; **repeated random stratified k-fold (5-fold, 50–100 repeats)** is the most reliable estimator.

For your data:
- Use **stratified 5-fold, repeated ≥20 times** as the primary estimator.
- If trials are recorded in blocks, use **`StratifiedGroupKFold` with block ID as the group** so an entire block is in either train or test — random k-fold across blocks leaks block-level state.
- Use **nested CV** (inner 3- or 4-fold) for any hyperparameter tuning — non-nested estimates are optimistically biased (Vabalas et al., 2019, *PLoS ONE* 14:e0224365).
- Add a **chronological 5-fold** sanity check (each fold = a contiguous chunk of trials). If accuracy drops markedly under chronological vs random splits, you have temporal drift, and the chronological number is the honest one.

### 15. ICA placement — **after notch and bad-channel interpolation, on a separately-filtered copy**

The current order (notch → ICA → bandpass) is close to optimal but should be refined to: **notch → bad-channel interpolate → CAR → make 1 Hz-HP copy → fit ICA on copy → ICLabel → apply unmixing to the 0.1–40 Hz analysis copy → CSD → epoch → Autoreject (local)**. This is the dual-filter approach validated by Winkler et al. (2015, doi:10.1109/EMBC.2015.7319296) and Klug & Gramann (2021). It avoids the "double-dip" pitfall — fitting ICA on data containing the slow CNV makes the decomposition noisier; fitting it on 1 Hz data and using *that* would destroy the CNV; the dual-filter solution gets both right.

---

## Recommended classical-ML vs deep-learning verdict

**Use classical Riemannian methods as the primary models; treat deep nets as exploratory baselines.** This is grounded in two pieces of evidence:

- **Lotte et al. (2018)** "A review of classification algorithms for EEG-based BCIs: a 10-year update" (*J Neural Eng* 15:031005, doi:10.1088/1741-2552/aab2f2) explicitly conclude that **Riemannian-geometry methods reach state-of-the-art on multiple BCI problems**, that **shrinkage LDA is particularly useful for small training samples**, and that **deep learning has not yet shown convincing improvement over state-of-the-art BCI methods**.
- **Roy et al. (2019)** "Deep learning-based electroencephalography analysis" (*J Neural Eng* 16:051001, doi:10.1088/1741-2552/ab260c) reviewed 154 DL-EEG papers and found a median DL gain of only 5.4% over classical baselines, with successful within-subject DL studies typically using hundreds–thousands of segments per subject.

At N≈60–80 trials and 64 channels, the noise floor for a deep net is high. The recommended primary classifier is therefore **xDAWN-covariances → tangent space → shrinkage-LDA**, with secondary baselines: MDM, FBCSP+sLDA, ShallowConvNet (with cropped training + augmentation), EEGNet at 128 Hz. Realistic expectation per the published CNV/SCP single-trial literature (Garipelli et al., 2013; Lew et al., 2012, 2014; Velu & de Sa, 2013): **balanced accuracy 60–75%** with ±10% per-fold CIs.

```python
# Primary pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace

clf = Pipeline([
    ("xdawn_cov", XdawnCovariances(nfilter=4, estimator="oas",
                                   xdawn_estimator="oas")),
    ("tangent",   TangentSpace(metric="riemann")),
    ("scaler",    StandardScaler()),
    ("lda",       LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
])

from sklearn.model_selection import (RepeatedStratifiedKFold,
                                     permutation_test_score)
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=0)
score, perm, p = permutation_test_score(
    clf, X, y, cv=cv, n_permutations=5000,
    scoring="balanced_accuracy", n_jobs=-1)
print(f"Balanced acc = {score:.3f}, permutation p = {p:.4f}")
```

## Evaluation metrics and CV (consolidated recipe)

Report, per subject and per model: **balanced accuracy with 95% bootstrap CI** (50 CV repeats); **AUC-ROC with bootstrap CI**; **Cohen's κ**; **confusion matrix**; **permutation p-value (5,000 shuffles, full pipeline refit per shuffle)**; the **binomial 95% upper-chance bound** (≈61.7% at N=60, ≈60% at N=80; Müller-Putz et al., 2008; Combrisson & Jerbi, 2015, doi:10.1016/j.jneumeth.2015.01.010). At the group level, run a **Wilcoxon signed-rank test** of per-subject balanced accuracies against chance.

## Suggested figures for the document

A pipeline flowchart (current vs. proposed, two columns); a filter-response plot showing 0.1 Hz vs 1 Hz HP applied to a simulated CNV (the 1 Hz curve will visibly destroy the slow ramp); a schematic CNV waveform at Cz with annotated S1, early CNV, late CNV, S2; a topomap triptych of CNV at baseline / early / late windows; an ICLabel example panel (kept brain IC vs rejected muscle IC with topomap, ERP-image, PSD, and probability bars); an Autoreject before/after epochs×channels heatmap; a learning curve of balanced accuracy vs n_train with shaded 95% CI; a sliding-window AUC time-course localising when discriminative information emerges (analogous to Garipelli et al., 2013 Fig 4–5; King & Dehaene, 2014); a CSP/ConvNet spatial-filter visualisation expected to peak at Cz/FCz/CPz; and a confusion-matrix template with per-fold variability.

## Conclusion

The fundamental insight is that **CNV decoding is an ERP problem dressed up as a BCI problem**: the signal of interest is a slow time-locked potential, not a band-power modulation, so classical motor-BCI defaults (1 Hz HP, CSP, EEGNet at 4–40 Hz) actively damage the discriminative signal. The ERP-style filtering of the current pipeline is therefore *correct* for ML — what needs to change is the artifact-handling stack (manual → automated, drop-15% → Autoreject local), the analytic reference (CAR → CSD), the baseline-correction policy (drop or move pre-S1), and the classifier family (group-mean ERP → xDAWN-tangent-space-LDA). The expected performance ceiling is modest — 60–75% balanced accuracy with wide CIs — so the most important methodological commitment is honest evaluation: nested, repeated, block-aware CV; permutation tests; and reporting against the small-N binomial chance bound rather than 50%. With these changes the pipeline becomes reproducible end-to-end, leakage-free, and aligned with peer-reviewed best practice for small-sample within-subject EEG decoding.