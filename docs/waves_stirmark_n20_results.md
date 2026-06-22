# WAVES-Core and StirMark-Compatible Pilot Results

Run date: 2026-06-22  
GPU: NVIDIA RTX 3090  
Samples: 20 fixed-key COCO images per method  
Codebook: exhaustive 2,048-key identification

## Scope

The pilot evaluates Tree-Ring, METR, RingID, HSFW-HSTR, and HSFW-HSQR with
16 WAVES-Core distortion conditions and eight StirMark-compatible geometric
conditions. The WAVES strengths are normalized values 0.4 and 0.8. At 0.8,
the geometric settings are 36-degree rotation and random resized crop retaining
60% of the image area.

The StirMark-compatible transforms use published StirMark 4 profile values in
Python. They are not binary-identical outputs from the legacy StirMark program.

## Clean Baseline

All five methods achieved AUC 1.0, TPR@1%FPR 1.0, top-1 identification 1.0,
and 11-bit accuracy 1.0. The attacked failures are therefore not caused by a
clean decoder or codebook mismatch.

## Attack-Family Means

| Method | Suite | AUC | TPR@1%FPR | ID Top-1 | Bit Acc |
|---|---|---:|---:|---:|---:|
| Tree-Ring | WAVES-Core | 0.968 | 0.819 | 0.325 | 0.652 |
| Tree-Ring | StirMark-compatible | 0.963 | 0.856 | 0.288 | 0.644 |
| METR | WAVES-Core | 0.920 | 0.775 | 0.669 | 0.843 |
| METR | StirMark-compatible | 0.884 | 0.750 | 0.487 | 0.745 |
| RingID | WAVES-Core | **0.976** | **0.916** | **0.778** | **0.928** |
| RingID | StirMark-compatible | **1.000** | **1.000** | **0.806** | **0.971** |
| HSFW-HSTR | WAVES-Core | 0.857 | 0.722 | 0.603 | 0.779 |
| HSFW-HSTR | StirMark-compatible | 0.885 | 0.781 | 0.394 | 0.666 |
| HSFW-HSQR | WAVES-Core | 0.934 | 0.759 | 0.666 | 0.834 |
| HSFW-HSQR | StirMark-compatible | 0.885 | 0.631 | 0.425 | 0.710 |

RingID ranks first for both verification and identification. HSQR has higher
identification than HSTR in both suites, while HSTR has higher verification on
the StirMark-compatible track.

## Geometric Stress Cases

The table reports `TPR@1%FPR / ID Top-1`.

| Method | Rotation 36 deg | Random crop-resize, area 0.60 | Center crop 25% | Rescale 75% | Affine shear 0.05 |
|---|---:|---:|---:|---:|---:|
| Tree-Ring | 0.35 / 0.00 | 0.50 / 0.00 | 0.05 / 0.00 | 1.00 / 0.85 | 1.00 / 0.00 |
| METR | 0.00 / 0.00 | 0.05 / 0.00 | 0.60 / 0.00 | 1.00 / 1.00 | 0.10 / 0.00 |
| RingID | **1.00 / 1.00** | 0.35 / 0.00 | **1.00 / 0.00** | 1.00 / 1.00 | 1.00 / 1.00 |
| HSFW-HSTR | 0.00 / 0.00 | 0.10 / 0.00 | 0.10 / 0.00 | 1.00 / 1.00 | 1.00 / 0.10 |
| HSFW-HSQR | 0.10 / 0.00 | 0.15 / 0.00 | 0.45 / 0.00 | 1.00 / 1.00 | 0.35 / 0.00 |

Simple global rescaling is mostly harmless, but crop-resize and cropping break
exact identification. RingID is the clearest case: it retains perfect
verification under 25% center crop while top-1 and top-5 identification both
fall to zero. Its bit accuracy remains 0.818, suggesting that structured payload
evidence survives but is assigned to the wrong radial position or ID bins.

## Main Conclusion

Verification robustness does not imply identification robustness. Tree-Ring
has high family-level verification AUC but weak identification, and RingID can
still detect a cropped watermark when it cannot recover the correct key. A paper
that reports only AUC or TPR would conceal the most relevant deployment failure.

The strongest next research direction is therefore crop/scale synchronization
for RingID-style identification, not a stronger claimed-key verifier. Keep the
rotation-invariant radial payload, estimate crop ratio and offset using an
independent synchronization channel, and decode the payload in scale-normalized,
phase/offset-aware radial coordinates.

## Recommended Confirmatory Experiment

1. Compare original RingID with the synchronization-aware decoder on clean,
   10%/25% center crop, randomized crop-resize, 75% rescale, affine shear, and
   10/36-degree rotation.
2. Increase to at least 200 images and distribute samples across multiple keys.
3. Use top-1, top-5, 11-bit accuracy, AUC, and TPR@1%FPR as separate outcomes.
4. Set a primary target of at least 0.80 top-1 under 25% crop and high
   crop-resize while preserving clean and rotation accuracy.
5. Add bootstrap confidence intervals and paired comparisons using the saved
   per-image arrays.

## Limitations

- With 20 negative samples, TPR at 0.1% or 1% FPR is coarse and is suitable for
  screening, not a final statistical claim.
- The run uses fixed key 0; it does not establish uniform robustness over all
  2,048 IDs.
- This is the distortion subset of WAVES, not its adversarial and regeneration
  attack tracks.
- The StirMark-compatible track reproduces profile values, not the original
  executable's exact pixel outputs.
