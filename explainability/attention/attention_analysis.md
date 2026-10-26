# Multimodal Attention Analysis Report

**Date:** 2026-10-07 22:53:57

## Architecture

| Parameter | Value |
|---|---|
| Number of attention heads | 4 |
| Number of modality tokens | 2 |
| Token dimension | 256 |
| Attention matrix shape | (Batch, 2, 2) |
| Head aggregation | Averaged across 4 heads (PyTorch MHA default) |

## Attention Matrix Layout

The 2x2 attention matrix represents the following interactions:

|  | Attends TO: IMAGE | Attends TO: METADATA |
|---|---|---|
| **Query: IMAGE** | IMAGE → IMAGE (self) | IMAGE → METADATA (cross) |
| **Query: METADATA** | METADATA → IMAGE (cross) | METADATA → METADATA (self) |

## Quantitative Results (n=500 test samples)

| Interaction | Mean Attention Weight |
|---|---|
| IMAGE → IMAGE | 0.5364 |
| IMAGE → METADATA | 0.4636 |
| METADATA → IMAGE | 0.7842 |
| METADATA → METADATA | 0.2158 |

**Stronger cross-modal direction:** META->IMG

## Interpretation

The attention mechanism provides evidence of representation interaction between the image and metadata modalities.

- The IMAGE token predominantly attends to itself (0.536) and to METADATA (0.464).
- The METADATA token attends to itself (0.216) and to IMAGE (0.784).
- METADATA -> IMAGE cross-modal attention is stronger than IMAGE -> METADATA, suggesting the metadata representation integrates image information more strongly.

## Scientific Limitations

> **IMPORTANT:** Attention weights SHOULD NOT automatically be interpreted as feature importance or causal evidence.
>
> The attention mechanism provides evidence of representation interaction between the image and metadata modalities. 
> It does NOT prove that any specific metadata feature or image region caused the prediction.
>
> These patterns are descriptive, not explanatory in a causal sense.

## Reproducibility

- Random seed: 42
- Checkpoint: `models/multimodal/attention_model.pt`
- Test embeddings: `data/processed/embeddings/test_embeddings.npz`
- Samples analyzed: 500
