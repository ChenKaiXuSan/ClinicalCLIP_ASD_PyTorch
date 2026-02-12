# CLIP-Style Video-Attention Alignment Model (VideoAttentionCLIP)

This document explains the structure and data flow of the CLIP-style video-attention alignment model implemented in [project/models/clip_align.py](project/models/clip_align.py).

## High-Level Goal
The model aligns video features with clinician-provided attention maps using a CLIP-like contrastive objective, while also producing classification logits for diagnosis. It supports multiple video backbones and optional attention-guided gating.

## Core Components

### 1) Encoders
- **Video encoder**: extracts a global feature vector from the input video.
  - Backbones: `3dcnn` (default), `2dcnn`, or `cnn_lstm`.
  - If `map_guided` is enabled with `3dcnn`, the video encoder becomes map-guided to incorporate attention directly.
- **Attention encoder**: a simple 3D CNN that encodes the attention map into a global feature vector.

### 2) Projection Heads (CLIP-style)
Two projection heads map features into a shared embedding space:
- `video_projection`: projects video features to `embed_dim`.
- `attn_projection`: projects attention features to `embed_dim`.

Both outputs are L2-normalized for contrastive learning.

### 3) Classifier
A linear classifier maps features to diagnosis logits.
- `classifier_source` controls which features are used:
  - `video` (default): video features
  - `attn`: attention features
  - `fusion`: average of video and attention features

## Detailed Architecture

### A) Video Encoders

#### a) Simple3DEncoder (default 3D CNN)
A small 3D CNN with global pooling. Produces a single feature vector per video.

#### b) Simple2DEncoder
Encodes each frame with a 2D CNN and averages frame features across time.

#### c) SimpleCNNLSTMEncoder
Encodes each frame with a 2D CNN, then aggregates with an LSTM and uses the final state.

#### d) MapGuidedVideoEncoder (3D CNN with attention gating)
When `map_guided=True` and `clip_backbone=3dcnn`, the model uses this encoder:
- 3D CNN produces spatiotemporal tokens.
- Attention map is summarized with a global average to a per-channel vector.
- A small MLP produces a gate vector in `[0, 1]`.
- Tokens are gated (channel-wise) before pooling.
- Pooled features are projected to `feature_dim`.

### B) Non-3D Backbones with Attention Gating
If `map_guided=True` and the backbone is not `3dcnn`:
- A separate `map_gate` MLP is applied to the attention summary.
- The resulting gate scales the final video feature vector.

### C) Attention Encoder
`Simple3DEncoder` encodes attention maps into an attention feature vector.

## Forward Pass (VideoAttentionCLIP)
Inputs:
- `video`: shape `(B, 3, T, H, W)`
- `attn_map`: shape `(B, C_attn, T, H, W)`

Steps:
1. **Video feature extraction**
   - If `map_guided` with `3dcnn`: use map-guided encoder.
   - Else: use chosen backbone and optional `map_gate` scaling.
2. **Attention feature extraction**
   - Encode `attn_map` with attention encoder.
3. **CLIP embeddings**
   - Project video and attention features into shared embedding space.
4. **Classification**
   - Produce logits from `video`, `attn`, or `fusion` features.

Outputs:
- `logits`: classification outputs
- `video_embed`, `attn_embed`: normalized CLIP embeddings
- `video_feat`, `attn_feat`: feature vectors before projection
- `video_tokens`, `video_gate`: optional outputs for map-guided 3D CNN

## Contrastive Objective
The CLIP-style loss aligns embeddings by matching each video to its corresponding attention map.

Given `video_embed` and `attn_embed`:
- Compute similarity matrix (dot product).
- Apply temperature scaling.
- Use cross-entropy loss for video-to-attn and attn-to-video directions.
- Final loss is the average of both.

## Key Configuration Fields
- `model.clip_feature_dim`: feature dimension before projection (default 512)
- `model.clip_embed_dim`: embedding dimension for contrastive learning (default 256)
- `model.model_class_num`: number of diagnostic classes
- `model.attn_in_channels`: attention map channels (default 1)
- `model.clip_backbone`: `3dcnn`, `2dcnn`, or `cnn_lstm`
- `model.map_guided`: enable attention-guided gating
- `model.map_guided_hidden_dim`: hidden dimension for map-guided gating
- `model.clip_classifier_source`: `video`, `attn`, or `fusion`

## Notes
- The model is intentionally lightweight for experimentation; larger backbones can be substituted later.
- `map_guided` adds interpretability by making video features depend on clinical attention patterns.

## Ablation and Comparison Experiments

This section documents the ablation experiments and the exact switches used to run them.

### B) Stepwise Ablation (Add Modules Gradually)

**B1: +CLIP alignment (no map-guided pooling)**
- Goal: test whether alignment loss helps (especially in low-data or cross-doctor settings).
- Switches:
   - `model.map_guided=false`
   - `loss.clip_weight>0`
   - `model.lambda_token=0`

**B2: +map-guided pooling (no CLIP)**
- Goal: test the benefit of guided pooling alone.
- Switches:
   - `model.map_guided=true`
   - `model.map_guided_type=spatiotemporal`
   - `loss.clip_weight=0`
   - `model.lambda_token=0`

**B3: full method (map-guided + CLIP)**
- Goal: show guided pooling and CLIP alignment are complementary.
- Switches:
   - `model.map_guided=true`
   - `model.map_guided_type=spatiotemporal`
   - `loss.clip_weight>0`
   - `model.lambda_token=0`

**B4: +token-level alignment loss (optional)**
- Goal: improve interpretability and robustness.
- Switches:
   - `model.map_guided=true`
   - `model.map_guided_type=spatiotemporal`
   - `loss.clip_weight>0`
   - `model.lambda_token>0`

### C) Map-Guided Variants (Key Novelty Comparison)

**C1: channel gating only**
- Goal: isolate the effect of channel-only gating.
- Switches:
   - `model.map_guided=true`
   - `model.map_guided_type=channel`

**C2: weighted pooling only (no gating)**
- Goal: isolate pure attention-weighted pooling.
- Switches:
   - `model.map_guided=true`
   - `model.map_guided_type=weighted_pool`

**C3: sigmoid gate vs. linear gate**
- Goal: assess sensitivity to attention noise and scale.
- Switches:
   - `model.map_guided=true`
   - `model.map_guided_type=spatiotemporal`
   - `model.map_guided_sigmoid_gate=true|false`

### D) Does CLIP Learn Clinician Semantics?

**D1: cross-modal retrieval**
- Metric: R@1 / R@5 for video->attn and attn->video.
- Implemented in test epoch end; logged as:
   - `test/retrieval_r1_v2a`, `test/retrieval_r5_v2a`
   - `test/retrieval_r1_a2v`, `test/retrieval_r5_a2v`

**D2: alignment score vs. classification correctness**
- Metric: mean similarity for correct/incorrect samples, their gap, and correlation.
- Logged as:
   - `test/align_sim_correct`, `test/align_sim_incorrect`, `test/align_sim_gap`, `test/align_sim_corr`

### Example Commands

```bash
# B1: CLIP only
python -m project.main model.map_guided=false loss.clip_weight=1.0 model.lambda_token=0

# B2: map-guided only
python -m project.main model.map_guided=true model.map_guided_type=spatiotemporal loss.clip_weight=0 model.lambda_token=0

# B3: full method
python -m project.main model.map_guided=true model.map_guided_type=spatiotemporal loss.clip_weight=1.0 model.lambda_token=0

# B4: add token loss
python -m project.main model.map_guided=true model.map_guided_type=spatiotemporal loss.clip_weight=1.0 model.lambda_token=0.1

# C1/C2/C3: map-guided variants
python -m project.main model.map_guided=true model.map_guided_type=channel
python -m project.main model.map_guided=true model.map_guided_type=weighted_pool
python -m project.main model.map_guided=true model.map_guided_type=spatiotemporal model.map_guided_sigmoid_gate=true
```

## Experiment Table Template

Use the table below to summarize ablations and map-guided comparisons in the paper.

| ID | Method | Key Switches | Loss | Metrics | Notes |
| --- | --- | --- | --- | --- | --- |
| B1 | CLIP alignment only | `model.map_guided=false` | $\mathcal{L}_{CE} + \lambda\,\mathcal{L}_{CLIP}$ | Acc, F1, R@1/R@5 | Tests if alignment helps without guidance |
| B2 | Map-guided pooling only | `model.map_guided=true`, `model.map_guided_type=spatiotemporal` | $\mathcal{L}_{CE}$ | Acc, F1 | Tests guided pooling alone |
| B3 | Full method | `model.map_guided=true`, `model.map_guided_type=spatiotemporal` | $\mathcal{L}_{CE} + \lambda\,\mathcal{L}_{CLIP}$ | Acc, F1, R@1/R@5 | Guidance + alignment |
| B4 | + token-level loss | `model.lambda_token>0` | $\mathcal{L}_{CE} + \lambda\,\mathcal{L}_{CLIP} + \mu\,\mathcal{L}_{token}$ | Acc, F1, R@1/R@5 | Stability / interpretability |
| C1 | Channel gating | `model.map_guided_type=channel` | Same as B2/B3 | Acc, F1 | Channel-only gating |
| C2 | Weighted pooling | `model.map_guided_type=weighted_pool` | Same as B2/B3 | Acc, F1 | Pooling only |
| C3 | Sigmoid vs linear gate | `model.map_guided_sigmoid_gate=true/false` | Same as B3/B4 | Acc, F1 | Sensitivity to map scale |
| D1 | Cross-modal retrieval | N/A (test metric) | N/A | R@1, R@5 (v2a, a2v) | Evidence of alignment |
| D2 | Alignment vs correctness | N/A (test metric) | N/A | Align sim gap, corr | Clinical consistency proxy |
