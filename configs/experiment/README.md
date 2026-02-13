# Hydra Experiment Configuration Guide

This directory contains predefined experiment configurations for ablation studies and comparisons.

## Usage

All experiments can be run using the experiment configuration group:

```bash
python -m project.main experiment=<NAME>
```

## Available Experiments

### Ablation Studies (B-series)

**B1: CLIP Alignment Only**
```bash
python -m project.main experiment=B1_clip_only
```
- No map-guided pooling, only CLIP alignment loss
- Isolates the alignment objective

**B2: Map-Guided Pooling Only**
```bash
python -m project.main experiment=B2_map_only
```
- Guided pooling without CLIP loss
- Tests if clinical guidance helps alone

**B3: Full Method (Default)**
```bash
python -m project.main experiment=B3_full
```
- Spatiotemporal map-guided pooling + CLIP alignment
- Expected best performance

**B4: Full + Token-Level Loss**
```bash
python -m project.main experiment=B4_full_token
```
- Adds token-level alignment regularizer
- Improves interpretability and stability

### Map-Guided Variants (C-series)

**C1: Channel Gating Only**
```bash
python -m project.main experiment=C1_channel_gate
```
- Channel-wise gating without spatial weighting
- Faster but may lose spatial structure info

**C2: Weighted Pooling Only**
```bash
python -m project.main experiment=C2_weighted_pool
```
- Pure attention-weighted pooling, no gating
- Tests if pooling alone is sufficient

**C3: Sigmoid Gate**
```bash
python -m project.main experiment=C3_sigmoid_gate
```
- Same as B3 but with sigmoid-based gating
- More robust to attention map noise

### Semantic Validation (D-series)

**D: Retrieval Metrics**
```bash
python -m project.main experiment=D_retrieval
```
- Full method optimized for cross-modal retrieval
- Automatic logging of:
  - `test/retrieval_r1_v2a`, `test/retrieval_r5_v2a`
  - `test/retrieval_r1_a2v`, `test/retrieval_r5_a2v`
  - `test/align_sim_correct`, `test/align_sim_incorrect`, `test/align_sim_gap`, `test/align_sim_corr`

## Combining with Other Overrides

You can combine experiment configs with other Hydra overrides:

```bash
# B1 with custom learning rate and batch size
python -m project.main experiment=B1_clip_only optimizer.lr=0.0005 data.batch_size=8

# B3 with custom epochs
python -m project.main experiment=B3_full train.max_epochs=100

# C2 with debug mode
python -m project.main experiment=C2_weighted_pool train.fast_dev_run=true
```

## Batch Jobs

See PBS scripts in `pegasus/`:
- `run_ablation_b.sh` - runs B1-B4
- `run_map_guided_c.sh` - runs C1-C3
- `run_semantics_d.sh` - runs D

## Adding New Experiments

1. Create a new YAML file in `configs/experiment/` with a descriptive name
2. Define the model, loss, and train configs needed
3. Add a comment explaining the goal and expected behavior
4. Run with `python -m project.main experiment=your_name`
