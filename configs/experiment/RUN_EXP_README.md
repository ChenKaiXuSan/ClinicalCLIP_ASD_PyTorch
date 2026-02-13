# Quick Experiment Runner

A simple bash script to run ablation and comparison experiments in sequence.

## Usage

```bash
# Run all experiments
./run_exp.sh all

# Run B-series ablations (B1-B4)
./run_exp.sh B

# Run C-series map-guided variants (C1-C3)
./run_exp.sh C

# Run D-series retrieval validation
./run_exp.sh D

# Run specific experiments
./run_exp.sh B1_clip_only B3_full C2_weighted_pool

# Mix and match series
./run_exp.sh B C
```

## Experiments

### B-Series: Stepwise Ablation
- **B1**: CLIP alignment only
- **B2**: Map-guided pooling only
- **B3**: Full method (default)
- **B4**: Full + token-level loss

### C-Series: Map-Guided Variants
- **C1**: Channel gating only
- **C2**: Weighted pooling only
- **C3**: Sigmoid gate

### D-Series: Semantic Validation
- **D**: Full method with retrieval metrics

## Output

The script saves logs with:
- ✓/✗ status for each experiment
- Elapsed time for full run
- Summary of passed/failed experiments

## Tips

- Run `./run_exp.sh all` to get baseline results for all methods once
- Use `./run_exp.sh B` to focus on main ablation first
- Logs go to `logs/train/` with timestamp subdirectories (Hydra default)
- Edit `configs/experiment/*.yaml` to adjust hyperparameters per experiment
