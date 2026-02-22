# Quick Experiment Runner

A simple bash script to run ablation and comparison experiments in sequence.

This folder now also provides split scripts for each experiment and series.

## Usage

```bash
# Run all experiments
./run_exp.sh all

# Run split all-series script (B + C)
./run_all_series.sh

# Run B-series ablations (B1-B4)
./run_exp.sh B
./run_B_series.sh

# Run C-series map-guided variants (C1-C3)
./run_exp.sh C
./run_C_series.sh

# Run specific experiments
./run_exp.sh B1_clip_only B3_full C2_weighted_pool

# Run individual split scripts
./run_B1_clip_only.sh
./run_B2_map_only.sh
./run_B3_full.sh
./run_B4_full_token.sh
./run_C1_channel_gate.sh
./run_C2_weighted_pool.sh
./run_C3_sigmoid_gate.sh

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
