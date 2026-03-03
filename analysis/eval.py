#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import itertools
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
	accuracy_score,
	balanced_accuracy_score,
	confusion_matrix,
	precision_recall_fscore_support,
)

try:
	import matplotlib.pyplot as plt
	import seaborn as sns
except Exception:
	plt = None
	sns = None


CLASS_NAME_MAP = {
	2: ["ASD", "non-ASD"],
	3: ["ASD", "DHS", "LCS_HipOA"],
	4: ["ASD", "DHS", "LCS_HipOA", "normal"],
}

SCALAR_METRIC_KEYS = [
	"accuracy",
	"balanced_accuracy",
	"precision_macro",
	"recall_macro",
	"f1_macro",
	"precision_weighted",
	"recall_weighted",
	"f1_weighted",
]


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Evaluate and compare best_preds across different experiments."
	)
	parser.add_argument(
		"--train-root",
		type=Path,
		default=Path("/work/SSR/share/code/ClinicalCLIP_ASD_PyTorch/logs/train"),
		help="Root directory containing experiment logs.",
	)
	parser.add_argument(
		"--output-root",
		type=Path,
		default=Path("/work/SSR/share/code/ClinicalCLIP_ASD_PyTorch/logs/eval_compare"),
		help="Output directory for evaluation reports.",
	)
	parser.add_argument(
		"--experiments",
		nargs="*",
		default=None,
		help="Optional experiment names to evaluate. If not set, all under train-root are used.",
	)
	parser.add_argument(
		"--reference-experiment",
		type=str,
		default=None,
		help="Reference experiment for significance test. Default: best f1_macro_mean experiment.",
	)
	parser.add_argument(
		"--target-folds",
		type=int,
		default=5,
		help="Prefer runs with this number of valid folds (default: 5).",
	)
	parser.add_argument(
		"--allow-incomplete-folds",
		action="store_true",
		help="Allow using runs with fewer folds if full-fold runs are unavailable.",
	)
	return parser.parse_args()


def count_valid_pairs(best_preds_dir: Path) -> int:
	pred_files = sorted(best_preds_dir.glob("*_pred.pt"))
	valid_pairs = 0
	for pred_file in pred_files:
		fold = pred_file.stem.replace("_pred", "")
		if (best_preds_dir / f"{fold}_label.pt").exists():
			valid_pairs += 1
	return valid_pairs


def find_preferred_run(
	experiment_dir: Path,
	target_folds: int,
	allow_incomplete_folds: bool,
) -> Path | None:
	full_candidates: list[Path] = []
	partial_candidates: list[tuple[int, Path]] = []
	for date_dir in sorted([p for p in experiment_dir.iterdir() if p.is_dir()]):
		for time_dir in sorted([p for p in date_dir.iterdir() if p.is_dir()]):
			best_preds = time_dir / "best_preds"
			if not best_preds.exists():
				continue
			valid_pairs = count_valid_pairs(best_preds)
			if valid_pairs <= 0:
				continue
			if valid_pairs >= target_folds:
				full_candidates.append(time_dir)
			else:
				partial_candidates.append((valid_pairs, time_dir))

	if full_candidates:
		return sorted(full_candidates)[-1]

	if not allow_incomplete_folds:
		return None

	if not partial_candidates:
		return None

	partial_candidates.sort(key=lambda x: (x[0], str(x[1])), reverse=True)
	return partial_candidates[0][1]


def load_best_preds_by_fold(best_preds_dir: Path) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], int]:
	fold_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
	max_class_idx = -1

	pred_files = sorted(best_preds_dir.glob("*_pred.pt"))
	for pred_file in pred_files:
		fold = pred_file.stem.replace("_pred", "")
		label_file = best_preds_dir / f"{fold}_label.pt"
		if not label_file.exists():
			continue

		pred_tensor = torch.load(pred_file, map_location="cpu")
		label_tensor = torch.load(label_file, map_location="cpu")

		if isinstance(pred_tensor, list):
			pred_tensor = torch.cat(pred_tensor, dim=0)
		if isinstance(label_tensor, list):
			label_tensor = torch.cat(label_tensor, dim=0)

		pred_tensor = pred_tensor.detach().cpu()
		label_tensor = label_tensor.detach().cpu().long().view(-1)

		if pred_tensor.ndim > 1:
			pred_cls = pred_tensor.argmax(dim=1).long().view(-1)
			inferred_classes = int(pred_tensor.shape[1])
		else:
			pred_cls = pred_tensor.long().view(-1)
			inferred_classes = int(max(pred_cls.max().item(), label_tensor.max().item()) + 1)

		max_class_idx = max(max_class_idx, inferred_classes - 1)

		fold_data[fold] = (label_tensor.numpy(), pred_cls.numpy())

	if not fold_data:
		raise ValueError(f"No valid pred/label pairs found in {best_preds_dir}")

	all_labels = [v[0] for v in fold_data.values()]
	all_preds = [v[1] for v in fold_data.values()]
	y_pred = np.concatenate(all_preds, axis=0)
	y_true = np.concatenate(all_labels, axis=0)
	num_classes = max(max_class_idx + 1, int(max(y_true.max(), y_pred.max()) + 1))
	return fold_data, num_classes


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> dict:
	labels = list(range(num_classes))
	cm = confusion_matrix(y_true, y_pred, labels=labels)
	with np.errstate(divide="ignore", invalid="ignore"):
		cm_norm = cm.astype(np.float64) / cm.sum(axis=1, keepdims=True)
		cm_norm = np.nan_to_num(cm_norm)

	p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
		y_true, y_pred, average="macro", zero_division=0
	)
	p_weight, r_weight, f_weight, _ = precision_recall_fscore_support(
		y_true, y_pred, average="weighted", zero_division=0
	)
	p_class, r_class, f_class, support = precision_recall_fscore_support(
		y_true, y_pred, labels=labels, average=None, zero_division=0
	)

	return {
		"accuracy": float(accuracy_score(y_true, y_pred)),
		"balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
		"precision_macro": float(p_macro),
		"recall_macro": float(r_macro),
		"f1_macro": float(f_macro),
		"precision_weighted": float(p_weight),
		"recall_weighted": float(r_weight),
		"f1_weighted": float(f_weight),
		"precision_per_class": p_class.tolist(),
		"recall_per_class": r_class.tolist(),
		"f1_per_class": f_class.tolist(),
		"support_per_class": support.tolist(),
		"confusion_matrix": cm.tolist(),
		"confusion_matrix_norm": cm_norm.tolist(),
	}


def compute_scalar_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
	p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
		y_true, y_pred, average="macro", zero_division=0
	)
	p_weight, r_weight, f_weight, _ = precision_recall_fscore_support(
		y_true, y_pred, average="weighted", zero_division=0
	)
	return {
		"accuracy": float(accuracy_score(y_true, y_pred)),
		"balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
		"precision_macro": float(p_macro),
		"recall_macro": float(r_macro),
		"f1_macro": float(f_macro),
		"precision_weighted": float(p_weight),
		"recall_weighted": float(r_weight),
		"f1_weighted": float(f_weight),
	}


def ci95(values: list[float]) -> dict:
	arr = np.asarray(values, dtype=np.float64)
	mean = float(arr.mean())
	if arr.size <= 1:
		return {"mean": mean, "std": 0.0, "ci95_low": mean, "ci95_high": mean}
	std = float(arr.std(ddof=1))
	margin = 1.96 * std / np.sqrt(arr.size)
	return {
		"mean": mean,
		"std": std,
		"ci95_low": float(mean - margin),
		"ci95_high": float(mean + margin),
	}


def paired_permutation_pvalue(a: list[float], b: list[float]) -> float:
	a_arr = np.asarray(a, dtype=np.float64)
	b_arr = np.asarray(b, dtype=np.float64)
	if a_arr.shape != b_arr.shape:
		raise ValueError("paired test requires equal-length vectors")
	diff = a_arr - b_arr
	obs = abs(float(diff.mean()))
	n = diff.size
	if n == 0:
		return 1.0

	count = 0
	total = 0
	for signs in itertools.product([-1.0, 1.0], repeat=n):
		s = np.asarray(signs, dtype=np.float64)
		val = abs(float((diff * s).mean()))
		if val >= obs - 1e-12:
			count += 1
		total += 1
	return float((count + 1) / (total + 1))


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
	with path.open("w", encoding="utf-8", newline="") as f:
		writer = csv.writer(f)
		writer.writerow(header)
		writer.writerows(rows)


def save_confusion_matrix(cm: np.ndarray, labels: list[str], title: str, save_path: Path, fmt: str) -> None:
	if plt is None or sns is None:
		return
	plt.figure(figsize=(8, 6))
	sns.heatmap(
		cm,
		annot=True,
		fmt=fmt,
		cmap="Reds",
		xticklabels=labels,
		yticklabels=labels,
	)
	plt.title(title)
	plt.xlabel("Predicted")
	plt.ylabel("Actual")
	plt.tight_layout()
	plt.savefig(save_path, dpi=300)
	plt.close()


def evaluate_experiment(experiment_name: str, run_dir: Path, output_root: Path) -> dict:
	best_preds_dir = run_dir / "best_preds"
	fold_data, num_classes = load_best_preds_by_fold(best_preds_dir)
	fold_ids = sorted(fold_data.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x))
	all_y_true = [fold_data[f][0] for f in fold_ids]
	all_y_pred = [fold_data[f][1] for f in fold_ids]
	y_true = np.concatenate(all_y_true, axis=0)
	y_pred = np.concatenate(all_y_pred, axis=0)
	metrics = compute_metrics(y_true, y_pred, num_classes)

	class_names = CLASS_NAME_MAP.get(num_classes, [f"class_{i}" for i in range(num_classes)])
	exp_output = output_root / experiment_name
	exp_output.mkdir(parents=True, exist_ok=True)

	cm = np.array(metrics["confusion_matrix"], dtype=np.int64)
	cm_norm = np.array(metrics["confusion_matrix_norm"], dtype=np.float64)

	cm_rows = [[class_names[i], *cm[i].tolist()] for i in range(len(class_names))]
	cm_norm_rows = [[class_names[i], *cm_norm[i].tolist()] for i in range(len(class_names))]
	write_csv(
		exp_output / "confusion_matrix_counts.csv",
		["actual\\pred", *class_names],
		cm_rows,
	)
	write_csv(
		exp_output / "confusion_matrix_norm.csv",
		["actual\\pred", *class_names],
		cm_norm_rows,
	)

	save_confusion_matrix(
		cm,
		class_names,
		f"{experiment_name} Confusion Matrix (Count)",
		exp_output / "confusion_matrix_counts.png",
		fmt="d",
	)
	save_confusion_matrix(
		cm_norm * 100.0,
		class_names,
		f"{experiment_name} Confusion Matrix (%)",
		exp_output / "confusion_matrix_percent.png",
		fmt=".2f",
	)

	per_class_rows: list[list] = []
	for idx, cls_name in enumerate(class_names):
		per_class_rows.append(
			[
				cls_name,
				metrics["precision_per_class"][idx],
				metrics["recall_per_class"][idx],
				metrics["f1_per_class"][idx],
				metrics["support_per_class"][idx],
			]
		)
	write_csv(
		exp_output / "per_class_metrics.csv",
		["class", "precision", "recall", "f1", "support"],
		per_class_rows,
	)

	per_fold_rows: list[list] = []
	fold_metric_dict: dict[str, list[float]] = {k: [] for k in SCALAR_METRIC_KEYS}
	fold_metric_by_id: dict[str, dict[str, float]] = {}
	for fold in fold_ids:
		fy_true, fy_pred = fold_data[fold]
		fold_m = compute_scalar_metrics(fy_true, fy_pred)
		fold_metric_by_id[fold] = fold_m
		for k in SCALAR_METRIC_KEYS:
			fold_metric_dict[k].append(float(fold_m[k]))
		per_fold_rows.append([
			fold,
			len(fy_true),
			fold_m["accuracy"],
			fold_m["balanced_accuracy"],
			fold_m["precision_macro"],
			fold_m["recall_macro"],
			fold_m["f1_macro"],
			fold_m["precision_weighted"],
			fold_m["recall_weighted"],
			fold_m["f1_weighted"],
		])
	write_csv(
		exp_output / "per_fold_metrics.csv",
		[
			"fold",
			"num_samples",
			"accuracy",
			"balanced_accuracy",
			"precision_macro",
			"recall_macro",
			"f1_macro",
			"precision_weighted",
			"recall_weighted",
			"f1_weighted",
		],
		per_fold_rows,
	)

	ci_rows: list[list] = []
	for k in SCALAR_METRIC_KEYS:
		stats = ci95(fold_metric_dict[k])
		ci_rows.append([k, stats["mean"], stats["std"], stats["ci95_low"], stats["ci95_high"]])
	write_csv(
		exp_output / "fold_ci95_metrics.csv",
		["metric", "mean", "std", "ci95_low", "ci95_high"],
		ci_rows,
	)

	summary = {
		"experiment": experiment_name,
		"run_dir": str(run_dir),
		"folds": sorted(fold_ids),
		"num_samples": int(len(y_true)),
		"num_classes": int(num_classes),
		"accuracy": metrics["accuracy"],
		"balanced_accuracy": metrics["balanced_accuracy"],
		"precision_macro": metrics["precision_macro"],
		"recall_macro": metrics["recall_macro"],
		"f1_macro": metrics["f1_macro"],
		"precision_weighted": metrics["precision_weighted"],
		"recall_weighted": metrics["recall_weighted"],
		"f1_weighted": metrics["f1_weighted"],
		"fold_metrics": fold_metric_dict,
		"fold_metric_by_id": fold_metric_by_id,
	}

	for k in SCALAR_METRIC_KEYS:
		stats = ci95(fold_metric_dict[k])
		summary[f"{k}_mean"] = stats["mean"]
		summary[f"{k}_std"] = stats["std"]
		summary[f"{k}_ci95_low"] = stats["ci95_low"]
		summary[f"{k}_ci95_high"] = stats["ci95_high"]

	with (exp_output / "summary.json").open("w", encoding="utf-8") as f:
		json.dump({**summary, **metrics}, f, indent=2, ensure_ascii=False)

	return summary


def main() -> None:
	args = parse_args()
	train_root = args.train_root
	output_root = args.output_root
	output_root.mkdir(parents=True, exist_ok=True)

	if not train_root.exists():
		raise FileNotFoundError(f"train root not found: {train_root}")

	if args.experiments:
		experiment_dirs = [train_root / name for name in args.experiments]
	else:
		experiment_dirs = sorted([p for p in train_root.iterdir() if p.is_dir()])

	all_summaries: list[dict] = []
	for exp_dir in experiment_dirs:
		if not exp_dir.exists() or not exp_dir.is_dir():
			continue
		run_dir = find_preferred_run(
			exp_dir,
			target_folds=args.target_folds,
			allow_incomplete_folds=args.allow_incomplete_folds,
		)
		if run_dir is None:
			print(
				f"[skip] {exp_dir.name}: no run with >= {args.target_folds} folds "
				f"(use --allow-incomplete-folds to fallback)"
			)
			continue

		print(f"[eval] {exp_dir.name} -> {run_dir}")
		summary = evaluate_experiment(exp_dir.name, run_dir, output_root)
		all_summaries.append(summary)

	if not all_summaries:
		raise RuntimeError("No experiment was evaluated. Please check train-root and experiment names.")

	all_summaries = sorted(all_summaries, key=lambda x: (x["f1_macro_mean"], x["accuracy_mean"]), reverse=True)

	reference_name = args.reference_experiment if args.reference_experiment else all_summaries[0]["experiment"]
	reference = None
	for row in all_summaries:
		if row["experiment"] == reference_name:
			reference = row
			break
	if reference is None:
		raise ValueError(f"reference experiment not found: {reference_name}")

	sig_rows: list[list] = []
	for row in all_summaries:
		if row["experiment"] == reference_name:
			sig_rows.append([
				row["experiment"],
				reference_name,
				len(row["folds"]),
				0.0,
				0.0,
				1.0,
				1.0,
			])
			continue
		common_folds = sorted(set(row["folds"]) & set(reference["folds"]))
		if not common_folds:
			p_f1 = float("nan")
			p_acc = float("nan")
		else:
			row_f1 = [row["fold_metric_by_id"][f]["f1_macro"] for f in common_folds]
			ref_f1 = [reference["fold_metric_by_id"][f]["f1_macro"] for f in common_folds]
			row_acc = [row["fold_metric_by_id"][f]["accuracy"] for f in common_folds]
			ref_acc = [reference["fold_metric_by_id"][f]["accuracy"] for f in common_folds]
			p_f1 = paired_permutation_pvalue(row_f1, ref_f1)
			p_acc = paired_permutation_pvalue(row_acc, ref_acc)
		sig_rows.append([
			row["experiment"],
			reference_name,
			len(common_folds),
			row["f1_macro_mean"] - reference["f1_macro_mean"],
			row["accuracy_mean"] - reference["accuracy_mean"],
			p_f1,
			p_acc,
		])
	write_csv(
		output_root / "significance_vs_reference.csv",
		[
			"experiment",
			"reference",
			"n_common_folds",
			"delta_f1_macro_mean",
			"delta_accuracy_mean",
			"pvalue_f1_macro",
			"pvalue_accuracy",
		],
		sig_rows,
	)
	summary_header = [
		"experiment",
		"run_dir",
		"num_samples",
		"num_classes",
		"accuracy",
		"accuracy_mean",
		"accuracy_std",
		"accuracy_ci95_low",
		"accuracy_ci95_high",
		"balanced_accuracy",
		"balanced_accuracy_mean",
		"balanced_accuracy_std",
		"balanced_accuracy_ci95_low",
		"balanced_accuracy_ci95_high",
		"precision_macro",
		"recall_macro",
		"f1_macro",
		"f1_macro_mean",
		"f1_macro_std",
		"f1_macro_ci95_low",
		"f1_macro_ci95_high",
		"precision_weighted",
		"recall_weighted",
		"f1_weighted",
	]
	summary_rows = [[row.get(k, "") for k in summary_header] for row in all_summaries]
	write_csv(output_root / "experiment_comparison_summary.csv", summary_header, summary_rows)
	with (output_root / "experiment_comparison_summary.json").open("w", encoding="utf-8") as f:
		json.dump(all_summaries, f, indent=2, ensure_ascii=False)

	paper_rows: list[list] = []
	for row in all_summaries:
		paper_rows.append([
			row["experiment"],
			f"{row['accuracy_mean']:.4f} ± {row['accuracy_std']:.4f}",
			f"[{row['accuracy_ci95_low']:.4f}, {row['accuracy_ci95_high']:.4f}]",
			f"{row['f1_macro_mean']:.4f} ± {row['f1_macro_std']:.4f}",
			f"[{row['f1_macro_ci95_low']:.4f}, {row['f1_macro_ci95_high']:.4f}]",
			f"{row['balanced_accuracy_mean']:.4f} ± {row['balanced_accuracy_std']:.4f}",
		])
	write_csv(
		output_root / "paper_table.csv",
		[
			"experiment",
			"accuracy_mean±std",
			"accuracy_95ci",
			"f1_macro_mean±std",
			"f1_macro_95ci",
			"balanced_accuracy_mean±std",
		],
		paper_rows,
	)
	with (output_root / "paper_table.md").open("w", encoding="utf-8") as f:
		f.write("| experiment | accuracy (mean±std) | accuracy 95% CI | f1_macro (mean±std) | f1_macro 95% CI | balanced_acc (mean±std) |\n")
		f.write("|---|---:|---:|---:|---:|---:|\n")
		for r in paper_rows:
			f.write(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} |\n")

	print("=" * 90)
	print("Experiment comparison summary (sorted by fold f1_macro mean, accuracy mean):")
	print("experiment\tnum_samples\tacc_mean±std\tf1_mean±std\tbalanced_acc_mean±std")
	for row in all_summaries:
		print(
			f"{row['experiment']}\t{row['num_samples']}\t"
			f"{row['accuracy_mean']:.4f}±{row['accuracy_std']:.4f}\t"
			f"{row['f1_macro_mean']:.4f}±{row['f1_macro_std']:.4f}\t"
			f"{row['balanced_accuracy_mean']:.4f}±{row['balanced_accuracy_std']:.4f}"
		)
	print("=" * 90)
	print(f"reference for significance test: {reference_name}")
	if plt is None or sns is None:
		print("plot libs not found: saved numeric CSV/JSON only, skipped confusion matrix PNG.")
	print(f"saved reports to: {output_root}")


if __name__ == "__main__":
	main()
