# ClinicalCLIP: Clinician-Guided Multimodal Gait Analysis with CLIP

This repository contains the official PyTorch implementation of ClinicalCLIP, a clinician-guided multimodal framework for video-based gait analysis and clinical diagnosis.

The core idea is to integrate clinical prior knowledge (e.g., physician-annotated attention maps) with vision–language pretraining (CLIP) to improve interpretability, robustness, and diagnostic relevance in gait-based analysis.

⸻

🔍 Motivation

Automated gait analysis has shown great potential for non-invasive clinical diagnosis.
However, most existing deep learning approaches:
	•	Treat all spatial–temporal regions equally
	•	Ignore clinically meaningful motion cues
	•	Lack interpretability for medical decision support

In real clinical practice, physicians focus on specific joints, body regions, and motion phases when assessing gait abnormalities.

ClinicalCLIP bridges this gap by aligning gait videos with clinician-guided attention using CLIP-style multimodal learning.

⸻

✨ Key Contributions
	•	🧠 Clinical Knowledge-Guided Learning
Incorporates clinician-provided attention maps highlighting diagnostically important gait regions.
	•	🔗 CLIP-Based Multimodal Alignment
Aligns gait video representations with clinical attention cues in a shared embedding space.
	•	🎥 Video-Level Spatiotemporal Modeling
Supports 3D CNN / Transformer-based backbones for robust gait representation learning.
	•	🔍 Interpretability by Design
Enables visual and quantitative analysis of where and when the model attends during gait.
	•	🏥 Non-Invasive Clinical Application
Designed for real-world clinical gait assessment without wearable sensors.

⸻

🧩 Framework Overview

Input:
  - Gait video (RGB)
  - Clinician-annotated attention maps (spatial / spatiotemporal)

Pipeline:
  Video Encoder (3D CNN / ViT)
        │
        ├── Visual Embedding
        │
  Attention Encoder
        │
        ├── Clinical Embedding
        │
  ──► CLIP-style Contrastive Alignment
        │
        └── Downstream Tasks
              • Diagnosis / Classification
              • Retrieval
              • Interpretability Analysis


⸻

📁 Repository Structure

ClinicalCLIP/
├── configs/                # Hydra configuration files
├── project/                # Training entrypoints and models
│   ├── dataloader/         # Video + attention map datasets
│   ├── models/             # CLIP alignment + baselines
│   └── trainer/            # PyTorch Lightning trainers
├── tests/                  # Unit tests
└── README.md


⸻

🚀 Getting Started

1️⃣ Environment Setup

conda create -n clinicalclip python=3.10
conda activate clinicalclip
pip install -r requirements.txt

2️⃣ Dataset Preparation

Expected data format (for CLIP alignment with clinician attention):

data/
├── videos/
│   └── subject_x/
│       └── gait.mp4
├── attention_maps/
│   └── subject_x/
│       └── attention.npy
└── labels.csv

Attention maps can be frame-level, joint-level, or region-level, depending on the experiment.
Set `data.doctor_results_path` and `data.skeleton_path` to enable attention maps.

⸻

3️⃣ Training

python -m project.main \
  train.backbone=clip_align \
  train.attn_map=True \
  model.clip_backbone=3dcnn

Hydra is used for all configurations.

⸻

📊 Evaluation

Supported evaluation settings include:
	•	Diagnosis accuracy / F1-score
	•	Cross-subject validation
	•	Attention consistency analysis
	•	Ablation on clinical priors

✅ Baseline & backbone comparisons:
  - train.backbone=3dcnn / 2dcnn / cnn_lstm / two_stream
  - CLIP alignment with model.clip_backbone=3dcnn / 2dcnn / cnn_lstm

🔬 CLIP implementation notes:
  - Video encoder and attention encoder produce embeddings aligned via contrastive loss.
  - Use model.clip_classifier_source=video/attn/fusion to choose classification head input.
  - Set loss.clip_weight to balance CLIP alignment vs classification loss.

python scripts/eval.py


⸻

📈 Visualization

The repository provides tools for:
	•	Attention heatmap overlay on gait videos
	•	Phase-wise gait attention analysis
	•	Case-level interpretability reports

python visualization/vis_attention.py


⸻

🏥 Clinical Use Case

This framework is designed for applications such as:
	•	Adult Spinal Deformity (ASD) gait assessment
	•	Neurological disorder screening
	•	Explainable clinical decision support
	•	Human-centered AI in medical video analysis

⸻

📄 Citation

If you find this work useful, please consider citing:

@article{chen2025clinicalclip,
  title   = {ClinicalCLIP: Clinician-Guided Multimodal Gait Analysis via Vision–Language Pretraining},
  author  = {Chen, Kaixu and collaborators},
  journal = {TBD},
  year    = {2025}
}


⸻

📬 Contact

Kaixu Chen
University of Tsukuba
📧 chenkaixusan@gmail.com

⸻

⭐ Acknowledgements

This project is inspired by interdisciplinary collaboration between computer vision researchers and clinicians, aiming to build trustworthy and interpretable medical AI systems.
