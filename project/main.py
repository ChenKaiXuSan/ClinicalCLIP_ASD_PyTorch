#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/deep-learning-project-template/project/main.py
Project: /workspace/deep-learning-project-template/project
Created Date: Friday November 29th 2024
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Saturday February 7th 2026 10:51:24 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2024 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

import logging
import os

import hydra
import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
  
from cross_validation import DefineCrossValidation
from dataloader.data_loader import WalkDataModule

# CLIP-style alignment
from trainer.train_clip_align import CLIPAlignModule

# compare experiment
from trainer.train_cnn import CNNModule

# compare experiment
from trainer.train_cnn_lstm import CNNLstmModule

# 3D CNN model
from trainer.train_res_3dcnn import SingleModule


def train(hparams: DictConfig, dataset_idx, fold: int):
    """the train process for the one fold.

    Args:
        hparams (hydra): the hyperparameters.
        dataset_idx (int): the dataset index for the one fold.
        fold (int): the fold index.

    Returns:
        list: best trained model, data loader
    """

    seed_everything(42, workers=True)

    # * select experiment
    if hparams.model.backbone == "3dcnn":
        classification_module = SingleModule(hparams)
    # * compare experiment
    elif hparams.model.backbone == "cnn_lstm":
        classification_module = CNNLstmModule(hparams)
    # * compare experiment
    elif hparams.model.backbone == "2dcnn":
        classification_module = CNNModule(hparams)
    # * CLIP alignment
    elif hparams.model.backbone == "clip":
        classification_module = CLIPAlignModule(hparams)

    else:
        raise ValueError("the experiment backbone is not supported.")

    data_module = WalkDataModule(hparams, dataset_idx)

    # for the tensorboard
    tb_logger = TensorBoardLogger(
        save_dir=os.path.join(hparams.log_path, "tensorboard"),
        name=str(fold),  # here should be str type.
    )

    csv_logger = CSVLogger(
        save_dir=os.path.join(hparams.log_path, "csv"),
        name=str(fold),  # here should be str type.
    )

    # some callbacks
    progress_bar = RichProgressBar(refresh_rate=100, leave=True)
    rich_model_summary = RichModelSummary(max_depth=3)

    # define the checkpoint becavier.
    model_check_point = ModelCheckpoint(
        dirpath=os.path.join(hparams.log_path, "checkpoint", str(fold)),
        filename="{epoch}-{val/loss:.2f}-{val/video_acc:.4f}",
        auto_insert_metric_name=False,
        monitor="val/video_acc",
        mode="max",
        save_last=True,
        save_top_k=2,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = Trainer(
        devices=[
            int(hparams.train.gpu_num),
        ],
        accelerator="gpu",
        max_epochs=hparams.train.max_epochs,
        logger=[tb_logger, csv_logger],
        check_val_every_n_epoch=1,
        callbacks=[
            progress_bar,
            rich_model_summary,
            model_check_point,
            lr_monitor,
        ],
        # limit_train_batches=2,
        # limit_val_batches=2,
        # limit_test_batches=2,
    )

    trainer.fit(classification_module, data_module)

    # the validate method will wirte in the same log twice, so use the test method.
    test_metrics = trainer.test(
        classification_module,
        data_module,
        ckpt_path="best",
        weights_only=False,
    )

    with open(os.path.join(tb_logger.log_dir, "test_metrics.txt"), "w") as f:
        f.write(str(test_metrics))


@hydra.main(
    version_base=None,
    config_path="../configs",  # * the config_path is relative to location of the python script
    config_name="config.yaml",
)
def init_params(config):

    fold_dataset_idx = DefineCrossValidation(config)()

    for fold, dataset_value in fold_dataset_idx.items():
        train(config, dataset_value, fold)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    os.environ["HYDRA_FULL_ERROR"] = "1"
    init_params()
