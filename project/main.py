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
from omegaconf import DictConfig
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger

from project.cross_validation import DefineCrossValidation
from project.dataloader.data_loader import WalkDataModule

# CLIP-style alignment
from project.trainer.train_clip_align import CLIPAlignModule

# compare experiment
from project.trainer.train_cnn import CNNModule

# compare experiment
from project.trainer.train_cnn_lstm import CNNLstmModule

#####################################
# select different experiment trainer
#####################################
# 3D CNN model
from project.trainer.train_res_3dcnn import SingleModule


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
        save_dir=os.path.join(hparams.log_path),
        name=str(fold),  # here should be str type.
    )

    # some callbacks
    progress_bar = TQDMProgressBar(refresh_rate=100)
    rich_model_summary = RichModelSummary(max_depth=2)

    # define the checkpoint becavier.
    model_check_point = ModelCheckpoint(
        filename="{epoch}-{val/loss:.2f}-{val/video_acc:.4f}",
        auto_insert_metric_name=False,
        monitor="val/video_acc",
        mode="max",
        save_last=False,
        save_top_k=2,
    )

    # define the early stop.
    early_stopping = EarlyStopping(
        monitor="val/video_acc",
        patience=3,
        mode="max",
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = Trainer(
        devices=[
            int(hparams.train.gpu_num),
        ],
        accelerator="gpu",
        max_epochs=hparams.train.max_epochs,
        logger=tb_logger,  # wandb_logger,
        check_val_every_n_epoch=1,
        callbacks=[
            progress_bar,
            rich_model_summary,
            model_check_point,
            early_stopping,
            lr_monitor,
        ],
        limit_train_batches=2,
        limit_val_batches=2,
        limit_test_batches=2,
    )

    trainer.fit(classification_module, data_module)

    # the validate method will wirte in the same log twice, so use the test method.
    trainer.test(
        classification_module,
        data_module,
        ckpt_path="best",
        weights_only=False,
    )


@hydra.main(
    version_base=None,
    config_path="../configs",  # * the config_path is relative to location of the python script
    config_name="config.yaml",
)
def init_params(config):
    #######################
    # prepare dataset index
    #######################

    fold_dataset_idx = DefineCrossValidation(config)()

    logging.info("#" * 50)
    logging.info("Start train all fold")
    logging.info("#" * 50)

    #########
    # K fold
    #########
    # * for one fold, we first train/val model, then save the best ckpt preds/label into .pt file.

    for fold, dataset_value in fold_dataset_idx.items():
        logging.info("#" * 50)
        logging.info("Start train fold: {}".format(fold))
        logging.info("#" * 50)

        train(config, dataset_value, fold)

        logging.info("#" * 50)
        logging.info("finish train fold: {}".format(fold))
        logging.info("#" * 50)

    logging.info("#" * 50)
    logging.info("finish train all fold")
    logging.info("#" * 50)


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    init_params()
