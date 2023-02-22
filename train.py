from share import *
import argparse, os, datetime
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
from datasets import DegradeDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning import seed_everything

from MyCallbacks import SetupCallback


def parse():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default="configs/default.yaml")
    p.add_argument('--seed', type=int, default=22)
    return p.parse_args()


if __name__ == "__main__":
    args = parse()
    config = OmegaConf.load(args.config)

    # SET SEED
    seed_everything(args.seed)

    # DEFINE NAMES
    assert config.name
    name = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    name = name + "_" + config.name
    logdir = os.path.join("logs", name)
    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")

    # DEFINE CALLBACK
    callbacks = [
        SetupCallback(
            now=name,
            logdir=logdir,
            ckptdir=ckptdir,
            cfgdir=cfgdir,
            config=config,
        ),
        ModelCheckpoint(
            **{
                "dirpath": os.path.join(ckptdir, 'trainstep_checkpoints'),
                "filename": "{epoch:06}-{step:09}",
                "verbose": True,
                'save_top_k': -1,
                'every_n_train_steps': config.logging.every_n_train_steps,
                'save_weights_only': True
            }),
        ImageLogger(
            batch_frequency=config.logging.batch_frequency,
            path_log_dir=logdir,
        ),
    ]

    # TRAINER
    trainer = pl.Trainer(
        **config.lightning,
        callbacks=callbacks,
    )

    # DATASET
    dataset = DegradeDataset()
    dataloader = DataLoader(dataset,
                            num_workers=config.data.num_workers,
                            batch_size=config.training.batch_size,
                            shuffle=False)

    # MODEL
    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model(config.path_config_model).cpu()
    model.load_state_dict(load_state_dict(config.path_sd_ckpt, location='cpu'))
    model.learning_rate = config.training.learning_rate
    model.sd_locked = config.training.sd_locked
    model.only_mid_control = config.training.only_mid_control

    # Train!
    trainer.fit(model, dataloader)
