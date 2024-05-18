import json
import pickle as pkl
from functools import partial
from pathlib import Path

import pytorch_lightning as pl
from loguru import logger
from torch.optim.lr_scheduler import OneCycleLR

from engine.data import get_loaders
from engine.model import SimpleMlp
from engine.schedulers import IdentityLR
from engine.trainer import Trainer
from engine.utils import device, get_accuracy

OUTPUT_PATH = Path("results/mlp_theoretical")
N_EPOCHS = 30


def main():
    pl.seed_everything(123)
    logger.info("Loading data")
    train_loader, test_loader = get_loaders("mlp")

    SCHEDULER_FACTORIES = [
        ("Constant", IdentityLR),
        (
            "OneCycle",
            partial(
                OneCycleLR,
                max_lr=1e-3,
                steps_per_epoch=len(train_loader),
                epochs=N_EPOCHS,
                anneal_strategy="linear",
                three_phase=True,
            ),
        ),
    ]

    logger.info("Running training multiple times.")
    for scheduler_name, scheduler_factory in SCHEDULER_FACTORIES:
        logger.info(f"Starting runs for {scheduler_name}")
        (OUTPUT_PATH / scheduler_name).mkdir(exist_ok=True, parents=True)
        for i in range(10):
            logger.info(f"Starting run {i}")
            model = SimpleMlp().to(device)
            trainer = Trainer(scheduler_factory)
            results = trainer.train(model, train_loader, 30)

            with open(
                OUTPUT_PATH / scheduler_name / f"results_{i}.pkl", "wb"
            ) as f:
                pkl.dump(results, f)
            with open(
                OUTPUT_PATH / scheduler_name / f"model_{i}.pkl", "wb"
            ) as f:
                pkl.dump(results, f)
            with open(
                OUTPUT_PATH / scheduler_name / f"accuracies_{i}.json", "w"
            ) as f:
                json.dump(
                    {
                        "train": get_accuracy(train_loader, model),
                        "test": get_accuracy(test_loader, model),
                    },
                    f,
                    indent=4,
                )


if __name__ == "__main__":
    main()
