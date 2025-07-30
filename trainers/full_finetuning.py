# trainers/full_finetuning.py

from trainers.base_trainer import BaseTrainer

class FullFinetuneTrainer(BaseTrainer):
    def __init__(self, config, model, train_loader, test_loader, logger):
        super().__init__(config, model, train_loader, test_loader, logger)
