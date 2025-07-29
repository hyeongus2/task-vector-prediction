# trainers/pretraining.py

from .full_finetuning import FullFinetuneTrainer

class PretrainTrainer(FullFinetuneTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger.info("[INFO] Pretraining mode — using full fine-tuning logic for now.")
        # TODO: Customize for self-supervised training if needed.
