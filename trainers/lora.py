# trainers/lora.py

from .full_finetuning import FullFinetuneTrainer

class LoRATrainer(FullFinetuneTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger.info("[INFO] LoRA trainer not yet implemented — defaulting to full fine-tuning logic.")
        # TODO: Apply LoRA modules here if needed.
