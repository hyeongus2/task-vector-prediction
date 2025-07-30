# trainers/lora.py

from peft import get_peft_model, LoraConfig, TaskType
from trainers.base_trainer import BaseTrainer
from models.model_utils import get_lora_target_modules

class LoRATrainer(BaseTrainer):
    def __init__(self, config, model, train_loader, test_loader, logger):
        # 1. Determine task type
        model_type = config["model"]["type"]
        data_task = config["data"]["task"]

        if model_type == "text":
            if data_task == "classification":
                return TaskType.SEQ_CLS
            elif data_task == "regression":
                return TaskType.REGRESSION
            elif data_task == "feature_extraction":
                return TaskType.FEATURE_EXTRACTION

        elif model_type == "image":
            return TaskType.FEATURE_EXTRACTION

        elif model_type in ["tabular", "synthetic"]:
            return TaskType.FEATURE_EXTRACTION

        else:
            raise ValueError(f"Unsupported model/data task combination: {model_type}, {data_task}")

        # 2. Find target modules using model_utils
        target_modules = config["lora"].get("target_modules", get_lora_target_modules(model))

        # 3. Build LoRA config
        peft_config = LoraConfig(
            task_type=data_task,
            inference_mode=False,
            r=config["lora"].get("r", 8),
            lora_alpha=config["lora"].get("alpha", 16),
            lora_dropout=config["lora"].get("dropout", 0.1),
            bias="none",
            target_modules=target_modules
        )

        # 4. Apply LoRA
        model = get_peft_model(model, peft_config)

        # 5. Continue as BaseTrainer
        super().__init__(config, model, train_loader, test_loader, logger)

        self.logger.info("LoRA applied. Trainable params:")
        self.model.print_trainable_parameters()
