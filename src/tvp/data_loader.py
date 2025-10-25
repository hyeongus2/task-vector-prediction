# src/tvp/data_loader.py
import logging
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import CLIPProcessor
from typing import List, Tuple, Dict
from .utils import set_seed

# Create a logger for this module
logger = logging.getLogger(__name__)


def get_dataloaders(config: Dict, processor: CLIPProcessor) -> Tuple[DataLoader, DataLoader, List[str]]:
    """
    Loads a dataset from the Hugging Face Hub, preprocesses it for CLIP, 
    and returns train/validation DataLoaders along with the list of class names.

    This function is designed to be generic, handling various dataset structures
    and splitting strategies based on the provided configuration.

    Args:
        config (Dict): A dictionary containing configuration parameters.
                       It's expected to have a nested 'data' key with sub-keys like
                       'name', 'image_column_name', 'label_column_name', 
                       'split_strategy', 'validation_split_size', 'batch_size', 'num_workers'.
        processor (CLIPProcessor): The CLIPProcessor for preprocessing images.

    Returns:
        Tuple[DataLoader, DataLoader, List[str]]: A tuple containing the
        training dataloader, validation dataloader, and the list of class names.
    """
    set_seed(config.get('seed', 42))
    
    # Access nested keys from the config file
    data_config = config['data']
    dataset_name = data_config['name']
    image_col = data_config['image_column_name']
    label_col = data_config['label_column_name']
    
    logger.info(f"Loading dataset: {dataset_name}")
    
    # Load the full dataset from Hugging Face Hub
    dataset = load_dataset(dataset_name)

    # Handle different dataset split strategies based on the config
    if data_config['split_strategy'] == 'pre_split' and 'test' in dataset:
        logger.info("Using pre-defined train and test splits.")
        train_dataset = dataset['train']
        val_dataset = dataset['test']
    elif data_config['split_strategy'] == 'auto_split':
        val_split_size = data_config.get('validation_split_size', 0.2)
        logger.info(f"Automatically splitting 'train' data. Validation size: {val_split_size}")
        # Split the 'train' dataset into a new training and validation set
        split_dataset = dataset['train'].train_test_split(test_size=val_split_size)
        train_dataset = split_dataset['train']
        val_dataset = split_dataset['test'] # This is the new validation set
    else:
        raise ValueError(f"Unsupported split strategy or missing splits for {dataset_name}")

    # Extract class names from the dataset features
    try:
        class_names = train_dataset.features[label_col].names
        
        # Create a preview string for logging
        if len(class_names) > 5:
            preview = f"['{class_names[0]}', '{class_names[1]}', ..., '{class_names[-1]}']"
        else:
            preview = str(class_names)
        logger.info(f"Found {len(class_names)} classes: {preview}")

    except (KeyError, AttributeError):
        # Fallback to config if class names are not in dataset features
        if 'class_names' in data_config:
            class_names = data_config['class_names']
            logger.info(f"Loaded {len(class_names)} classes from config file.")
        else:
            logger.error(f"Could not automatically extract class names for column '{label_col}'.")
            raise ValueError("Please ensure the dataset has class name features or provide them in the config.")
    
    # Preprocessing function that will be applied to the dataset
    def transform(examples):
        processed = processor(images=examples[image_col], return_tensors="pt", padding=True, truncation=True)
        processed['labels'] = examples[label_col]
        return processed

    # Apply the transformation to the datasets
    remove_cols = [image_col, label_col]
    num_map_workers = data_config.get('num_workers', 4)
    logger.info(f"Applying transformation to datasets using {num_map_workers} processes...")
    train_dataset = train_dataset.map(transform, batched=True, num_proc=num_map_workers, remove_columns=remove_cols)
    val_dataset = val_dataset.map(transform, batched=True, num_proc=num_map_workers, remove_columns=remove_cols)
    
    # Set the dataset format to return PyTorch tensors
    train_dataset.set_format(type='torch')
    val_dataset.set_format(type='torch')
    
    logger.info("Creating PyTorch DataLoaders.")
    # Create the final DataLoader objects
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=data_config['batch_size'],
        shuffle=True,
        num_workers=data_config['num_workers']
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=data_config['batch_size'],
        shuffle=False,
        num_workers=data_config['num_workers']
    )
    
    return train_dataloader, val_dataloader, class_names