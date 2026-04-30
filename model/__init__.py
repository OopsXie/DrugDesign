"""
Model factory for DrugMSM
Provides flexible model selection interface
"""

import importlib
from loguru import logger


def get_model_class(model_name='mft'):
    """
    Get model class by name.
    
    Args:
        model_name (str): Model name, options: 'mft', 'transformer', 'encoder_only'
    
    Returns:
        Model class
    """
    model_mapping = {
        'mft': ('Lmser_Transformerr', 'MFT'),
        'transformer': ('Transformer', 'MFT'),
        'encoder_only': ('Transformer_Encoder', 'MFT'),
    }
    
    if model_name not in model_mapping:
        raise ValueError(f"Unknown model: {model_name}. Available options: {list(model_mapping.keys())}")
    
    module_name, class_name = model_mapping[model_name]
    try:
        module = importlib.import_module(f'.{module_name}', package='model')
        model_class = getattr(module, class_name)
        logger.info(f"Loaded model: {model_name} (class: {class_name} from {module_name}.py)")
        return model_class
    except (ImportError, AttributeError) as e:
        raise RuntimeError(f"Failed to load model {model_name}: {e}")


__all__ = ['get_model_class']
