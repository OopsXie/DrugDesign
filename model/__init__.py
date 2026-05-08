"""
Model factory for DrugGAN-MSM
Provides flexible model selection interface
"""

import importlib
from loguru import logger


def get_model_class(model_name='drug_gan_msm'):
    """
    Get model class by name.

    Args:
        model_name (str): Model name, options:
            'drug_gan_msm', 'kg_drug_gan_msm'

    Returns:
        Model class
    """
    model_mapping = {
        'drug_gan_msm': ('DrugGAN_MSM', 'Generator'),
        'kg_drug_gan_msm': ('KG_DrugGAN_MSM', 'KGDrugGANModel'),
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
