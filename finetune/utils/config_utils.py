import importlib
import os


def get_config_class():
    """
    Resolve the Config class from a module specified by KRONOS_CONFIG_MODULE.

    Default module is "config" (finetune/config.py).
    """
    module_name = os.getenv("KRONOS_CONFIG_MODULE", "config")
    module = importlib.import_module(module_name)
    if not hasattr(module, "Config"):
        raise AttributeError(f"Module '{module_name}' does not define a Config class.")
    return module.Config
