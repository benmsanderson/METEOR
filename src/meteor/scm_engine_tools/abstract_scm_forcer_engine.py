from abc import ABC, abstractmethod

from ciceroscm import concentrations_emissions_handler

from . import ciceroscm_forcer_engine
from . import fair_forcer_engine

def scm_engine_factory(scm_type: str, config=None):
    """Factory function to create SCM engine instances based on the SCM type.

    Args:
        scm_type (str): The type of SCM (e.g., 'git', 'svn', 'mercurial').

    Returns:
        An instance of the corresponding SCM engine class.

    Raises:
        ValueError: If the provided SCM type is not supported.
    """
    if scm_type == 'default' or scm_type == 'ciceroscm':
        return ciceroscm_forcer_engine.CiceroscmEngineForPatternScaling(config)

    if scm_type == "fair":
        return fair_forcer_engine.FairscmEngineForPatternScaling(config)

    raise ValueError(f"Unknown scm type: {scm_type}")

class SCMFacade:
    """Facade class to interact with different SCM engines."""

    def __init__(self, config=None):
        self.config = self._wet_and_set_configs(config)

    def _wet_and_set_configs(self, config):
        """Wet and set configurations for the SCM engine.

        Args:
            config (dict): Configuration dictionary.
        Returns:
            The processed configuration dictionary.
        """        # Implement any necessary processing of the configuration here
        if config is None:
            config = {"scm_type": "default"}
        if "scm_type" not in config or config["scm_type"] not in ["default", "ciceroscm", "fair"]:
            config["scm_type"] = "default"
        if config["scm_type"] == "default":
            self.scm_type = "ciceroscm"
        else:
            self.scm_type = config["scm_type"]
        
        config.pop("scm_type", None)  # Remove scm_type from config as it's now stored in self.scm_type

        if self.scm_type == "ciceroscm":
            config = concentrations_emissions_handler.check_pamset(config)

        return config

    def run_to_get_scaling(self, exp_list):
        """Run the appropriate SCM engine to get scaling data.

        Args:
            exp_list (list): List of experiments.

        Returns:
            Scaling data from the selected SCM engine.
        """
        if self.scm_type == "ciceroscm":
            scm_engine = ciceroscm_forcer_engine.CiceroscmEngineForPatternScaling(self.config)
        return scm_engine.run_to_get_scaling(exp_list)
    
    def run_and_return_per_forcer_results(exp_list):
        """Run the SCM engine and return per-forcer results.

        Args:
            exp_list (list): List of experiments.

        Returns:
            Per-forcer results from the SCM engine.
        """
        scm_type = self.config.get("scm_type", "default") if self.config else "default"
        scm_engine = scm_engine_factory(scm_type, self.config)
        return scm_engine.run_and_return_per_forcer_results(exp_list)

class AbstractScmForcerEngine(ABC):
    """Abstract base class for SCM forcer engines."""

    def __init__(self, config=None):
        self.config = config