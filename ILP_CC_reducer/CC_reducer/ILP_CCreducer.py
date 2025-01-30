from abc import abstractmethod, ABC
from typing import Any
import pyomo.environ as pyo


class ILPCCReducer(ABC):
    
    @staticmethod
    @abstractmethod
    def define_model_without_obj():
        """Defines model except objective."""
        pass
    
    @staticmethod
    @abstractmethod
    def define_objectives():
        """Defines objective functions of the model."""
        pass
    
    @staticmethod
    @abstractmethod
    def process_data(S_filename: str, N_filename: str, C_filename: str):
        """Processes data from DataPortal."""
        pass

    # @staticmethod
    # @abstractmethod
    # def execute(model: pyo.AbstractModel, data: pyo.ConcreteModel, subdivisions: int) -> list[list[Any]]:
    #     """Apply the ILP algorithm to the given instance."""
    #     pass
    #
    # @staticmethod
    # @abstractmethod
    # def definir_objetivo(model: pyo.ConcreteModel, **kwargs):
    #     """Defines model objective."""
    #     pass