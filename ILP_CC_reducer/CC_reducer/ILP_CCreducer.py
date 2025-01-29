from abc import abstractmethod, ABC
from typing import Any
import pyomo.environ as pyo


class ILPCCReducer(ABC):
    
    @staticmethod
    @abstractmethod
    def define_sets():
        """Defines model sets."""
        pass
    
    @staticmethod
    @abstractmethod
    def define_parameters():
        """Defines model parameters."""
        pass
    
    @staticmethod
    @abstractmethod
    def define_variables():
        """Defines model variables."""
        pass
    
    @staticmethod
    @abstractmethod
    def define_constraints():
        """Defines model constraints."""
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