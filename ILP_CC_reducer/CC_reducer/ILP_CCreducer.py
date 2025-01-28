from abc import abstractmethod, ABC
from typing import Any
import pyomo.environ as pyo


class ILPCCReducer(ABC):

    @staticmethod
    @abstractmethod
    def get_name() -> str:
        """Name of the ILP algorithm."""
        pass

    @staticmethod
    @abstractmethod
    def get_description() -> str:
        pass

    @staticmethod
    @abstractmethod
    def get_method_to_operate_name() -> str:
        pass

    @staticmethod
    @abstractmethod
    def execute(model: pyo.AbstractModel, data: pyo.ConcreteModel, subdivisions: int) -> list[list[Any]]:
        """Apply the ILP algorithm to the given instance."""
        pass
    
    @staticmethod
    @abstractmethod
    def definir_objetivo(self, model: pyo.ConcreteModel, **kwargs):
        """Defines model objective."""
        pass