from abc import ABC, abstractmethod

from typing import Any

import pyomo.environ as pyo # ayuda a definir y resolver problemas de optimización
import pyomo.dataportal as dp # permite cargar datos para usar en esos modelos de optimización



class Algorithm(ABC):
    
    @staticmethod
    @abstractmethod
    def get_name() -> str:
        """Name of the algorithm"""
        pass
    
    @staticmethod
    @abstractmethod
    def get_description() -> str:
        pass

    @staticmethod
    @abstractmethod
    def execute(model: pyo.AbstractModel, data: dp.DataPortal, *args) -> list[list[Any]]:
        """Apply the algorithm to the given model instance."""
        pass
    
