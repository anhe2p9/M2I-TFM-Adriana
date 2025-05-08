from abc import ABC, abstractmethod

from typing import Any

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
    def execute(data: dp.DataPortal, *args) -> list[list[Any]]:
        """Apply the algorithm to the given model instance."""
        pass
    
