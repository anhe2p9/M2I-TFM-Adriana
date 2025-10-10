from abc import ABC, abstractmethod

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
    def execute(data: dict, tau:int, *args, **kwargs):
        """Apply the algorithm to the given model instance."""
        pass
    
