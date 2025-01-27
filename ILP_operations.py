from abc import abstractmethod, ABC
from typing import Any


class ILPOperations(ABC):

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
    def reduce_CC(code: Any, instance: Any) -> Any:
        """Apply the ILP algorithm to the given instance."""
        pass