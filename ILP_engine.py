import inspect
import importlib

from flamapy.metamodels.fm_metamodel.models import FeatureModel, Feature, Constraint

from fm_refactorings.models import FMRefactoring
from fm_refactorings import transformations as REFACTORINGS
from fm_refactorings.transformations import __all__ as REFACTORINGS_NAMES
from numpy import integer


class ILPEngine():

    def __init__(self) -> None:
        pass

    def get_operations(self) -> list[ILPOperations]:
        """Return the list of all ILP operations available."""
        return [self.get_operation_from_name(ref_name) for ref_name in OPERATIONS_NAMES]
    
    def apply_operation(self, refactoring: ILPOperations, files: ModelInstance, threshold: integer, weights: (w1,w2,w3)) -> Solution:
        """Apply the given algorithm to the given instance of model."""
        return refactoring.transform(files, threshold)

    def apply_operations(self, refactoring: FMRefactoring, files: ModelInstance, threshold: integer, subdivisions: integer) -> Solution:
        """Apply the given algorithm a given number of times (subdivisions)."""
        instances = refactoring.get_instances(fm)
        for instance in instances:
            print(f'|->Applying {refactoring.get_name()} to instance {str(instance)}...')
            fm = self.apply_refactoring(refactoring, fm, instance)
        return fm

    def get_operation_from_name(self, refactoring_name: str) -> FMRefactoring:
        """Given the name of an operation class, return the instance class of the ."""
        modules = inspect.getmembers(REFACTORINGS)
        modules = filter(lambda x: inspect.ismodule(x[1]), modules)
        modules = [importlib.import_module(m[1].__name__) for m in modules]
        class_ = next((getattr(m, refactoring_name) for m in modules if hasattr(m, refactoring_name)), None)
        return class_