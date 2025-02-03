import inspect
import importlib
from typing import Any

import sys
from pathlib import Path

import pyomo.environ as pyo
import pyomo.dataportal as dp # permite cargar datos para usar en esos modelos de optimización

from ILP_CC_reducer.models.multiobjILPmodel import MultiobjectiveILPmodel

from ILP_CC_reducer.Algorithm.Algorithm import Algorithm
from ILP_CC_reducer import algorithms as ALGORITHMS
from ILP_CC_reducer.algorithms import __all__ as ALGORITHMS_NAMES



class ILPEngine():

    def __init__(self) -> None:
        pass

    def get_algorithms(self) -> list[Algorithm]:
        """Return the list of all ILP operations available."""
        return [self.get_algorithm_from_name(ref_name) for ref_name in ALGORITHMS_NAMES]
    
    def load_concrete(self, data_folder: Path) -> dp.DataPortal:
        
        model = MultiobjectiveILPmodel()
        
        files = { "sequences": None, "nested": None, "conflict": None }

        for file in data_folder.iterdir():
            if file.is_file():
                clear_name = file.stem  # Name without extension
                
                if clear_name.endswith("sequences"):
                    files["sequences"] = file
                elif clear_name.endswith("nested"):
                    files["nested"] = file
                elif clear_name.endswith("conflict"):
                    files["conflict"] = file
    
        # Verificar si todos los archivos han sido encontrados
        if None in files.values():
            sys.exit(f'The model instance must be a folder with three CSV files.')
        
        data = model.process_data(str(files["sequences"]), str(files["nested"]), str(files["conflict"]))
        
        return data
        

    def apply_algorithm(self, algorithm: Algorithm, ILPm: pyo.AbstractModel, instance: dp.DataPortal, tau: int, *args) -> Any:
        """Apply the given refactoring to the given instance (feature or constraint) of the given FM."""
        
        if not hasattr(ILPm, 'tau'):
            ILPm.add_component('tau', pyo.Param(within=pyo.NonNegativeReals, initialize=int(tau), mutable=False)) # Threshold
        
        if len(args) == 0:
            return algorithm.execute(ILPm, instance)
        else:
            args = args[0]
            
            # Process weights                
            if isinstance(args, tuple):
                # weights = list(map(float, args.weights.split(',')))
                if len(args) != 3:
                    sys.exit("Weights parameter w1,w2,w3 must be exactly three weights separated by comma (',').")
                return algorithm.execute(ILPm, instance, args)
            elif isinstance(args, int):
                subdivisions = int(args)
                return algorithm.execute(ILPm, instance, subdivisions)
            else:
                sys.exit("Weights parameter w1,w2,w3 must be exactly three weights separated by comma (',').\nSubdivisions must be an integer parameter.")

    
    def get_algorithm_from_name(self, algorithm_name: str) -> Algorithm:
        """Given the name of an algorithm class, return the instance class of the algorithm."""
        modules = inspect.getmembers(ALGORITHMS)
        modules = filter(lambda x: inspect.ismodule(x[1]), modules)
        modules = [importlib.import_module(m[1].__name__) for m in modules]
        class_ = next((getattr(m, algorithm_name) for m in modules if hasattr(m, algorithm_name)), None)
        return class_
    
