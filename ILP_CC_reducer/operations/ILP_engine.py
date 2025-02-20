import inspect
import importlib
from typing import Any

import sys
from pathlib import Path

import os
import csv

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
        """Apply the given algorithm to the given model instance."""
        
        if not hasattr(ILPm, 'tau'):
            ILPm.add_component('tau', pyo.Param(within=pyo.NonNegativeReals, initialize=int(tau), mutable=False)) # Threshold
            
        args_list = tuple(item for item in args if item)
        
        return algorithm.execute(ILPm, instance, *args_list)
    
    def apply_algorithms(self, algorithm: Algorithm, ILPm: pyo.AbstractModel, instance: dp.DataPortal, tau: int, *args) -> list[list[Any]]:
        """Apply the given algorithm to all model instances."""
        #
        # csv_data = [["class", "method", "initialComplexity", "solution", "extractions", "notNestedSolution", "notNestedExtractions", 
        #      "reductionComplexity", "finalComplexity",
        #      "minExtractedLOC", "maxExtractedLOC", "meanExtractedLOC", "totalExtractedLOC", 
        #      "minReductionOfCC", "maxReductionOfCC", "meanReductionOfCC", "totalReductionOfCC", 
        #      "modelStatus", "executionTime"]]
        #
        #
        # # Base directory of the project
        # base_dir = Path(__file__).resolve().parent.parent
        #
        # instance_folder = base_dir / "original_code_data"
        #
        # for project_folder in sorted(os.listdir(instance_folder)):
        #     project_folder = Path(project_folder)
        #     print(f"Project folder: {project_folder}")
        #     total_path = instance_folder / project_folder
        #     for class_folder in sorted(os.listdir(total_path)):
        #         class_folder = Path(class_folder)
        #         print(f"Class folder: {class_folder}")
        #         total_path = instance_folder / project_folder / class_folder
        #         for method_folder in sorted(os.listdir(total_path)):
        #             method_folder = Path(method_folder)
        #             print(f"Method folder: {method_folder}")
        #             total_path = instance_folder / project_folder / class_folder / method_folder
        #             print(f"Total path: {total_path}")
        #             if os.path.isdir(total_path):
        #                 print(f"Processing Class_Method: {method_folder}")
        #                 results_csv = self.apply_algorithm(algorithm, ILPm, instance, tau, args)
        #
        #
        # # Escribir datos en un archivo CSV
        # with open("results.csv", mode="w", newline="", encoding="utf-8") as file:
        #     writer = csv.writer(file)
        #     writer.writerows(results_csv)
        #
        # print("Archivo CSV creado correctamente.")
        #
        #

        pass
    
    
    
    def get_algorithm_from_name(self, algorithm_name: str) -> Algorithm:
        """Given the name of an algorithm class, return the instance class of the algorithm."""
        modules = inspect.getmembers(ALGORITHMS)
        modules = filter(lambda x: inspect.ismodule(x[1]), modules)
        modules = [importlib.import_module(m[1].__name__) for m in modules]
        class_ = next((getattr(m, algorithm_name) for m in modules if hasattr(m, algorithm_name)), None)
        return class_
    
