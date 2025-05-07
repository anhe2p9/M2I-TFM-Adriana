import inspect
import importlib
from typing import Any

# import sys
# import csv
# import os
from pathlib import Path

import pyomo.environ as pyo
import pyomo.dataportal as dp # permite cargar datos para usar en esos modelos de optimización

# from ILP_CC_reducer.models.multiobjILPmodel import MultiobjectiveILPmodel
# from ILP_CC_reducer.models.ILPmodelRsain import ILPmodelRsain

from ILP_CC_reducer.algorithm.algorithm import Algorithm
from ILP_CC_reducer import algorithms as ALGORITHMS
from ILP_CC_reducer.algorithms import __all__ as ALGORITHMS_NAMES



class ILPEngine():

    def __init__(self) -> None:
        pass

    def get_algorithms(self) -> list[Algorithm]:
        """Return the list of all ILP operations available."""
        return [self.get_algorithm_from_name(ref_name) for ref_name in ALGORITHMS_NAMES]
    
    def load_concrete(self, data_folder: Path, model: pyo.AbstractModel) -> dp.DataPortal:
        
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
                elif clear_name.endswith("feasible_extractions_offsets"):
                    files["feasible_extractions_offsets"] = file


        data = model.process_data(str(files["sequences"]), str(files["nested"]), str(files["conflict"]), str(files["feasible_extractions_offsets"]))
        
        return data # files_status_dict = {"missingFiles": lista_de_archivos_que_faltan, "emtyFiles": lista_de_archivos_vacios, "data": data, "offsets": Offsets_filename}
        
        
        

    def apply_algorithm(self, algorithm: Algorithm, instance: dp.DataPortal, tau: int, *args) -> Any:
        """Apply the given algorithm to the given model instance."""
        
        args_list = tuple(item for item in args if item)
        
        return algorithm.execute(instance, tau, *args_list)
    
    # def apply_algorithms(self, algorithm: algorithm, ILPm: pyo.AbstractModel, instance: dp.DataPortal, tau: int, *args) -> None:
    #     """Apply the given algorithm to all model instances."""
    #     # TODO: dado un conjunto de instancias, aplicar a todas ellas el correspondiente algoritmo y generar el csv de resultados
    #     # la verdad es que no tengo claro si hacer los multiobjetivo así, creo que lo voy a enfocar más en hacer cada uno por separado
    #     # y sacar el frente de pareto o algo así
    #     pass
    
    def apply_rsain_model(self, algorithm: Algorithm, model: pyo.ConcreteModel, data: dp.DataPortal, tau: str, csv_data: list[str], folders_data: dict) -> None:
        """Creates a csv with the results data."""
        if hasattr(model, 'tau'):
            model.del_component('tau')
            model.add_component('tau', pyo.Param(within=pyo.NonNegativeReals, initialize=int(tau), mutable=False)) # Threshold
        else:
            model.add_component('tau', pyo.Param(within=pyo.NonNegativeReals, initialize=int(tau), mutable=False)) # Threshold
    
    
        return algorithm.execute(model, data, csv_data, folders_data)

    
    
    def get_algorithm_from_name(self, algorithm_name: str) -> Algorithm:
        """Given the name of an algorithm class, return the instance class of the algorithm."""
        modules = inspect.getmembers(ALGORITHMS)
        modules = filter(lambda x: inspect.ismodule(x[1]), modules)
        modules = [importlib.import_module(m[1].__name__) for m in modules]
        class_ = next((getattr(m, algorithm_name) for m in modules if hasattr(m, algorithm_name)), None)
        return class_
    
