import pyomo.environ as pyo # ayuda a definir y resolver problemas de optimización
import pyomo.dataportal as dp # permite cargar datos para usar en esos modelos de optimización

import sys
import algorithms_utils
from typing import Any

import csv
import os

from ILP_CC_reducer.models.ILPmodelRsain import ILPmodelRsain
from ILP_CC_reducer.Algorithm.Algorithm import Algorithm


class addOneObjectiveAlgorithm(Algorithm):

    @staticmethod
    def get_name() -> str:
        return 'Solve Model with one objective'
    
    @staticmethod
    def get_description() -> str:
        return ("It obtains the solution for just sequences objective.")

    @staticmethod
    def execute(model: pyo.AbstractModel, data: dp.DataPortal) -> None:
        
        uniobj_model = ILPmodelRsain()
        
        if hasattr(model, 'obj'):
            model.del_component('obj')
        model.add_component('obj', pyo.Objective(rule=lambda m: uniobj_model.sequencesObjective(m)))
        
        
        
        concrete = model.create_instance(data) # para crear una instancia de modelo y hacerlo concreto
        solver = pyo.SolverFactory('cplex')
        results = solver.solve(concrete)
        
        concrete.pprint()
        
        print('===============================================================================')
        if (results.solver.status == 'ok'):
            print('Sequences selected:')
            for s in concrete.S:
                print(f"x[{s}] = {concrete.x[s].value}")
        print('===============================================================================')
    
        print(results)

    




