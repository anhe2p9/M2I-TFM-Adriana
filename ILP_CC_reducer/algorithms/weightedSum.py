import pyomo.environ as pyo # ayuda a definir y resolver problemas de optimización
import pyomo.dataportal as dp # permite cargar datos para usar en esos modelos de optimización

import sys
import utils
from typing import Any

from ILP_CC_reducer.CC_reducer.ILP_CCreducer import ILPCCReducer
from ILP_CC_reducer.models.multiobjILPmodel import MultiobjectiveILPmodel


class WeightedSumAlgorithm(ILPCCReducer):

    @staticmethod
    def get_name() -> str:
        return 'Weigthed Sum Algorithm'
    
    @staticmethod
    def get_description() -> str:
        return ("It obtains soported ILP solutions based on the given weights.")

    @staticmethod
    def execute(model: pyo.AbstractModel, data: dp.DataPortal, *args) -> list[list[Any]]:
        
        csv_data = [["Weight1","Weight2","Weight3","Num.sequences","LOCdif","CCdif"]]
        
        args = args[0]
        
 
        if isinstance(args, int):
            print(f"Proccessing all ILP results with {args} subdivisions")
    
            for i in range(args+1):
                for j in range(args+1):
                    w1, w2, w3 = utils.generate_weights(args, i, j)
                    
                    _, newrow = process_model(model, data, w1 ,w2, w3)
                    
                    csv_data.append(newrow)
                
                    if i == 0:
                        break    
            return csv_data
            
            
            
        elif all(isinstance(arg, float) for arg in args):
            print(f"Proccessing the optimal ILP solution with weights: {args}")
                        
            w1, w2, w3 = args
            
            concrete, newrow = process_model(model, data, w1 ,w2, w3)
                  
            csv_data.append(newrow)
            
            concrete.pprint()
                    
            return csv_data
        
            
        else:
            sys.exit(f'The algorithm parameters must be a number of subdivisions s or three weights w1,w2,w3.')



def process_model(model: pyo.AbstractModel, data: dp.DataPortal, w1 ,w2, w3):
    
    ilp_model = MultiobjectiveILPmodel()
    
    if hasattr(model, 'obj'):
        model.del_component('obj')  # Eliminar el componente existente
        model.add_component('obj', pyo.Objective(rule=lambda m: weightedSum(m, w1, w2, w3, ilp_model)))
    else:
        model.obj = pyo.Objective(rule=lambda m: weightedSum(m,w1, w2, w3, ilp_model))
    
    concrete = model.create_instance(data) # para crear una instancia de modelo y hacerlo concreto
    solver = pyo.SolverFactory('cplex')
    # results = solver.solve(concrete)
    solver.solve(concrete)
    

    sequences_sum = sum(concrete.x[i].value for i in concrete.S if i != 0)
    
    xLOC = [concrete.loc[i] for i in concrete.S if concrete.x[i].value == 1]
    zLOC = [concrete.loc[j] for j,ii in concrete.N if concrete.z[j,ii].value == 1]
    
    maxLOCselected = abs(max(xLOC) - max(zLOC))
    minLOCselected = min(concrete.loc[i] for i in concrete.S if concrete.x[i].value == 1)
    LOCdif = abs(maxLOCselected - minLOCselected)
    
    xCC = [concrete.nmcc[i] for i in concrete.S if concrete.x[i].value == 1]
    zCC = [concrete.ccr[j,ii] for j,ii in concrete.N if concrete.z[j,ii].value == 1]
    
    
    maxCCselected = abs(max(xCC) - max(zCC))
    minCCselected = min(concrete.nmcc[i] for i in concrete.S if concrete.x[i].value == 1)
    CCdif = abs(maxCCselected - minCCselected)
    
    
    newrow = [round(w1,2),round(w2,2),round(w3,2),sequences_sum,LOCdif,CCdif]
    
    return concrete, newrow
    

def weightedSum(m, sequencesWeight, LOCdiffWeight, CCdiffWeight, ilp_model):
    return (sequencesWeight * ilp_model.sequencesObjective(m) +
            LOCdiffWeight * ilp_model.LOCdifferenceObjective(m) +
            CCdiffWeight * ilp_model.CCdifferenceObjective(m))




