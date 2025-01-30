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
    def execute(model: pyo.AbstractModel, data: dp.DataPortal, tau:int, *args) -> list[list[Any]]:
        
        csv_data = [["Weight1","Weight2","Weight3","Num.sequences","LOCdif","CCdif"]]
        ilp_model = MultiobjectiveILPmodel(tau)
        
        if len(args) == 1 and isinstance(args[0], int):
            args = args[0]
            print(f"Proccessing all ILP results with {args} subdivisions")
    
            for i in range(args+1):
                for j in range(args+1):
                    weights = utils.generate_weights(args, i, j)
                    
                    if hasattr(model, 'obj'):
                        model.del_component('obj')  # Eliminar el componente existente
                        model.add_component('obj', pyo.Objective(rule=lambda m: weightedSum(m, weights['w1'], weights['w2'], weights['w3'], ilp_model)))
                    else:
                        model.obj = pyo.Objective(rule=lambda m: weightedSum(m, weights['w1'], weights['w2'], weights['w3'], ilp_model))
                    
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
                    
                    
                    newrow = [round(weights["w1"],2),round(weights["w2"],2),round(weights["w3"],2),sequences_sum,LOCdif,CCdif]
                    
                    csv_data.append(newrow)
                
                    if i == 0:
                        break    
            return csv_data
            
            
            
        elif len(args) == 3 and all(isinstance(arg, float) for arg in args):
            print(f"Proccessing the optimal ILP solution with weights: {args}")
            
            w1, w2, w3 = args
            if hasattr(model, 'obj'):
                model.del_component('obj')  # Eliminar el componente existente
                model.add_component('obj', pyo.Objective(rule=lambda m: model.weightedSum(m, w1, w2, w3)))
            else:
                model.obj = pyo.Objective(rule=lambda m: model.weightedSum(m, w1, w2, w3))
        
            concrete = model.create_instance(data) # para crear una instancia de modelo y hacerlo concreto
            solver = pyo.SolverFactory('cplex')
            # results = solver.solve(concrete)
            solver.solve(concrete)
        
            newrow = [round(weights["w1"],2),round(weights["w2"],2),round(weights["w3"],2),sequences_sum,LOCdif,CCdif]        
            csv_data.append(newrow)
                    
            return csv_data
        
            
        else:
            sys.exit(f'The algorithm parameters must be a number of subdivisions s or three weights w1,w2,w3.')

    # concrete = model.create_instance(data) # para crear una instancia de modelo y hacerlo concreto
    #
    # solver = pyo.SolverFactory('cplex')
    # results = solver.solve(concrete)
    # concrete.pprint()
    #
    # num_constraints = sum(len(constraint) for constraint in concrete.component_objects(Constraint, active=True))
    # print(f"There are {num_constraints} constraints")
    # if (results.solver.status == 'ok'):
    #     print('Optimal solution found')
    #     print('Objective value: ', pyo.value(concrete.obj))
    #     print('Sequences selected:')
    #     for s in concrete.S:
    #         print(f"x[{s}] = {concrete.x[s].value}")
    




    # Generar las subdivisiones
    # n_divisions = 6
    # theta_div = 2  # índice de 0 a n_divisions-1
    # phi_div = 0  # índice de 0 a n_divisions-1
    # weights = utils.generate_weights(n_divisions, theta_div, phi_div)
    # weights = utils.generate_weights(6,6,6)
    #
    # # print(weights)
    # # print("Variables: ", weights["w1"], weights["w2"], weights["w3"])
    #
    #
    #
    # model.obj = pyo.Objective(rule=lambda m: weightedSum(m, weights["w1"], weights["w2"], weights["w3"]))
    
    # model.min_LOC_difference = pyo.Constraint(model.S, rule=min_LOC_difference) # ESTO SOLO PARA EPSILON-CONSTRAINT
    # model.min_CC_difference = pyo.Constraint(model.S, rule=min_CC_difference) # ESTO SOLO PARA EPSILON-CONSTRAINT
    

def weightedSum(m, sequencesWeight, LOCdiffWeight, CCdiffWeight, ilp_model):
    return (sequencesWeight * ilp_model.sequencesObjective(m) +
            LOCdiffWeight * ilp_model.LOCdifferenceObjective(m) +
            CCdiffWeight * ilp_model.CCdifferenceObjective(m))




