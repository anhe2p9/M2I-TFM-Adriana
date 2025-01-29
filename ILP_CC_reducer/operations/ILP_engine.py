import csv
import utils

import pyomo.environ as pyo # ayuda a definir y resolver problemas de optimización

from ILP_CC_reducer.models.multiobjILPmodel import MultiobjectiveILPmodel
from ILP_CC_reducer.models.ILPmodelRsain import ILPmodelRsain

from ILP_CC_reducer.CC_reducer.ILP_CCreducer import ILPCCReducer
from ILP_CC_reducer.algorithms import __all__ as ALGORITHMS_NAMES



class ILPEngine():

    def __init__(self) -> None:
        pass

    def get_operations(self) -> list[ILPCCReducer]:
        """Return the list of all ILP operations available."""
        return [self.get_operation_from_name(ref_name) for ref_name in ALGORITHMS_NAMES]
    
    def load_concrete(self, datapath: str) -> None:
        # data.load(filename=file_path, index=model.S, param=(model.loc, model.nmcc))
        pass
    
    def apply_defined_weighted_sum(self, algorithm: ILPCCReducer, model: MultiobjectiveILPmodel | ILPmodelRsain, data: pyo.ConcreteModel, w1: int, w2: int, w3:int) -> None:
        """Apply weighted sum algorithm with specified weights to the given instance."""
        return algorithm.execute(algorithm, model, data, w1, w2, w3)

    def apply_succesive_weighted_sum(self, algorithm: ILPCCReducer, model: MultiobjectiveILPmodel | ILPmodelRsain, data: pyo.ConcreteModel, subdivisions: int) -> None:
        """Apply weighted sum algorithm a given number of times (subdivisions) with different weights."""
        
        csv_data = [["Weight1","Weight2","Weight3","Num.sequences","LOCdif","CCdif"]]
        
        for i in range(subdivisions):
            for j in range(subdivisions):
                weights = utils.generate_weights(subdivisions, j, i)
                
                self.apply_defined_weighted_sum(algorithm, model, data, weights["w1"], weights["w2"], weights["w3"])
                
                
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
                

        
        print(csv_data)
        
        # Escribir datos en un archivo CSV
        with open("C:/Users/X1502/eclipse-workspace/git/M2I-TFM-Adriana/resultsWS.csv", mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerows(csv_data)
        
        print("Archivo CSV creado correctamente.")
        

    def apply_epsilon_constraint(self, algorithm: ILPCCReducer, model: pyo.AbstractModel, data: pyo.ConcreteModel) -> None:
        """Applies epsilon constraint algorithm to the given instance."""
        pass
    
    # def apply_biesbinsky(self, algorithm: ILPCCReducer, model: pyo.AbstractModel, data: pyo.ConcreteModel) -> None:
    #     """Apply NO SÉ BIEN CUÁL."""
    #     pass
    
    def apply_TPA(self, algorithm: ILPCCReducer, model: pyo.AbstractModel, data: pyo.ConcreteModel,) -> None:
        """Applies TPA algorithm from SOTA to the given instance."""
        pass