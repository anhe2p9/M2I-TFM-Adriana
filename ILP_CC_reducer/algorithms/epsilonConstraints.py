import pyomo.environ as pyo # ayuda a definir y resolver problemas de optimización
import pyomo.dataportal as dp # permite cargar datos para usar en esos modelos de optimización

from typing import Any

from ILP_CC_reducer.Algorithm.Algorithm import Algorithm
from ILP_CC_reducer.models import MultiobjectiveILPmodel




class EpsilonConstraintAlgorithm(Algorithm):

    @staticmethod
    def get_name() -> str:
        return 'Epsilon Constraint Algorithm'
    
    @staticmethod
    def get_description() -> str:
        return ("It obtains supported and non-supported ILP solutions.")

    @staticmethod
    def execute(model: pyo.AbstractModel, data: dp.DataPortal, *args) -> list[list[Any]]:
        
        multiobj_model = MultiobjectiveILPmodel()
        concrete = model.create_instance(data)
        
        # Process args        
        epsilon = args[0]
        epsilon1 = epsilon[0]
        epsilon2 = epsilon[1]
        beta = args[1]     
         
        
        concrete.epsilon[1]=epsilon1
        concrete.epsilon[2]=100
        concrete.beta[1]=beta
        concrete.beta[2]=1
        
        solver=pyo.SolverFactory('cplex')
        results = solver.solve(concrete)
        
        if results.solver.status == 'ok':
            f1 = pyo.value(multiobj_model.LOCdifferenceObjective(concrete))
            f2 = pyo.value(multiobj_model.CCdifferenceObjective(concrete))
            
            print(f"f1: {f1}")
            print(f"f2: {f2}")
    
            pf1 = 10
            solution=1
            while results.solver.status == 'ok' and f1 > epsilon2 and pf1 > f1:
                solution = solution +1
                concrete.epsilon[1]=f1-1
                results = solver.solve(concrete)
                pf1=f1
                f1 = pyo.value(multiobj_model.LOCdifferenceObjective(concrete))
                f2 = pyo.value(multiobj_model.CCdifferenceObjective(concrete))
                
                print('Sequences selected:')
                for s in concrete.S:
                    print(f"x[{s}] = {concrete.x[s].value}")
