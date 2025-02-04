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
        
        if hasattr(model, 'O'):
            model.del_component('O')
        model.add_component('O', pyo.Set(initialize=[1,2,3])) # Set for epsilon-constraint
        
        if hasattr(model, 'epsilon'):
            model.del_component('epsilon')
        model.add_component('epsilon', pyo.Param(model.O, initialize={1: 1.0, 2: 2.0, 3: 1.0}, mutable=True)) # Epsilon values
        
        if hasattr(model, 'beta'):
            model.del_component('beta')
        model.add_component('beta', pyo.Param(model.O, within=pyo.NonNegativeReals, initialize={1:1.0, 2:0.1, 3:0.1}, mutable=True)) # Weights for the objective function
        
        
        if hasattr(model, 'epsilonConstraint'):
            model.del_component('epsilonConstraint')
        model.add_component('epsilonConstraint', pyo.Constraint(model.O, rule=multiobj_model.epsilonConstraint))
        
        print(f"ARGS EPSILON CONSTRAINT: {args}")
        
        # Process args        
        epsilon = args[0]
        epsilon1 = epsilon[0]
        epsilon2 = epsilon[1]
        epsilon3 = epsilon[2]
        
        beta = args[1]   
        beta1 = beta[0]
        beta2 = beta[1]
        
        
        
        if hasattr(model, 'obj'):
            model.del_component('obj')
        model.add_component('obj', pyo.Objective(rule=lambda m: multiobj_model.epsilonObjective(m)))
         
        
        concrete = model.create_instance(data)
        concrete.epsilon[1] = epsilon1
        concrete.epsilon[2] = epsilon2
        concrete.epsilon[3] = epsilon3
        concrete.beta[1] = beta1
        concrete.beta[2] = beta2
        concrete.beta[3] = 1
        
        solver=pyo.SolverFactory('cplex')
        results = solver.solve(concrete)
        concrete.pprint()
        
        
        
        if results.solver.status == 'ok':
            f1 = pyo.value(multiobj_model.LOCdifferenceObjective(concrete))
            f2 = pyo.value(multiobj_model.CCdifferenceObjective(concrete))
            
            print(f"f1: {f1}")
            print(f"f2: {f2}")
    
            pf1 = 10
            solution=1
            
            print(f"comparison: f1: {f1}, epsilon2: {epsilon2} and pf1: {pf1}, f1: {f1}")
            
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
