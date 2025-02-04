import pyomo.environ as pyo # ayuda a definir y resolver problemas de optimización
import pyomo.dataportal as dp # permite cargar datos para usar en esos modelos de optimización

from typing import Any

from ILP_CC_reducer.Algorithm.Algorithm import Algorithm
from ILP_CC_reducer.models import MultiobjectiveILPmodel
from pyomo.solvers.tests.solvers import initialize




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
        
        # if hasattr(model, 'O'):
        #     model.del_component('O')
        # model.add_component('O', pyo.Set(initialize=[1,2,3])) # Set for epsilon-constraint
        #
        # if hasattr(model, 'epsilon'):
        #     model.del_component('epsilon')
        # model.add_component('epsilon', pyo.Param(model.O, initialize={1: 1.0, 2: 2.0}, mutable=True)) # Epsilon values
        #
        # if hasattr(model, 'lambda'):
        #     model.del_component('lambda')
        # model.add_component('lambda', pyo.Param(model.O, within=pyo.NonNegativeReals, initialize={1:1.0, 2:2.0}, mutable=True)) # Weights for the objective function
        #
        #
        # if hasattr(model, 'epsilonConstraint'):
        #     model.del_component('epsilonConstraint')
        # model.add_component('epsilonConstraint', pyo.Constraint(model.O, rule=multiobj_model.epsilonConstraint))
        #
        #

        
        if hasattr(model, 'obj'):
            model.del_component('obj')
        model.add_component('obj', pyo.Objective(rule=lambda m: multiobj_model.LOCdifferenceObjective(m)))
         
        
        concrete = model.create_instance(data)
        solver=pyo.SolverFactory('cplex')
        results = solver.solve(concrete)
        
        
        
        if results.solver.status == 'ok':
            f2 = pyo.value(multiobj_model.LOCdifferenceObjective(concrete))
            
            if hasattr(model, 'f2'):
                model.del_component('f2')
            model.add_component('f2', pyo.Var(initialize=f2))
            
            if hasattr(model, 'f2Constraint'):
                model.del_component('f2Constraint')
            model.add_component('f2Constraint', pyo.Constraint(rule=lambda m: multiobj_model.LOCdifferenceObjective(m)))
            
            if hasattr(model, 'obj'):
                model.del_component('obj')
            model.add_component('obj', pyo.Objective(rule=lambda m: multiobj_model.sequencesObjective(m)))
            
            solver=pyo.SolverFactory('cplex')
            results = solver.solve(concrete)
            
            print(results)
            
            
            print(f"f2: {f2}")
            f1 = pyo.value(multiobj_model.sequencesObjective(concrete))
            pareto_front = []
            
            print(f"fp1: {f1}")
            epsilon = f1 - 1
            
            
            # while results.solver.status == 'ok' and len_sequences > fp1[0]:
                # solution = solution +1
                # concrete.epsilon[1]=f1-1
                # results = solver.solve(concrete)
                # pf1=f1
                # f1 = pyo.value(multiobj_model.LOCdifferenceObjective(concrete))
                # f2 = pyo.value(multiobj_model.CCdifferenceObjective(concrete))
                #
                # print('Sequences selected:')
                # for s in concrete.S:
                #     print(f"x[{s}] = {concrete.x[s].value}")
