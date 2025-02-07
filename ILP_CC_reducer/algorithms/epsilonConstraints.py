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


        # Solve {min f2}
        if hasattr(model, 'obj'):
            model.del_component('obj')
        model.add_component('obj', pyo.Objective(rule=lambda m: multiobj_model.LOCdifferenceObjective(m)))

        concrete = model.create_instance(data)
        solver=pyo.SolverFactory('cplex')
        results = solver.solve(concrete)
        
        # concrete.pprint()
        
        
        
        if results.solver.status == 'ok':
            
            """ Solve {min f1(x) subject to f2(x) <= f2(z)} """
            # f2(z) = f2
            f2 = pyo.value(multiobj_model.LOCdifferenceObjective(concrete))
            
            # new static variable to implement new constraint f2(x) <= f2(z)
            if hasattr(model, 'f2'):
                model.del_component('f2')
            model.add_component('f2', pyo.Var(initialize=f2))
            
            # new constraint f2(x) <= f2(z)
            if hasattr(model, 'f2Constraint'):
                model.del_component('f2Constraint')
            model.add_component('f2Constraint', pyo.Constraint(rule=lambda m: multiobj_model.LOCdifferenceConstraint(m)))
            
            # new objective min f1(x)
            if hasattr(model, 'obj'):
                model.del_component('obj')
            model.add_component('obj', pyo.Objective(rule=lambda m: multiobj_model.sequencesObjective(m)))
            
            concrete = model.create_instance(data)
            solver=pyo.SolverFactory('cplex')
            results = solver.solve(concrete)
            
            # concrete.pprint()
                        
            
            """ z <- solve {min f1(x) subject to f2(x) <= f2(z)} """
            # f1(z)
            fp1 = pyo.value(multiobj_model.sequencesObjective(concrete))
            # add f1(z) to pareto front
            pareto_front = [fp1]
            
            # epsilon <- f1(z) - 1
            epsilon = fp1 - 1
            
            # Delete f2Constraint to obtain the lower bound of f1(x) (first component of a utopian point)
            if hasattr(model, 'f2Constraint'):
                model.del_component('f2Constraint')
            solver=pyo.SolverFactory('cplex')
            results = solver.solve(concrete)  
            
            # f1(x)
            f1_x = lambda i: sum(concrete.x[i].value for i in concrete.S) # FUNCIÓN F1(X) QUE SUME TODOS LOS VALORES DE F1 PARA CUALQUIER X
            # [1,0,0,0,0,0,0,0,0,0,0] = 1
            # [0,1,0,0,0,0,0,0,0,0,0] = 1
            # [0,0,1,0,0,0,0,0,0,0,0] = 1
            # [0,0,0,1,0,0,0,0,0,0,0] = 1
            # [0,0,0,0,1,0,0,0,0,0,0] = 1
            # [0,0,0,0,0,1,0,0,0,0,0] = 1
            # [0,0,0,0,0,0,1,0,0,0,0] = 1
            # [0,0,0,0,0,0,0,1,0,0,0] = 1
            # [0,0,0,0,0,0,0,0,1,0,0] = 1
            # [0,0,0,0,0,0,0,0,0,1,0] = 1
            # [0,0,0,0,0,0,0,0,0,0,1] = 1
            # [1,1,0,0,0,0,0,0,0,0,0] = 2
            # [1,0,1,0,0,0,0,0,0,0,0] = 2
            # [1,0,0,1,0,0,0,0,0,0,0] = 2
            # [1,0,0,0,1,0,0,0,0,0,0] = 2
            # [1,0,0,0,0,1,0,0,0,0,0] = 2
            #          ...             ...
            print(f"f1_x: {f1_x}")
            # lower bound for f1(x)
            u1 = 1 # NO SÉ CÓMO HACER QUE FUNCIONE SIN QUE SEA NEGATIVO
            # l = 0.01 # NO SÉ QUÉ VALOR DARLE A ESTO
            
            # l = epsilon - f1(x)
            if hasattr(model, 'l'):
                model.del_component('l')
            model.add_component('l', pyo.Var())
            
            
            
            print(f"f2: {f2}")
            print(f"fp1: {fp1}")
            print(f"epsilon: {epsilon}")
            print(f"u1: {u1}")
            
            
            
            
            while results.solver.status == 'ok' and f1_x <= epsilon: # NO SÉ CÓMO PONER f1(x), ¿se podría poner f1(x) = 1? porque máximo va a ser 1
                
                # estimate a lambda value > 0
                lambd = 1/(fp1 - u1)
                
                """ Solve {min f2(x) - lambda * l, subject to f1(x) + l = epsilon} """
                # min f2(x) - lambda * l
                if hasattr(model, 'obj'):
                    model.del_component('obj')
                model.add_component('obj', pyo.Objective(rule=lambda m: multiobj_model.epsilonObjective(m, 'LOC', lambd)))
                # subject to f1(x) + l = epsilon
                if hasattr(model, 'epsilonConstraint'):
                    model.del_component('epsilonConstraint')
                model.add_component('epsilonConstraint', pyo.Constraint(rule=lambda m: multiobj_model.epsilonConstraint(m, 'SEQ', epsilon)))
                
                
                concrete = model.create_instance(data)
                solver=pyo.SolverFactory('cplex')
                results = solver.solve(concrete)
                
                concrete.pprint()
                
                
                f2_z = pyo.value(multiobj_model.epsilonObjective(concrete, 'LOC', lambd))
                pareto_front.append(f2_z)
                
                
                fp1 = pyo.value(multiobj_model.sequencesObjective(concrete))
                epsilon = fp1 - 1
                
                
                f1_x_max = max(concrete.x[s].value for s in concrete.S)
                print(f"f1_x_max: {f1_x_max}")
                
                
                print(f"comprobacion: {f1_x_max} <= {epsilon}")
