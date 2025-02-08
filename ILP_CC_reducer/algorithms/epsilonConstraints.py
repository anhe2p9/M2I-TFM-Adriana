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


        # Solve {min f1}
        if hasattr(model, 'obj'):
            model.del_component('obj')
        model.add_component('obj', pyo.Objective(rule=lambda m: multiobj_model.sequencesObjective(m)))

        concrete = model.create_instance(data)
        solver=pyo.SolverFactory('cplex')
        results = solver.solve(concrete)
        
        # concrete.pprint()
        
        
        
        if results.solver.status == 'ok':
            
            """ Solve {min f2(x) subject to f1(x) <= f1(z)} """
            # f1(z) = f1
            f1 = pyo.value(multiobj_model.sequencesObjective(concrete))
            
            # new static variable to implement new constraint f1(x) <= f1(z)
            if hasattr(model, 'f1'):
                model.del_component('f1')
            model.add_component('f1', pyo.Var(initialize=f1))
            
            # new constraint f1(x) <= f1(z)
            if hasattr(model, 'f1Constraint'):
                model.del_component('f1Constraint')
            model.add_component('f1Constraint', pyo.Constraint(rule=lambda m: multiobj_model.sequencesConstraint(m)))
            
            # new objective min f2(x)
            if hasattr(model, 'obj'):
                model.del_component('obj')
            model.add_component('obj', pyo.Objective(rule=lambda m: multiobj_model.CCdifferenceObjective(m)))
            
            concrete = model.create_instance(data)
            solver=pyo.SolverFactory('cplex')
            results = solver.solve(concrete)
            
            # concrete.pprint()
                        
            
            """ z <- solve {min f2(x) subject to f1(x) <= f1(z)} """
            # f2(z)
            fp2 = pyo.value(multiobj_model.CCdifferenceObjective(concrete))
            # add f2(z) to pareto front
            pareto_front = [fp2]
            
            # epsilon <- f2(z) - 1
            epsilon = fp2
            if hasattr(model, 'epsilon'):
                model.del_component('epsilon')
            model.add_component('epsilon', pyo.Var(initialize=epsilon))
            
            # Delete f2Constraint to obtain the lower bound of f2(x) (first component of a utopian point)
            if hasattr(model, 'f1Constraint'):
                model.del_component('f1Constraint')
                
            if hasattr(model, 'obj'):
                model.del_component('obj')
            model.add_component('obj', pyo.Objective(rule=lambda m: multiobj_model.CCdifferenceObjective(m)))
            
            solver=pyo.SolverFactory('cplex')
            results = solver.solve(concrete)  
            
            # f2(x)
            f2_x = pyo.value(multiobj_model.CCdifferenceObjective(concrete)) # FUNCIÓN F1(X) QUE SUME TODOS LOS VALORES DE F1 PARA CUALQUIER X
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
            
            # lower bound for f1(x)
            u2 = 1 # NO SÉ CÓMO HACER QUE FUNCIONE SIN QUE SEA NEGATIVO
            # l = 0.01 # NO SÉ QUÉ VALOR DARLE A ESTO
            
            # l = epsilon - f1(x)
            if hasattr(model, 'l'):
                model.del_component('l')
            model.add_component('l', pyo.Var())
            
            
            
            print(f"f2: {f1}")
            print(f"fp1: {fp2}")
            print(f"epsilon: {epsilon}")
            print(f"u1: {u2}")
            print(f"f1_x: {f2_x}")
            
            
            
            
            print('===============================================================================')
            if (results.solver.status == 'ok'):
                print('Optimal solution found')
                print('Objective value: ', pyo.value(concrete.obj))
                print('Sequences selected:')
                for s in concrete.S:
                    print(f"x[{s}] = {concrete.x[s].value}")
            print('===============================================================================')
            
            
            
            while results.solver.status == 'ok' and f2_x <= epsilon: # NO SÉ CÓMO PONER f1(x), ¿se podría poner f1(x) = 1? porque máximo va a ser 1
                
                # estimate a lambda value > 0
                lambd = 1/(fp2 - u2)
                
                """ Solve {min f1(x) - lambda * l, subject to f2(x) + l = epsilon} """
                # min f1(x) - lambda * l
                if hasattr(model, 'obj'):
                    model.del_component('obj')
                model.add_component('obj', pyo.Objective(rule=lambda m: multiobj_model.epsilonSequencesObjective(m, lambd)))
                # subject to f1(x) + l = epsilon
                if hasattr(model, 'epsilonConstraint'):
                    model.del_component('epsilonConstraint')
                model.add_component('epsilonConstraint', pyo.Constraint(rule=lambda m: multiobj_model.epsilonConstraint(m, 'SEQ', epsilon)))
                
                
                concrete = model.create_instance(data)
                solver=pyo.SolverFactory('cplex')
                results = solver.solve(concrete)
                
                # concrete.pprint()
                
                
                f2_z = pyo.value(multiobj_model.epsilonObjective(concrete, 'CC', lambd))
                pareto_front.append(f2_z)
                
                
                fp1 = pyo.value(multiobj_model.sequencesObjective(concrete))
                epsilon = fp1 - 1
                
                
                f1_x_max = max(concrete.x[s].value for s in concrete.S)
                print(f"f1_x_max: {f1_x_max}")
                
                
                print(f"comprobacion: {f1_x_max} <= {epsilon}")
                
                print('===============================================================================')
                if (results.solver.status == 'ok'):
                    print('Optimal solution found')
                    print('Objective value: ', pyo.value(concrete.obj))
                    print('Sequences selected:')
                    for s in concrete.S:
                        print(f"x[{s}] = {concrete.x[s].value}")
                print('===============================================================================')
