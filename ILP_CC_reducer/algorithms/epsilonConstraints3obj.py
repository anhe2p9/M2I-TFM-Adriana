import pyomo.environ as pyo # ayuda a definir y resolver problemas de optimización
import pyomo.dataportal as dp # permite cargar datos para usar en esos modelos de optimización

from typing import Any

from ILP_CC_reducer.Algorithm.Algorithm import Algorithm
from ILP_CC_reducer.models import MultiobjectiveILPmodel
# from pyomo.solvers.tests.solvers import initialize



class EpsilonConstraintAlgorithm3obj(Algorithm):

    @staticmethod
    def get_name() -> str:
        return 'Epsilon Constraint Algorithm with 3 obj'
    
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
            model.add_component('obj', pyo.Objective(rule=lambda m: multiobj_model.LOCdifferenceObjective(m)))
            
            concrete = model.create_instance(data)
            solver=pyo.SolverFactory('cplex')
            results = solver.solve(concrete)
            
            # concrete.pprint()
            
            
            """ z <- solve {min f2(x) subject to f1(x) <= f1(z)} """
            # f2(z)
            fp2 = pyo.value(multiobj_model.LOCdifferenceObjective(concrete))
            """ add f2(z) to pareto front """
            pareto_front = [fp2]
            
            
            
                        
            """ Solve {min f3(x) subject to f1(x) <= f1(z)} """
            # new objective min f3(x)
            if hasattr(model, 'obj'):
                model.del_component('obj')
            model.add_component('obj', pyo.Objective(rule=lambda m: multiobj_model.CCdifferenceObjective(m)))
            
            concrete = model.create_instance(data)
            solver=pyo.SolverFactory('cplex')
            results = solver.solve(concrete)
            
            # concrete.pprint()
            
            
            """ z <- solve {min f3(x) subject to f1(x) <= f1(z)} """
            # f3(z)
            fp3 = pyo.value(multiobj_model.CCdifferenceObjective(concrete))
            """ add f3(z) to pareto front """
            pareto_front.append(fp3)
            
            # Delete f2Constraint to obtain the lower bound of f1(x) (first component of a utopian point)
            if hasattr(model, 'f2Constraint'):
                model.del_component('f2Constraint')
                
            # Delete f3Constraint to obtain the lower bound of f1(x) (first component of a utopian point)
            if hasattr(model, 'f3Constraint'):
                model.del_component('f3Constraint')
                
            if hasattr(model, 'obj'):
                model.del_component('obj')
            model.add_component('obj', pyo.Objective(rule=lambda m: multiobj_model.sequencesObjective(m)))
            
            solver=pyo.SolverFactory('cplex')
            results = solver.solve(concrete)  
            
            # f2(x)
            f2_x = pyo.value(multiobj_model.LOCdifferenceObjective(concrete)) # FUNCIÓN F1(X) QUE SUME TODOS LOS VALORES DE F1 PARA CUALQUIER X
            
            # f3(x)
            f3_x = pyo.value(multiobj_model.CCdifferenceObjective(concrete)) # FUNCIÓN F1(X) QUE SUME TODOS LOS VALORES DE F1 PARA CUALQUIER X
            
            
            # lower bound for f1(x)
            u2 = 0 # NO SÉ CÓMO HACER QUE FUNCIONE SIN QUE SEA NEGATIVO
            u3 = 0
            # l = 0.01 # NO SÉ QUÉ VALOR DARLE A ESTO
            
            # l2 = epsilon - f2(x)
            if hasattr(model, 'l2'):
                model.del_component('l2')
            model.add_component('l2', pyo.Var())
            
            if hasattr(model, 'eps2'):
                model.del_component('eps2')
            model.add_component('eps2', pyo.Var(initialize = fp2 - 1))
            
            # l3 = epsilon - f3(x)
            if hasattr(model, 'l3'):
                model.del_component('l3')
            model.add_component('l3', pyo.Var())
            if hasattr(model, 'eps3'):
                model.del_component('eps3')
            model.add_component('eps3', pyo.Var(initialize = fp3 - 1))
            
            
            # eps2 = fp2 - 1
            # eps3 = fp3 - 1
            
            
            
            print(f"f1: {f1}")
            
            print(f"f2_x: {f2_x}, f3_x: {f3_x}")
            print(f"fp2: {fp2}, fp3: {fp3}")
            # print(f"epsilon: eps2: {eps2}, eps3: {eps3}")
            
            
            print('===============================================================================')
            if (results.solver.status == 'ok'):
                print('Optimal solution found')
                print('Objective value: ', pyo.value(concrete.obj))
                print('Sequences selected:')
                for s in concrete.S:
                    print(f"x[{s}] = {concrete.x[s].value}")
            print('===============================================================================')
            
            
            
            while results.solver.status == 'ok' and f2_x <= multiobj_model.eps2 and f3_x <= multiobj_model.eps3:
                
                print(f"f2_x: {f2_x}, f3_x: {f3_x}")
                print(f"fp2: {fp2}, fp3: {fp3}")
                print(f"epsilon: eps2: {multiobj_model.eps2}, eps3: {multiobj_model.eps3}")
                
                # estimate a lambda value > 0
                lambd = 1/(f1 - u1)
                
                """ Solve {min f1(x) - lambda * (l2 + l3), subject to f2(x) + l2 = epsilon AND f3(x) + l3 = epsilon} """
                # min f1(x) - lambda * (l2 + l3)
                if hasattr(model, 'obj'):
                    model.del_component('obj')
                model.add_component('obj', pyo.Objective(rule=lambda m: multiobj_model.epsilonObjective(m, lambd)))
                # subject to f2(x) + l2 = epsilon
                if hasattr(model, 'LOC_Constraint'):
                    model.del_component('LOC_Constraint')
                model.add_component('LOC_Constraint', pyo.Constraint(rule=lambda m: multiobj_model.epsilonConstraint(m, 'LOC', m.eps2)))
                # subject to f3(x) + l3 = epsilon
                if hasattr(model, 'CC_Constraint'):
                    model.del_component('CC_Constraint')
                model.add_component('CC_Constraint', pyo.Constraint(rule=lambda m: multiobj_model.epsilonConstraint(m, 'CC', m.ep3)))
                
                
                
                concrete = model.create_instance(data)
                solver=pyo.SolverFactory('cplex')
                results = solver.solve(concrete)
                
                # concrete.pprint()
                
                
                f1_z = pyo.value(multiobj_model.epsilonObjective(concrete, lambd))
                pareto_front.append(f1_z)
                
                
                fp2 = pyo.value(multiobj_model.LOCdifferenceObjective(concrete))
                eps2 = fp2 -1
                
                fp3 = pyo.value(multiobj_model.CCdifferenceObjective(concrete))
                eps3 = fp3 -1
                
                print(f"f2_x: {f2_x}, f3_x: {f3_x}")
                print(f"fp2: {fp2}, fp3: {fp3}")
                print(f"epsilon: eps2: {eps2}, eps3: {eps3}")
                
                
                print('===============================================================================')
                if (results.solver.status == 'ok'):
                    print('Optimal solution found')
                    print('Objective value: ', pyo.value(concrete.obj))
                    print('Sequences selected:')
                    for s in concrete.S:
                        print(f"x[{s}] = {concrete.x[s].value}")
                print('===============================================================================')
