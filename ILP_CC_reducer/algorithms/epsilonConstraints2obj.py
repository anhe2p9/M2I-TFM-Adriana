import pyomo.environ as pyo # ayuda a definir y resolver problemas de optimización
import pyomo.dataportal as dp # permite cargar datos para usar en esos modelos de optimización

from typing import Any
import sys

from ILP_CC_reducer.Algorithm.Algorithm import Algorithm
from ILP_CC_reducer.models import MultiobjectiveILPmodel



class EpsilonConstraintAlgorithm2obj(Algorithm):

    @staticmethod
    def get_name() -> str:
        return 'Epsilon Constraint Algorithm'
    
    @staticmethod
    def get_description() -> str:
        return ("It obtains supported and non-supported ILP solutions.")

    @staticmethod
    def execute(model: pyo.AbstractModel, data: dp.DataPortal, obj2: str) -> list[list[Any]]:
        
        multiobj_model = MultiobjectiveILPmodel()
        
        second_objective = multiobj_model.LOCdifferenceObjective if obj2 == 'LOC' else multiobj_model.CCdifferenceObjective

        # Solve {min f2}
        if hasattr(model, 'obj'):
            model.del_component('obj')
        model.add_component('obj', pyo.Objective(rule=lambda m: second_objective(m)))

        concrete = model.create_instance(data)
        solver=pyo.SolverFactory('cplex')
        results = solver.solve(concrete)
        
        # concrete.pprint()
        
        
        
        if results.solver.status == 'ok':
            
            """ Solve {min f1(x) subject to f2(x) <= f2(z)} """
            # f2(z) = f2
            f2 = pyo.value(concrete.obj)
            print(f"f2: {f2}")
            if obj2 == 'LOC':
                print(f"tmax: {concrete.cmax.value}, tmin: {concrete.cmin.value}")
            else:
                print(f"cmax: {concrete.tmax.value}, cmin: {concrete.tmin.value}")
            
            # new static variable to implement new constraint f2(x) <= f2(z)
            if hasattr(model, 'f2'):
                model.del_component('f2')
            model.add_component('f2', pyo.Param(within=pyo.NonNegativeReals, initialize=f2))
            
            # new constraint f2(x) <= f2(z)
            if hasattr(model, 'f2Constraint'):
                model.del_component('f2Constraint')
            model.add_component('f2Constraint', pyo.Constraint(rule=lambda m: multiobj_model.SecondObjdiffConstraint(m, obj2)))
            
            # new objective min f1(x)
            if hasattr(model, 'obj'):
                model.del_component('obj')
            model.add_component('obj', pyo.Objective(rule=lambda m: multiobj_model.sequencesObjective(m)))
            
            concrete = model.create_instance(data)
            solver=pyo.SolverFactory('cplex')
            results = solver.solve(concrete)
            
            # concrete.pprint()
                        
            
            """ z <- solve {min f1(x) subject to f2(x) <= f2(z)} """
            # z
            z = pyo.value(concrete.obj)
            f1z = pyo.value(multiobj_model.sequencesObjective(concrete))
            # FP <- {z} (add z to Pareto front)
            pareto_front = [z]
            
            print(f"f2 param: {concrete.f2.value}")
            print(f"f2 constraint: {second_objective(concrete)}")
            
            print(f"new objective (step 2): {pyo.value(concrete.obj)}")
            
            print(f"z: {z}, f1z: {f1z}")
            
            # epsilon <- f1(z) - 1
            if hasattr(model, 'epsilon'):
                model.del_component('epsilon')
            model.add_component('epsilon', pyo.Param(within=pyo.NonNegativeReals, initialize=f1z-1, mutable=True))

            
            # lower bound for f1(x)
            # f1 = pyo.value(multiobj_model.sequencesObjective(concrete))
            u1 = f1z - 1
            
             
            
            # f1(x)
            # f1_x = f1 # FUNCIÓN F1(X) QUE SUME TODOS LOS VALORES DE F1 PARA CUALQUIER X
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
            
            
            
            # l = epsilon - f1(x)
            if hasattr(model, 'l'):
                model.del_component('l')
            model.add_component('l', pyo.Var(initialize = model.epsilon - f1z))
            
            
            
            concrete = model.create_instance(data)
            solver=pyo.SolverFactory('cplex')
            results = solver.solve(concrete)
            
            print(f"epsilon: {concrete.epsilon.value}")
            
            
            print(f"f1z: {f1z}")
            print(f"u1: {u1}")
            
            print(f"l param: {concrete.l.value}")
            
            
            sequences_sum = sum(concrete.x[i].value for i in concrete.S if i != 0)
            
            if obj2 == 'LOC':
                xLOC = [concrete.loc[i] for i in concrete.S if concrete.x[i].value == 1]
                zLOC = [concrete.loc[j] for j,ii in concrete.N if concrete.z[j,ii].value == 1]
                
                max_selected = abs(max(xLOC) - max(zLOC))
                min_selected = min(concrete.loc[i] for i in concrete.S if concrete.x[i].value == 1)
            
            elif obj2 == 'CC':
        
                xCC = [concrete.nmcc[i] for i in concrete.S if concrete.x[i].value == 1]
                zCC = [concrete.ccr[j,ii] for j,ii in concrete.N if concrete.z[j,ii].value == 1]
                
                
                max_selected = abs(max(xCC) - max(zCC))
                min_selected = min(concrete.nmcc[i] for i in concrete.S if concrete.x[i].value == 1)
                
            else:
                sys.exit("Second objective parameter must be one between 'LOC' and 'CC'.")
                
            obj2_dif = abs(max_selected - min_selected)
            
            
            
            print('===============================================================================')
            if (results.solver.status == 'ok'):
                print('Objective SEQUENCES: ', sequences_sum +1)
                print(f'Second objective value ({obj2}): {obj2_dif}')
                print('Sequences selected:')
                for s in concrete.S:
                    print(f"x[{s}] = {concrete.x[s].value}")
            print('===============================================================================')
            
            
            
            while results.solver.status == 'ok' and f1z <= concrete.epsilon.value: # NO SÉ CÓMO PONER f1(x), ¿se podría poner f1(x) = 1? porque máximo va a ser 1
                
                # estimate a lambda value > 0
                lambd = 1/(f1z - u1)
                
                """ Solve {min f2(x) - lambda * l, subject to f1(x) + l = epsilon} """
                # min f2(x) - lambda * l
                if hasattr(model, 'obj'):
                    model.del_component('obj')
                model.add_component('obj', pyo.Objective(rule=lambda m: multiobj_model.epsilonObjective(m, lambd, obj2)))
                # subject to f1(x) + l = epsilon
                if hasattr(model, 'epsilonConstraint'):
                    model.del_component('epsilonConstraint')
                model.add_component('epsilonConstraint', pyo.Constraint(rule=lambda m: multiobj_model.epsilonConstraint(m, 'SEQ')))
                
                
                concrete = model.create_instance(data)
                solver=pyo.SolverFactory('cplex')
                results = solver.solve(concrete)
                
                # concrete.pprint()
                
                """ z <- solve {min f2(x) - lambda * l, subject to f1(x) + l = epsilon} """
                fp2 = pyo.value(multiobj_model.epsilonObjective(concrete, lambd, obj2))
                """ PF = PF U {z} """
                pareto_front.append(fp2)
                
                
                concrete.epsilon = f1z - 1
                
                #
                # f1_x_max = max(concrete.x[s].value for s in concrete.S)
                # print(f"f1_x_max: {f1_x_max}")
                
                print(f"f1z: {f1z}")
                print(f"fp2: {fp2}")
                print(f"epsilon: {model.epsilon}")
                print(f"u1: {u1}")
                print(f"lambda: {lambd}")
                    
                
                print(f"comprobacion: {f1z} <= {model.epsilon}")
                
                print('===============================================================================')
                if (results.solver.status == 'ok'):
                    print('Objective SEQUENCES: ', sequences_sum +1)
                    print(f'Second objective value ({obj2}): {obj2_dif}')
                    print('Sequences selected:')
                    for s in concrete.S:
                        print(f"x[{s}] = {concrete.x[s].value}")
                print('===============================================================================')
