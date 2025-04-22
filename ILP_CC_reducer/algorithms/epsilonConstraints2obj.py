import pyomo.environ as pyo # ayuda a definir y resolver problemas de optimización
import pyomo.dataportal as dp # permite cargar datos para usar en esos modelos de optimización

import sys

import csv
import os

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
    def execute(data: dp.DataPortal, tau: int, obj2: str) -> None:
        
        multiobj_model = MultiobjectiveILPmodel()
        
        multiobj_model.model.tau = pyo.Param(within=pyo.NonNegativeReals, initialize=tau, mutable=False) # Threshold
        
        second_objective = multiobj_model.LOCdifferenceObjective if obj2 == 'LOC' else multiobj_model.CCdifferenceObjective

        # Solve {min f2}
        multiobj_model.model.obj = pyo.Objective(rule=lambda m: second_objective(m))

        concrete = multiobj_model.model.create_instance(data)
        
        # Guardar el modelo en un archivo .lp antes de resolverlo
        concrete.write(f'output/Econstraint_paso1.lp', io_options={'symbolic_solver_labels': True})
        
        solver=pyo.SolverFactory('cplex')
        result = solver.solve(concrete)
        
        # concrete.pprint()
        
        
        
        if result.solver.status == 'ok':
            
            """ Solve {min f1(x) subject to f2(x) <= f2(z)} """
            # f2(z) = f2
            f2z = pyo.value(concrete.obj)
            print(f"f2z: {f2z}")
            if obj2 == 'LOC':
                print(f"tmax: {concrete.tmax.value}, tmin: {concrete.tmin.value}")
            else:
                print(f"cmax: {concrete.cmax.value}, cmin: {concrete.cmin.value}")
            
            # new parameter to implement new constraint f2(x) <= f2(z)
            multiobj_model.model.f2z = pyo.Param(within=pyo.NonNegativeReals, initialize=f2z)
            
            # new constraint f2(x) <= f2(z)
            multiobj_model.model.f2Constraint = pyo.Constraint(rule=lambda m: multiobj_model.SecondObjdiffConstraint(m, obj2))
            
            # new objective: min f1(x)
            if hasattr(multiobj_model.model, 'obj'):
                multiobj_model.model.del_component('obj')
            multiobj_model.model.add_component('obj', pyo.Objective(rule=lambda m: multiobj_model.sequencesObjective(m))) # FRANCIS CREE QUE SE PUEDE SOBREESCRIBIR LA VARIABLE Y YA ESTÁ
            
            concrete = multiobj_model.model.create_instance(data)
            
            # Guardar el modelo en un archivo .lp antes de resolverlo
            concrete.write(f'output/Econstraint_paso2.lp', io_options={'symbolic_solver_labels': True})
            
            solver=pyo.SolverFactory('cplex')
            result = solver.solve(concrete)
            
            # concrete.pprint()
            
            
            """ z <- solve {min f1(x) subject to f2(x) <= f2(z)} """
            # z
            z = pyo.value(concrete.obj)
            f1z = pyo.value(multiobj_model.sequencesObjective(concrete))
            # FP <- {z} (add z to Pareto front)
            pareto_front = [z]
            
            print(f"f2 param: {concrete.f2z.value}")
            print(f"f2 constraint: {second_objective(concrete)}")
            
            print(f"new objective (step 2): {pyo.value(concrete.obj)}")
            
            print(f"z: {z}, f1z: {f1z}")
            
            # epsilon <- f1(z) - 1
            multiobj_model.model.epsilon = pyo.Param(within=pyo.NonNegativeReals, initialize=f1z-1, mutable=True)

            
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
            multiobj_model.model.l = pyo.Var(within=pyo.NonNegativeReals)
            # pyo.Param(initialize = multiobj_model.model.epsilon - multiobj_model.sequencesObjective(concrete), mutable=True)
            
            
            # concrete = multiobj_model.model.create_instance(data)
            #
            #
            # solver=pyo.SolverFactory('cplex')
            # result = solver.solve(concrete)
            #
            # print(f"epsilon: {concrete.epsilon.value}")
            #
            #
            # print(f"f1z: {f1z}")
            # print(f"u1: {u1}")
            #
            # print(f"l param: {concrete.l.value}")
            #
            #
            #
            # sequences_sum, obj2_dif = calculate_results(concrete, obj2)
            #
            #
            #
            # print('===============================================================================')
            # if (result.solver.status == 'ok'):
            #     print('Objective SEQUENCES: ', sequences_sum)
            #     print(f'Second objective value ({obj2}): {obj2_dif}')
            #     print('Sequences selected:')
            #     for s in concrete.S:
            #         print(f"x[{s}] = {concrete.x[s].value}")
            # print('===============================================================================')
            #
            #
            csv_data = [["Num.sequences","CCdif"]]
            # newrow = [sequences_sum +1, obj2_dif]
            # csv_data.append(newrow)
            #
            #
            #
            #
            # """epsilon = f1(z) - 1"""
            # f1z = pyo.value(multiobj_model.sequencesObjective(concrete))
            # multiobj_model.model.epsilon = f1z - 1
            #

            
            f2_found = True # while loop control
            multiobj_model.model.del_component('f2Constraint') # delete f2(x) <= f2(z) constraint
            
            while f2_found: # NO SÉ CÓMO PONER f1(x)
                # estimate a lambda value > 0
                lambd = 1/(f1z - u1)
                
                """ Solve {min f2(x) - lambda * l, subject to f1(x) + l = epsilon} """
                # min f2(x) - lambda * l
                if hasattr(multiobj_model.model, 'obj'):
                    multiobj_model.model.del_component('obj')
                multiobj_model.model.add_component('obj', pyo.Objective(rule=lambda m: multiobj_model.epsilonObjective(m, lambd, obj2)))

                # subject to f1(x) + l = epsilon
                if hasattr(multiobj_model.model, 'epsilonConstraint'):
                    multiobj_model.model.del_component('epsilonConstraint')
                multiobj_model.model.add_component('epsilonConstraint', pyo.Constraint(rule=lambda m: multiobj_model.epsilonConstraint(m, 'SEQ')))
                
                
                concrete = multiobj_model.model.create_instance(data)
                
                # Guardar el modelo en un archivo .lp antes de resolverlo
                concrete.write(f'output/Econstraint_paso3.lp', io_options={'symbolic_solver_labels': True})
                
                solver=pyo.SolverFactory('cplex')
                result = solver.solve(concrete)
                
                print(f"variable l value: {concrete.l.value}")
                
                # concrete.pprint()
                
                if (result.solver.status == 'ok') and (result.solver.termination_condition == 'optimal'):
                    
                    # z <- solve {min f2(x) - lambda * l, subject to f1(x) + l = epsilon}
                    fp2 = pyo.value(multiobj_model.epsilonObjective(concrete, lambd, obj2))
                    
                    
                    """ PF = PF U {z} """
                    z = pyo.value(concrete.obj)
                    # FP <- PF U {z} (add z to Pareto front)
                    pareto_front.append(z)
                    
                    """ epsilon = f1(z) - 1 """
                    f1z = pyo.value(multiobj_model.sequencesObjective(concrete))
                    
                    
                    
                    if hasattr(multiobj_model.model, 'epsilon'):
                        multiobj_model.model.del_component('epsilon')
                    multiobj_model.model.add_component('epsilon', pyo.Param(within=pyo.NonNegativeReals, initialize=f1z-1, mutable=True))
                    
                    
                    
                    print(f"f1z: {f1z}")
                    
                    # multiobj_model.model.epsilon.set_value(f1z - 1)
                    
                    
                    
                    u1 = f1z - 1
                    
                    
                    sequences_sum, obj2_dif = calculate_results(concrete, obj2)
                    
                    
                    print(f"fp2: {fp2}")
                    print(f"epsilon: {concrete.epsilon.value}")
                    print(f"u1: {u1}")
                    print(f"lambda: {lambd}")
                        
                    
                    print(f"comprobacion: {f1z} <= {concrete.epsilon.value}")
                    
                    # print('===============================================================================')
                    if (result.solver.status == 'ok'):
                        print('Objective SEQUENCES: ', sequences_sum)
                        print(f'Second objective value ({obj2}): {obj2_dif}')
                        print('Sequences selected:')
                        for s in concrete.S:
                            print(f"x[{s}] = {concrete.x[s].value}")
                    print('===============================================================================')
                    
                    newrow = [sequences_sum, obj2_dif]
                    csv_data.append(newrow)
                    
                else:
                    f2_found = False
                
            
            
            # Write data in a CSV file.
            filename = "output/epsilon_constr_2obj_output.csv"
            
            if os.path.exists(filename):
                os.remove(filename)
                    
            with open(filename, mode="w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerows(csv_data)
                print("CSV file correctly created.")







                
def calculate_results(concrete_model: pyo.ConcreteModel, second_obj_str: str):
    sequences_sum = sum(concrete_model.x[i].value for i in concrete_model.S)
            
    if second_obj_str == 'LOC':
        xLOC = [concrete_model.loc[i] for i in concrete_model.S if concrete_model.x[i].value == 1]
        zLOC = [concrete_model.loc[j] for j,ii in concrete_model.N if concrete_model.z[j,ii].value == 1]
        
        max_selected = abs(max(xLOC) - max(zLOC))
        min_selected = min(concrete_model.loc[i] for i in concrete_model.S if concrete_model.x[i].value == 1)
    
    elif second_obj_str == 'CC':

        xCC = [concrete_model.nmcc[i] for i in concrete_model.S if concrete_model.x[i].value == 1]
        zCC = [concrete_model.ccr[j,ii] for j,ii in concrete_model.N if concrete_model.z[j,ii].value == 1]
        
        
        max_selected = abs(max(xCC) - max(zCC))
        min_selected = min(concrete_model.nmcc[i] for i in concrete_model.S if concrete_model.x[i].value == 1)
        
    else:
        sys.exit("Second objective parameter must be one between 'LOC' and 'CC'.")
        
    obj2_dif = abs(max_selected - min_selected)
    
    return sequences_sum, obj2_dif