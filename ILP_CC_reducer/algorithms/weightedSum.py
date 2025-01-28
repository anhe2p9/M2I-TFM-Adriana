import pyomo.environ as pyo # ayuda a definir y resolver problemas de optimización

from ILP_CC_reducer.CC_reducer.ILP_CCreducer import ILPCCReducer


class WeightedSumAlgorithm(ILPCCReducer):

    @staticmethod
    def get_name() -> str:
        return 'WeigthedSumAlgorithm'
    
    @staticmethod
    def get_description() -> str:
        return ("It obtains soported ILP solutions based on the given weights.")

    @staticmethod
    def execute(model: pyo.AbstractModel, data: pyo.ConcreteModel, w1: float, w2: float, w3: float) -> pyo.Objective:
        
        if hasattr(model, 'obj'):
            model.del_component('obj')  # Eliminar el componente existente
            model.add_component('obj', pyo.Objective(rule=lambda m: model.weightedSum(m, w1, w2, w3)))
        else:
            model.obj = pyo.Objective(rule=lambda m: model.weightedSum(m, w1, w2, w3))
    
        return model.obj



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
    
    




