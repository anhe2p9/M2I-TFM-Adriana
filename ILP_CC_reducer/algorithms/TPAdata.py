import pyomo.environ as pyo # ayuda a definir y resolver problemas de optimización

from ILP_CC_reducer.CC_reducer.ILP_CCreducer import ILPCCReducer


class TPAdataAlgorithm(ILPCCReducer):

    @staticmethod
    def get_name() -> str:
        return 'TPA data Algorithm'
    
    @staticmethod
    def get_description() -> str:
        return ("It obtains model matrices to apply TPA Algorithm.")

    @staticmethod
    def execute(model: pyo.AbstractModel, data: pyo.ConcreteModel, w1: float, w2: float, w3: float) -> None:
        pass
