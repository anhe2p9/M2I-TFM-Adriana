from abc import ABC, abstractstaticmethod


class ILPCCReducer(ABC):
    
    @abstractstaticmethod
    def define_model_without_obj(self):
        """Defines model except objective."""
        pass
    
    @abstractstaticmethod
    def process_data(self, S_filename: str, N_filename: str, C_filename: str):
        """Processes data from DataPortal."""
        pass
    
    @abstractstaticmethod
    def sequencesObjective(self, m):
        """Models number of sequences objective."""
        pass
    
    @abstractstaticmethod
    def LOCdifferenceObjective(self, m):
        """Models LOC difference objective."""
        pass
        
    @abstractstaticmethod
    def CCdifferenceObjective(self, m):
        """Models CC difference objective."""
        pass
    
    
    
    