# This sub-libary should contain all the necessary code to compute different metrics 
# given the raw output of spike trains from the simulation.
from ..simulator.data import Simulation
from ..utils import parameter



class Metric:
    """
    Abstract Class for all Metrics
    """
    def __init__(self, broad:bool):
        self.broad = broad

class IRateMetric(Metric):
    """
    Class Wrapper for firing rate of inhibitor neurons in the network
    """
    def __init__(self):
        super().__init__(broad=True)

    def compute(self, simulation:Simulation):
        pass


class ERateMetric(Metric):
    """
    Class Wrapper for firing rate of exitatory neurons in the network
    """
    def __init__(self):
        super().__init__(broad=True)

    def compute(self, simulation:Simulation):
        pass


