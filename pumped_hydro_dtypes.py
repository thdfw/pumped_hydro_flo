from typing import Optional
from typing import Literal, Optional
from pydantic import BaseModel


class FloParamsPumpedHydro(BaseModel):
    Year: int = 2024
    FloStartHr: int = 1
    FloHours: int = 8784
    StoreMwh: int = 2800
    GenerationMw: int = 500
    PumpingMw: int = 500
    RegMidpointMw: int = 350
    PumpLossPercent: int = 10
    GenLossPercent: int = 10
    LocationId: int = 4001
    Lmp: list = []
    RegMcp: list = []
    EnergyDiscretization: int = 100
    InitialEnergySlice: float = 50
    TypeName: Literal["flo.params.pumped_hydro"] = "flo.params.pumped_hydro"
    Version: Literal["000"] = "000"

    def to_dict(self):
        return vars(self)


class DParams():
    def __init__(self, flo_params: FloParamsPumpedHydro) -> None:
        if len(flo_params.Lmp) < flo_params.FloHours:
            raise Exception(f"LMP sequence must be at least {flo_params.FloHours} hours long")
        if len(flo_params.RegMcp) < flo_params.FloHours:
            raise Exception(f"RegMCP sequence must be at least {flo_params.FloHours} hours long")
        self.flo_params = flo_params
        self.year = flo_params.Year
        self.start_hour = flo_params.FloStartHr - 1
        self.horizon = flo_params.FloHours
        self.energy_discretization = flo_params.EnergyDiscretization
        self.storage_capacity_mwh = flo_params.StoreMwh
        self.initial_energy_slice = flo_params.InitialEnergySlice
        self.reg_midpoint_mw = flo_params.RegMidpointMw
        self.generation_mw = flo_params.GenerationMw
        self.pumping_mw = flo_params.PumpingMw
        self.pump_loss_percent = flo_params.PumpLossPercent
        self.gen_loss_percent = flo_params.GenLossPercent
        self.lmp = flo_params.Lmp[self.start_hour:self.start_hour+self.horizon]
        self.reg_mcp = flo_params.RegMcp[self.start_hour:self.start_hour+self.horizon]


class DNode():
    def __init__(self, time_slice: int, energy_slice: int, params: DParams):
        self.params = params
        self.time_slice = time_slice
        self.energy_slice = energy_slice
        self.energy = int(
            self.energy_slice/self.params.energy_discretization * self.params.storage_capacity_mwh
        )
        # Dijkstra's algorithm
        self.pathcost = 0 if time_slice==self.params.horizon else 1e9
        self.next_node: DNode = None     
        self.index: Optional[int] = None   

    def __repr__(self):
        return f"Node[{self.time_slice}, {self.energy_slice}/{self.params.energy_discretization}]"


class DEdge():
    def __init__(self, tail:DNode, head:DNode, cost:float, mwh_to_grid:float):
        self.tail: DNode = tail
        self.head: DNode = head
        self.cost = cost
        self.mwh_to_grid = mwh_to_grid

    def __repr__(self):
        return f"Edge[{self.tail} --cost:{round(self.cost,3)} USD, mwh_to_grid:{round(self.mwh_to_grid,2)} MWh--> {self.head}]"
    