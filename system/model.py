from typing import Dict

class PowerSystem:
    def __init__(self, nodes: Dict[int, dict]) -> None:
        self.nodes = nodes
        self.dim = len(nodes.keys())
        self.Y = None
        self._compute_Y()

    def __getitem__(self, index):
        if not isinstance(index, int):
            raise ValueError("Only integer index")
        for items in list(self.nodes.items()):
            if items[0] == index:
                return items
        raise ValueError("Not such item")
    
    def _compute_Y(self):
        return self
    
    def solve(self):
        self.solve_powerflow()
        self.solve_analisys()
        return self

    def solve_powerflow(self):
        return self
    
    def solve_analisys(self):
        return self