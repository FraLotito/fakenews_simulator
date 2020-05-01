from .network import Network


class Simulator:
    def __init__(self, N_nodes, n_interests, random_const, random_phy_const):
        self.network = Network(N_nodes, n_interests, random_const, random_phy_const)
