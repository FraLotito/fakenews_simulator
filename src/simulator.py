from network import Network, Node
import math


class Simulator:
    def __init__(self, N_common, N_influencers, N_interests, random_const, random_phy_const):
        self.network = Network(N_common, N_influencers, N_interests, random_const, random_phy_const)

def irradiate(a : Node, b : Node):
    En = 1
    #permeability
    a.education_rate = 0.8
    #proximity
    d = 0.9
    mb = 1
    alpha = 1
    beta = 2
    return En * a.education_rate * ((alpha * d + beta * mb) / (alpha + beta))

if __name__ == "__main__":
    a = Node(1, 1, 5)
    b = Node(1, 1, 5)
    print(irradiate(a, b))