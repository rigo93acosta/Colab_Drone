class Node:
    """
    Class User
    """

    def __init__(self, name='Node'):
        self.pos = []
        self.name = name
        self.th_downlink = 0
        self.th_uplink = 0
        self.connection = False
        self.index_dron = None
        self._threshold_downlink = -3
        self._threshold_uplink = 5
        self.power_tx = 200e-03  # Power of TX 200mW

    def __repr__(self):
        return f'{self.name}(Position({self.pos[0]}, {self.pos[1]}, {self.pos[2]}), Connection:{self.connection})'

    @property
    def position(self):
        return self.pos

    @position.setter
    def position(self, list_pos):
        self.pos = list_pos

    @property
    def throughput_downlink(self):
        return self.th_downlink

    @throughput_downlink.setter
    def throughput_downlink(self, valor):
        self.th_downlink = valor

    @property
    def throughput_uplink(self):
        return self.th_uplink

    @throughput_uplink.setter
    def throughput_uplink(self, valor):
        self.th_uplink = valor
