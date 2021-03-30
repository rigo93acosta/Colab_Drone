import random


class Drone:
    """
    Class Drone
    """

    def __init__(self, frequency):
        self.pos = []
        self.name = 'Drone'
        self.max_capacity = 25
        self.users = []
        self.shift = []
        self.altitude = []

        self.status_tx = True  # Transmission is enable(True) or disable(False)
        self.freq_tx = random.choice(frequency)
        self.all_freq = frequency  # Frequency transmission available

    def __repr__(self):
        return f'{self.name}(Position({self.pos[0]}, {self.pos[1]}, {self.pos[2]}), {len(self.users)} Users Connected' \
               f') and F_tx:{self.freq_tx} '

    @property
    def position(self):
        return self.pos

    @position.setter
    def position(self, pos_list):
        self.pos = pos_list

    @property
    def get_len_users(self):
        return len(self.users)
