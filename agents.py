from enum import IntEnum
import numpy as np
import gym


class Drone:
    """
    Class Drone
    """

    class DefaultActions(IntEnum):
        up = 0  # Drone move up
        down = 1  # Drone move down
        right = 2  # Drone move right
        left = 3  # Drone move left
        forw = 4  # Drone move forward
        back = 5  # Drone move backward
        stop = 6  # Drone not move

    def __init__(self, frequency):
        self.pos = []
        self.name = 'Drone'
        self.capacity = 37.5e6 * (1 - 0.1 * np.random.rand())
        self.actual_capacity = 0
        self.max_capacity = 20
        self.actions = self.DefaultActions
        self.users = []
        self.shift = []

        self.distance = 0   # TODO: Distance traveled

        self.status_tx = True  # TODO: Transmission is enable(True) or disable(False)
        self.freq_tx = frequency[0]
        self.all_freq = frequency  # TODO: Frequency transmission available

        self.save_dict = {'save_users': [], 'save_freq': 0,
                          'save_position': [], 'save_status': True}

        self.action_space = gym.spaces.Discrete(len(self.actions))

        self.observation_space = gym.spaces.Dict(
            {'position': gym.spaces.MultiDiscrete([self.space_map[0].shape[0],
                                                   self.space_map[1].shape[0],
                                                   self.space_map[2].shape[0]]),
             'frequencies': gym.spaces.Discrete(len(self.all_freq)),
             'tx_status': gym.spaces.Discrete(2)
             }
        )

        self.q_table = np.zeros((self.observation_space['position'].nvec[0],
                                 self.observation_space['position'].nvec[1],
                                 self.observation_space['position'].nvec[2],
                                 self.observation_space['tx_status'].n,
                                 self.observation_space['frequencies'].n,
                                 self.action_space.n))

    def __repr__(self):
        return f'{self.name}(Position({self.pos[0]}, {self.pos[1]}, {self.pos[2]}), {len(self.users)} Users Connected' \
               f') and F_tx:{self.freq_tx} '

    @property
    def space_map(self):
        """
        All positions drone on map

        Returns:
            List values
        """
        all_pos_x = np.arange(0, 500, 50)
        all_pos_y = np.arange(0, 500, 50)
        all_pos_z = np.arange(100, 1001, 100)

        return [all_pos_x, all_pos_y, all_pos_z]

    @property
    def position(self):
        return self.pos

    @position.setter
    def position(self, pos_list):
        self.pos = pos_list

    @property
    def get_len_users(self):
        return len(self.users)

    def choice_action(self, obs_state, epsilon):
        """
        Drone select action
        Args:
            obs_state: drone actual state
            epsilon: epsilon-greedy

        Returns:
        Select action
        """
        if np.random.uniform(0, 1) < epsilon:  # TODO: Random
            a_selected = np.random.randint(self.action_space.n)
        else:  # TODO: Knowledge
            val = self.q_table[obs_state[0], obs_state[1], obs_state[2], obs_state[3], obs_state[4]]
            a_selected = np.random.choice([action_ for action_, value_ in enumerate(val) if value_ == np.max(val)])

        action_correct = self.validate_action(a_selected, obs_state)

        return action_correct, a_selected

    def validate_action(self, action, now_state):  # TODO: Only if drone are active
        """
        Validation action drone
        Args:
            action: action selected
            now_state: actual state drone

        Returns:
        Correct action
        """
        action_back = action
        max_space_z = self.observation_space['position'].nvec[2] - 1
        max_space_y = self.observation_space['position'].nvec[1] - 1
        max_space_x = self.observation_space['position'].nvec[0] - 1

        if action == self.actions.up:
            if now_state[2] == max_space_z:
                action_back = 6

        elif action == self.actions.down:
            if now_state[2] == 0:
                action_back = 6

        elif action == self.actions.right:
            if now_state[0] == max_space_x:
                action_back = 6

        elif action == self.actions.left:
            if now_state[0] == 0:
                action_back = 6

        elif action == self.actions.forw:
            if now_state[1] == max_space_y:
                action_back = 6

        elif action == self.actions.back:
            if now_state[1] == 0:
                action_back = 6

        elif action == self.actions.stop:
            action_back = 6

        return action_back

    def learn(self, old_state, new_state, values):
        """
        Drone learn interaction with environment
        Args:
            old_state: Previous state drone
            new_state: Actual state drone
            values: [0]-Learning Rate [1]-Discount factor [2]-Reward scenario [3]-Old action
        """
        max_future = np.max(self.q_table[new_state[0], new_state[1], new_state[2], new_state[3], new_state[4]])
        actual_value_state = self.q_table[old_state[0], old_state[1], old_state[2],
                                          old_state[3], old_state[4], values[3]]
        new_q = (1 - values[0]) * actual_value_state + values[0] * (values[2] + values[1] * max_future)
        self.q_table[old_state[0], old_state[1], old_state[2], old_state[3], old_state[4], values[3]] = new_q

    def action_step(self, value):
        """
        Step action drone
        Args:
            value: action choice for algorithm
        """
        if value == self.actions.stop:
            self.pos[0] += 0
            self.pos[1] += 0
            self.distance = 0

        elif value == self.actions.up:
            self.pos[2] += 100
            self.distance = 100

        elif value == self.actions.down:
            self.pos[2] -= 100
            self.distance = 100

        elif value == self.actions.right:
            self.pos[0] += 50
            self.distance = 50

        elif value == self.actions.left:
            self.pos[0] -= 50
            self.distance = 50

        elif value == self.actions.forw:
            self.pos[1] += 50
            self.distance = 50

        elif value == self.actions.back:
            self.pos[1] -= 50
            self.distance = 50

        self.shift.append(value)

    def save_best(self):
        """
        Save best scenario
        """
        self.save_dict['save_users'].clear()
        self.save_dict['save_users'] = self.users.copy()
        self.save_dict['save_position'].clear()
        self.save_dict['save_position'] = self.pos.copy()
        self.save_dict['save_status'] = self.status_tx
        self.save_dict['save_freq'] = self.freq_tx

    def load_best(self):
        """
        Load best scenario
        """
        self.users.clear()
        self.users = self.save_dict['save_users'].copy()
        self.pos.clear()
        self.pos = self.save_dict['save_position'].copy()
        self.status_tx = self.save_dict['save_status']
        self.freq_tx = self.save_dict['save_freq']


class User:
    """
    Class User
    """

    def __init__(self, name='User', init_req_th=0):
        self.pos = []
        self.name = name
        self.req_th = np.power(10, init_req_th / 10)
        self.th_allocate = 0
        self.value_sinr = 0
        self.connection = False
        self.index_dron = None
        self.save_status = False  # TODO: Save status connection
        self.save_index = None  # TODO: Save id dron TRUE connection

    def __repr__(self):
        return f'{self.name}(Position({self.pos[0]}, {self.pos[1]}, {self.pos[2]}), Connection:{self.connection})'

    @property
    def position(self):
        return self.pos

    @position.setter
    def position(self, list_pos):
        self.pos = list_pos

    def action_step(self, mov):
        """
        User action step
        Args:
            mov: array with action move user
        """
        if (self.pos[0] + mov[0] >= 0) and (self.pos[0] + mov[0]) < 500:
            self.pos[0] += mov[0]
        else:
            self.pos[0] += 0

        if (self.pos[1] + mov[1] >= 0) and (self.pos[1] + mov[1]) < 500:
            self.pos[1] += mov[1]
        else:
            self.pos[1] += 0

    @property
    def throughput(self):
        return self.th_allocate

    @throughput.setter
    def throughput(self, valor):
        self.th_allocate = valor

    def save_best(self):
        """
        Save best scenario
        """
        self.save_index = self.index_dron
        self.save_status = self.connection

    def load_best(self):
        """
        Load best scenario
        """
        self.index_dron = self.save_index
        self.connection = self.save_status
