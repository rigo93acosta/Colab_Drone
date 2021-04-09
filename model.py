import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')

from node import Node

with open('mapa.pickle', 'rb') as f:
    map = pickle.load(f)


class Model:

    def __init__(self, agents, n_users=300, frequency=None, n_run=0):

        if frequency is None:
            frequency = [1e09]

        self.terreno = map['terreno']  # Need to allocate users in the map
        self.agents = agents
        self.total_user = n_users
        self.output_chapter = f'run'
        self.user_list = []
        self.all_freq = frequency
        self.val_a, self.val_b = self.values_a_b
        self.simulation_run = n_run
        self.info = 'cluster'
        self.table_cov = {}
        self.tdma = None  # Time Distribution Multi Access
        self.rsrp = None
        self.rsrp_rc = None
        self.angle_aperture = 60

    def reset(self):
        """
        Reset or Init Environment

        Returns:
            States drones
        """

        def calc_thr(x):
            return 180e03 * np.log2(1 + np.power(10, x / 10))

        with open(f'users_d_{self.info}.pickle', 'rb') as file:
            user_distribution = pickle.load(file)

        with open(f'info_{self.info}.pickle', 'rb') as file:
            data_info = pickle.load(file)
        data_info = data_info[0]

        self.tdma = np.ones((len(self.agents), self.agents[0].max_capacity), dtype='int16') * -1

        for id_drone, drone in enumerate(self.agents):
            drone.position = data_info[self.simulation_run][id_drone]['position']
            drone.users = data_info[self.simulation_run][id_drone]['users_id'].copy()
            for index, value in enumerate(data_info[self.simulation_run][id_drone]['users_id']):
                self.tdma[id_drone][index] = value
            np.random.shuffle(self.tdma[id_drone])

        temp_position_user = user_distribution[self.simulation_run]
        for index_user in range(self.total_user):
            temp_user = Node(name='User_' + str(index_user + 1))
            temp_user.position = [temp_position_user[index_user][0], temp_position_user[index_user][1], 0]
            for index_drone, drone in enumerate(self.agents):
                if index_user in drone.users:
                    temp_user.index_dron = index_drone
                    temp_user.connection = True
                    break
            self.user_list.append(temp_user)

        sinr_downlink = self._interplay_downlink()
        for index_user, user in enumerate(self.user_list):
            user.th_downlink = 1.3 * calc_thr(sinr_downlink[index_user, user.index_dron])

        self.rsrp, self.rsrp_rc = self._table_rsrp()

    @property
    def calc_users_connected(self):
        """
        Calculate nodes downlink connected

        Returns:
            Nodes downlink connected
        """
        iteration_reward = 0
        for drone in self.agents:
            val = np.asarray(drone.users)
            if len(val) == 0:
                iteration_reward += 0
            else:
                iteration_reward += val.shape[0]

        return iteration_reward

    @property
    def values_a_b(self):
        """
        Calculate values for simulation relationship with LOS o not LOS

        Returns:
            Constants value A and B
        """
        c_a = [9.34e-01, 2.30e-01, -2.25e-03, 1.86e-05, 1.97e-02, 2.44e-03, 6.58e-06,
               0, -1.24e-04, -3.34e-06, 0, 0, 2.73e-07, 0, 0, 0]
        c_b = [1.17e-00, -7.56e-02, 1.98e-03, -1.78e-05, -5.79e-03, 1.81e-04, -1.65e-06,
               0, 1.73e-05, -2.02e-07, 0, 0, -2.00e-08, 0, 0, 0]

        c_a = np.asarray(c_a).reshape(4, 4)
        c_b = np.asarray(c_b).reshape(4, 4)

        z_a = []
        z_b = []

        for j in range(4):
            for i in range(4 - j):
                z_a.append(c_a[i, j] * np.power(map['alpha'] * map['beta'], i) * np.power(map['gamma'], j))

        for j in range(4):
            for i in range(4 - j):
                z_b.append(c_b[i, j] * np.power(map['alpha'] * map['beta'], i) * np.power(map['gamma'], j))

        return sum(z_a), sum(z_b)

    @staticmethod
    def _loss_path_average(dr, dh, data_a, data_b, f_tx):
        """
        Calculate loss free space + LOS
        Friis equation
        Args:
            dr: Distance users-drones in 2D
            dh: Heights drones
            data_a: Values for LOS
            data_b: Values for LOS

        Returns:
            Loss Path between users-drones
        """

        # Calculate for a frequency 1GHz
        np.seterr(divide='ignore', invalid='ignore')
        val_division = np.true_divide(dh, dr)
        term_a = 1 + data_a * np.exp(-data_b * (np.rad2deg(np.arctan(val_division)) - data_a))
        term_c = 1
        term_d = 20
        term_b = 10 * np.log10(np.square(dh) + np.square(dr)) + 20 * np.log10(f_tx) + 20 * np.log10(4 * np.pi / 3e08)

        return term_b + term_c

    def _table_rsrp(self):
        """
        Args:

        Returns:
            Loss without noise
        """
        user_array = np.asarray([user.position for user in self.user_list])
        dron_array = np.asarray([dron.position for dron in self.agents])
        dist_2d = np.zeros((user_array.shape[0], dron_array.shape[0]))
        drone_height = dron_array[:, 2]
        for idx_dron in range(len(self.agents)):
            dist_2d[:, idx_dron] = np.sqrt(np.sum(np.square(user_array[:, :2] - dron_array[idx_dron, :2]), axis=1))

        dist_3d = np.sqrt(np.square(dist_2d) + np.square(drone_height))
        # Equation 7 in the paper.
        power_tx = 10 * np.log10(self.user_list[0].power_tx)  # Power Tx IoT device
        path_loss = 20*np.log10(4*np.pi*dist_3d*1e09/3e08) + 1
        result = power_tx - path_loss
        result2 = np.copy(result)
        result2[self.table_cov['user_false'], self.table_cov['drone_false']] = -500
        return result, result2

    def _interplay_downlink(self):

        frequency_drone = []
        status_drone = []
        for id_d, dron in enumerate(self.agents):
            frequency_drone.append(dron.freq_tx)
            status_drone.append(dron.status_tx)

        user_array = np.asarray([user.position for user in self.user_list])
        dron_array = np.asarray([dron.position for dron in self.agents])
        dist_dron_r = np.zeros((user_array.shape[0], dron_array.shape[0]))

        for idx_dron in range(len(self.agents)):
            dist_dron_r[:, idx_dron] = np.sqrt(np.sum(np.square(user_array[:, :2] - dron_array[idx_dron, :2]), axis=1))

        eirp = -3  # -3 dBW
        dron_rc = dron_array[:, 2] * np.tan(np.deg2rad(self.angle_aperture / 2))  # Radio coverage all drones
        dist_drones = dist_dron_r[:, :len(self.agents)]
        drones_altura = dron_array[:len(self.agents), 2]
        result_loss = eirp - self._loss_path_average(dist_drones, drones_altura,
                                                     self.val_a, self.val_b, frequency_drone)
        table_ok_cob = dist_dron_r[:, :len(self.agents)] <= dron_rc[:len(self.agents)]
        user_false, dron_false = np.where(table_ok_cob == False)
        result_loss[user_false, dron_false] = -530  # Equal to -500dBm

        self.table_cov['user_false'] = user_false
        self.table_cov['drone_false'] = dron_false

        for index, status in enumerate(status_drone):
            if not status:
                result_loss[:, index] = -2000

        table_rsrp_linear = np.power(10, result_loss / 10)
        table_rsrp_inv_linear = np.zeros((self.total_user, len(self.agents)))
        for j in range(table_rsrp_inv_linear.shape[1]):
            f_tx_e = frequency_drone[j]
            index = np.where(np.asarray(frequency_drone) == f_tx_e)
            index = index[0]
            for k in index:
                if k != j:
                    table_rsrp_inv_linear[:, j] += table_rsrp_linear[:, k]

        # Additive White Gaussian Noise
        awgn_linear = np.power(10, -120 / 10) / 1000

        return 10 * np.log10(table_rsrp_linear) - 10 * np.log10(awgn_linear + table_rsrp_inv_linear)

    def calc_sinr(self):
        """
        Calculate power, awgn, interference
        Args:

        Returns:
            SINR array
        """

        def db2linear(value_db):
            return np.power(10, value_db / 10)

        power_slot = np.zeros((len(self.agents), self.agents[0].max_capacity), dtype='float64')
        for index_slot in range(self.tdma.shape[1]):
            user_slots = self.tdma[:, index_slot]
            for index, index_user in enumerate(user_slots):
                if index_user != -1:
                    val = self.rsrp[index_user, self.user_list[index_user].index_dron]
                    power_slot[index, index_slot] = val

        # Equal to sum at denominator for equation 8
        interference = np.zeros((len(self.agents), self.agents[0].max_capacity), dtype='float64')
        for index_slot in range(interference.shape[1]):  # Time Slots
            temp_slots = self.tdma[:, index_slot]
            for j, index_user in enumerate(temp_slots):  # Users TX
                if power_slot[j, index_slot] != 0:
                    index_drone = self.user_list[index_user].index_dron
                    vector_values = self.rsrp_rc[temp_slots, index_drone]
                    false_1 = temp_slots == -1
                    false_2 = temp_slots == index_user
                    linear_vector = db2linear(vector_values)
                    linear_vector[false_1 | false_2] = 0
                    interference[j, index_slot] = np.sum(linear_vector)

        # Additive White Gaussian Noise
        awgn_linear = np.power(10, -120 / 10) / 1000
        return power_slot, awgn_linear, interference

    def render(self, filename=f'Env.png'):
        """
        Render image environment
        Args:
            filename: Render filename
        """
        from PIL import Image

        cell_text = []
        for row in self.tdma:
            cell_text.append([f'{x}' for x in row])

        collabel = [f'TS{i}' for i in range(self.tdma.shape[1])]
        rowlabel = [f'D{i}' for i in range(self.tdma.shape[0])]

        plt.figure(linewidth=2, tight_layout={'pad': 1})  # figsize=(5,3))

        the_table = plt.table(cellText=cell_text,
                              rowLabels=rowlabel,
                              rowLoc='center',
                              colLabels=collabel,
                              cellLoc='center',
                              loc='center',
                              fontsize=40)

        the_table.scale(1, 2)
        ax = plt.gca()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.box(on=None)
        plt.draw()
        fig = plt.gcf()
        try:
            os.mkdir(self.output_chapter)
        except:
            pass
        dir_file = self.output_chapter + '/' + filename
        plt.savefig(f'{dir_file}',
                    # bbox='tight',
                    edgecolor=fig.get_edgecolor(),
                    facecolor=fig.get_facecolor(),
                    dpi=150
                    )
        plt.close()