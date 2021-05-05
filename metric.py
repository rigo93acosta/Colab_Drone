import numpy as np
import matplotlib.pyplot as plt


class Metric:

    def __init__(self, ind_run, ind_episodes, actions):

        self.total_run = ind_run
        self.episodes = ind_episodes
        self.save_reward = []
        self.dron_RAN = []
        self.macro_RAN = []
        self.save_backhaul = []
        self.save_backhaul_1 = []
        self.save_status = []
        self.save_worse_thr = []
        self.save_mean_sinr = []
        self.actions_name = actions

    @staticmethod
    def calc_drone_ran(drones):
        """
        Calculate % occupation user RAN

        Args:
            drones:

        Returns:
            object:
        """
        total_active = 0
        actual = 0
        for drone in drones:
            if drone.status_tx:
                actual += drone.get_len_users
                total_active += 1

        total_ran = total_active * 50
        return actual * 100 / total_ran

    @staticmethod
    def calc_backhaul(drones, flag_backhaul):
        """
        Calculate average backhaul

        Args:
            drones:
            flag_backhaul:

        Returns:
            object:
        """
        total_backhaul = []
        for drone in drones:
            if drone.status_tx:
                total_backhaul.append(drone.actual_capacity)
        if flag_backhaul:   # Backhaul per drone
            return np.mean(total_backhaul) / 1e06
        else:   # Backhaul global
            return np.sum(total_backhaul) / 1e06

    @staticmethod
    def _calc_status(drones):

        val_temp = 0
        for dron in drones:
            if dron.status_tx:
                val_temp += 1

        return val_temp

    def update(self, reward_max, reward, drones, frequencies,
               worse_thr, mean_sinr):
        """
        Update metrics simulation
        Args:
            reward_max: Max Reward
            reward: Reward of the best position
            drones: List drones
            frequencies: List frequencies in use
            worse_thr: Worse throughput user
            power: Power total
            efficiency: Efficiency
        """
        self.save_reward.append((reward_max - reward) * 100 / reward_max)
        self.dron_RAN.append(self.calc_drone_ran(drones))
        self.save_backhaul.append(self.calc_backhaul(drones, True) / len(frequencies))
        self.save_backhaul_1.append(self.calc_backhaul(drones, False) / len(frequencies))
        self.save_status.append(self._calc_status(drones))
        self.save_worse_thr.append(worse_thr)
        self.save_mean_sinr.append(mean_sinr)

    def save_metric(self, run_i=0):
        """
        Save metrics simulation for independent run

        Args:
        run_i: Run in action
        """
        np.savez(f'Run_{run_i}', data=self.save_reward)
        np.savez(f'Run_load_{run_i}', data=self.dron_RAN)
        np.savez(f'Run_backhaul_drone{run_i}', data=self.save_backhaul)
        np.savez(f'Run_backhaul_global{run_i}', data=self.save_backhaul_1)
        np.savez(f'Run_status_{run_i}', data=self.save_status)
        np.savez(f'Run_worse_{run_i}', data=self.save_worse_thr)
        np.savez(f'Run_sinr_{run_i}', data=self.save_mean_sinr)

    def extra_metric(self, chapter, drones, n_episodes):

        temp = []
        for visual_index, drone in enumerate(drones):
            counts, bins = np.histogram(drone.shift, bins=len(self.actions_name))
            if counts[6] != 0:
                counts[6] -= n_episodes
            temp.append(counts)

        temp = np.ceil(np.mean(np.array(temp), axis=0))
        _, ax = plt.subplots()
        ax.bar(self.actions_name, temp)
        ax.set_xlabel(f'Actions')
        ax.set_ylabel(f'Repeat')
        ax.set_title(f'Average actions')
        filename = chapter + '/' + f'Ave_action'
        plt.savefig(f'{filename}.png', dpi=200)
        plt.close()
