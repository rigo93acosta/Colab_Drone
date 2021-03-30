from drone import Drone
from model import Model
import numpy as np
from operator import itemgetter
from itertools import count


def simulation(run_i=0, n_iterations=5, n_agents=10, frequency="1e09", n_users=200, distribution='grid',
               step_z=2, mail=False):

    frequency_list = [float(item) for item in frequency.split(',')]
    agents = [Drone(frequency_list) for _ in range(n_agents)]
    for index_drone, agent in enumerate(agents):
        agent.name = f'Drone_{index_drone}'
    model = Model(agents, frequency=frequency_list, n_users=n_users, n_run=run_i)
    model.info = distribution

    model.dir_sim = f'Run_{run_i}'
    model.reset()
    print(f'Node-Downlink: {model.calc_users_connected}')
    copy_tdma = np.copy(model.tdma)

    for iteration in count():
        power, awgn, interference = model.calc_sinr()
        for index_drone in range(len(agents)):
            nodes_power = {model.tdma[index_drone, i]: power[index_drone, i] for i in range(25)}
            nodes_noise = {i: interference[index_drone, i] for i in range(25)}
            sorted_power = {k: v for k, v in sorted(nodes_power.items(), key=lambda item: item[1])}
            sorted_noise = {k: v for k, v in sorted(nodes_noise.items(), key=lambda item: item[1])}
            power_key = list(sorted_power.keys())
            noise_key = list(sorted_noise.keys())
            new_slot = np.zeros((25,)).tolist()
            for key, value in zip(noise_key, power_key):
                new_slot[key] = value
            for index_slot, value in enumerate(new_slot):
                model.tdma[index_drone][index_slot] = value
            power, awgn, interference = model.calc_sinr()

        # model.tdma = np.copy(copy_tdma)
        power, awgn, interference = model.calc_sinr()
        sinr_uplink = power - 10 * np.log10(awgn + interference)
        cond_1 = sinr_uplink >= 0
        part_1 = np.sum(cond_1 == True) - (250 - model.calc_users_connected)
        min_sinr = np.min(sinr_uplink)
        index_false = sinr_uplink != 150
        max_sinr = np.max(sinr_uplink[index_false])
        print(f'Iter-{iteration} - {part_1} - '
              f'Min: {min_sinr:.2f} dB - '
              f'Max: {max_sinr:.2f} dB')

        model.render(f'Iteration_{iteration}.png')
        if iteration == n_iterations:
            break


if __name__ == '__main__':

    for run in range(20):
        simulation(run_i=run, n_iterations=200)
