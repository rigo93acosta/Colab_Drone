import time

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

    power, awgn, interference = model.calc_sinr()
    power_copy = np.copy(power)
    power_copy = np.where(power_copy == 0, -500.0, power_copy)

    # Step 0: Voy a beneficiar al primer drone en esta primera implementacion
    index_drone = 0
    nodes_power = {i: power_copy[index_drone, i] for i in range(25)}    # Get power
    nodes_slot = [model.tdma[index_drone, i] for i in range(25)]    # Get Slot Time
    sorted_power = {k: v for k, v in sorted(nodes_power.items(), key=lambda item: item[1])}  # Sort power
    power_key = list(sorted_power.keys())   # Keys dict to list
    for index_slot, value in enumerate(power_key):
        model.tdma[index_drone][index_slot] = nodes_slot[value]

    # Step 1: Asignar a cada slot ordenadamente el nodo de cada otro drone que menos interferencia causa
    for index_drone in range(len(agents)-1):
        nodes_power = {i: power_copy[index_drone+1, i] for i in range(25)}  # Get power
        nodes_slot = [model.tdma[index_drone+1, i] for i in range(25)]  # Get Slot Time
        sorted_power = {k: v for k, v in sorted(nodes_power.items(), key=lambda item: item[1])}  # Sort power
        power_key = list(sorted_power.keys())  # Keys dict to list
        for index_slot, value in enumerate(power_key):
            model.tdma[index_drone+1][index_slot] = nodes_slot[value]

    # Step 2:
    for _ in count():
        power, awgn, interference = model.calc_sinr()
        del power_copy
        power_copy = np.copy(power)
        power_copy = np.where(power_copy == 0, -500.0, power_copy)
        sinr_uplink = power - 10 * np.log10(awgn + interference)

        direction = np.where(sinr_uplink == np.amin(sinr_uplink))  # Drone and Slot Victim

        slot_power = power_copy[:, direction[1][0]]  # Power Slot
        node_drone_array = np.argsort(slot_power)
        node_drone_max = node_drone_array[-1]    # Max Interference node in drone
        drone_power = power_copy[node_drone_max, :]
        slot_min_array = np.argsort(drone_power)
        slot_min = slot_min_array[0]

        val_sinr_max_i = sinr_uplink[node_drone_max, direction[1][0]]  # SINR de slot maximo interferente
        val_sinr_min_i = sinr_uplink[node_drone_max, slot_min]  # SINR de slot minimo interferente

        copy_tdma = np.copy(model.tdma)
        # Changed slots
        worst_value = model.tdma[node_drone_max, direction[1][0]]
        model.tdma[node_drone_max, direction[1][0]] = model.tdma[node_drone_max, slot_min]
        model.tdma[node_drone_max, slot_min] = worst_value

        power, awgn, interference = model.calc_sinr()
        sinr_uplink = power - 10 * np.log10(awgn + interference)
        now_sinr_max_i = sinr_uplink[node_drone_max, direction[1][0]]
        now_sinr_min_i = sinr_uplink[node_drone_max, slot_min]

        if now_sinr_min_i < val_sinr_min_i or now_sinr_max_i < val_sinr_max_i:
            model.tdma = np.copy(copy_tdma)
            del copy_tdma
        else:
            print('Changed 1')
            continue

        power, awgn, interference = model.calc_sinr()
        del power_copy
        power_copy = np.copy(power)
        power_copy = np.where(power_copy == 0, -500.0, power_copy)
        sinr_uplink = power - 10 * np.log10(awgn + interference)

        direction = np.where(sinr_uplink == np.amin(sinr_uplink))  # Drone and Slot Victim

        slot_power = power_copy[:, direction[1][0]]  # Power Slot
        node_drone_array = np.argsort(slot_power)
        node_drone_max = node_drone_array[-1]  # Max Interference node in drone
        drone_power = power_copy[node_drone_max, :]
        slot_min_array = np.argsort(drone_power)
        slot_min = slot_min_array[1]

        val_sinr_max_i = sinr_uplink[node_drone_max, direction[1][0]]  # SINR de slot maximo interferente
        val_sinr_min_i = sinr_uplink[node_drone_max, slot_min]  # SINR de slot minimo interferente

        copy_tdma = np.copy(model.tdma)
        # Changed slots
        worst_value = model.tdma[node_drone_max, direction[1][0]]
        model.tdma[node_drone_max, direction[1][0]] = model.tdma[node_drone_max, slot_min]
        model.tdma[node_drone_max, slot_min] = worst_value

        power, awgn, interference = model.calc_sinr()
        sinr_uplink = power - 10 * np.log10(awgn + interference)
        now_sinr_max_i = sinr_uplink[node_drone_max, direction[1][0]]
        now_sinr_min_i = sinr_uplink[node_drone_max, slot_min]

        if now_sinr_min_i < val_sinr_min_i or now_sinr_max_i < val_sinr_max_i:
            model.tdma = np.copy(copy_tdma)
            del copy_tdma
        else:
            print('Changed 2')
            continue

        power, awgn, interference = model.calc_sinr()
        del power_copy
        power_copy = np.copy(power)
        power_copy = np.where(power_copy == 0, -500.0, power_copy)
        sinr_uplink = power - 10 * np.log10(awgn + interference)

        direction = np.where(sinr_uplink == np.amin(sinr_uplink))  # Drone and Slot Victim

        slot_power = power_copy[:, direction[1][0]]  # Power Slot
        node_drone_array = np.argsort(slot_power)
        node_drone_max = node_drone_array[-2]  # Max Interference node in drone
        drone_power = power_copy[node_drone_max, :]
        slot_min_array = np.argsort(drone_power)
        slot_min = slot_min_array[0]

        val_sinr_max_i = sinr_uplink[node_drone_max, direction[1][0]]  # SINR de slot maximo interferente
        val_sinr_min_i = sinr_uplink[node_drone_max, slot_min]  # SINR de slot minimo interferente

        copy_tdma = np.copy(model.tdma)
        # Changed slots
        worst_value = model.tdma[node_drone_max, direction[1][0]]
        model.tdma[node_drone_max, direction[1][0]] = model.tdma[node_drone_max, slot_min]
        model.tdma[node_drone_max, slot_min] = worst_value

        power, awgn, interference = model.calc_sinr()
        sinr_uplink = power - 10 * np.log10(awgn + interference)
        now_sinr_max_i = sinr_uplink[node_drone_max, direction[1][0]]
        now_sinr_min_i = sinr_uplink[node_drone_max, slot_min]

        if now_sinr_min_i < val_sinr_min_i or now_sinr_max_i < val_sinr_max_i:
            model.tdma = np.copy(copy_tdma)
            del copy_tdma
            flag_end = True
        else:
            print('Changed 3')
            continue


        # Calculate metric
        power, awgn, interference = model.calc_sinr()
        sinr_uplink = power - 10 * np.log10(awgn + interference)
        cond_1 = sinr_uplink >= 0
        part_1 = np.sum(cond_1 == True) - (250 - model.calc_users_connected)
        min_sinr = np.min(sinr_uplink)
        # index_false = sinr_uplink != 150
        # max_sinr = np.max(sinr_uplink[index_false])
        print(part_1, min_sinr)
        time.sleep(2)

        if flag_end:
            break
    # model.render(f'TDMA_TX.png')


if __name__ == '__main__':

    for run in range(20):
        print(f'Simulation {run}')
        simulation(run_i=run, n_iterations=200)
