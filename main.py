import argparse
import os
import pickle
import time
from shutil import copy
from operator import itemgetter
from itertools import count

from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from os.path import basename
import smtplib

import imageio
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import numpy as np

from agents import Drone
from metric import Metric
from multidrone import MultiDroneEnv


def send_mail(name_simulation='Test'):
    msg = MIMEMultipart()
    msg['From'] = "riacosta@uclv.cu"
    # msg['To'] = ', '.join('riacosta@uclv.cu')
    msg['To'] = 'riacosta@uclv.cu'
    msg['Subject'] = f'{name_simulation} Simulation End'
    msg.attach(MIMEText("End Simulation"))
    files_list = ['fig_6.pickle', 'fig_7.pickle', 'fig_11.pickle', 'fig_13.pickle',
                  'fig_12.pickle', 'fig_status.pickle']
    for f in files_list:
        with open(f, "rb") as fil:
            ext = f.split('.')[-1:]
            attached_file = MIMEApplication(fil.read(), _subtype=ext)
            attached_file.add_header(
                'content-disposition', 'attachment', filename=basename(f))
        msg.attach(attached_file)

    server = smtplib.SMTP('mta.uclv.edu.cu', 587)
    server.starttls()
    server.login("riacosta@uclv.cu", "rigo1993.")
    server.sendmail("riacosta@uclv.cu", "riacosta@uclv.cu", msg.as_string())
    server.quit()


def mail_end_episode(number_episode=0):
    msg = MIMEMultipart()
    msg['From'] = "riacosta@uclv.cu"
    # msg['To'] = ', '.join('riacosta@uclv.cu')
    msg['To'] = 'riacosta@uclv.cu'
    msg['Subject'] = f'Simulation_{number_episode} End'
    msg.attach(MIMEText(f"End Simulation_{number_episode}"))
    server = smtplib.SMTP('mta.uclv.edu.cu', 587)
    server.starttls()
    server.login("riacosta@uclv.cu", "rigo1993.")
    server.sendmail("riacosta@uclv.cu", "riacosta@uclv.cu", msg.as_string())
    server.quit()


def show_iter(values_iter, n_episode, val_i):
    """
    The number of iterations are show for each episode

    Args:
        values_iter: list of iterations
        n_episode: number of episodes in simulation
        val_i: number of the independent run
    """
    _, ax = plt.subplots()
    ax.bar(np.arange(n_episode), values_iter)
    ax.set_xticks(list(np.arange(0, n_episode, 5)))
    ax.set_xlabel(f'Episodes')
    ax.set_ylabel(f'Num of iterations')
    ax.set_title(f'Run_{val_i}')
    plt.savefig(f'Iter_x_Episode_{val_i}.png', dpi=100)
    plt.close()


def fig_status(total_run):
    global_reward = []
    for i in range(total_run):
        a = np.load(f'Run_status_{i}.npz')
        global_reward.append(a['data'])

    global_reward = np.stack(global_reward)
    with open('fig_status.pickle', 'wb') as f:
        pickle.dump([global_reward], f)


def fig_6(total_run):
    global_reward = []
    for i in range(total_run):
        a = np.load(f'Run_{i}.npz')
        global_reward.append(a['data'])

    global_reward = np.stack(global_reward)
    with open('fig_6.pickle', 'wb') as f:
        pickle.dump([global_reward], f)


def fig_7(total_run):
    global_reward = []
    for i in range(total_run):
        a = np.load(f'Run_load_{i}.npz')
        global_reward.append(a['data'])

    global_reward = np.stack(global_reward)
    with open('fig_7.pickle', 'wb') as f:
        pickle.dump([global_reward], f)


def fig_11(total_run):
    global_reward = []
    for i in range(total_run):
        a = np.load(f'Run_backhaul_drone{i}.npz')
        global_reward.append(a['data'])

    global_reward = np.stack(global_reward)
    with open('fig_11.pickle', 'wb') as f:
        pickle.dump([global_reward], f)


def fig_12(total_run):
    global_reward = []
    for i in range(total_run):
        a = np.load(f'Run_backhaul_global{i}.npz')
        global_reward.append(a['data'])

    global_reward = np.stack(global_reward)
    with open('fig_12.pickle', 'wb') as f:
        pickle.dump([global_reward], f)


def fig_13(total_run):
    global_reward = []
    for i in range(total_run):
        a = np.load(f'Run_worse_{i}.npz')
        global_reward.append(a['data'])

    global_reward = np.stack(global_reward)
    with open('fig_13.pickle', 'wb') as f:
        pickle.dump([global_reward], f)


def fig_sinr(total_run):
    global_reward = []
    for i in range(total_run):
        a = np.load(f'Run_sinr_{i}.npz')
        global_reward.append(a['data'])

    global_reward = np.stack(global_reward)
    with open('fig_sinr.pickle', 'wb') as f:
        pickle.dump([global_reward], f)


def search_worse_user(user_list):
    """

    :param user_list: List of users in environment
    :return: index worse throughput user
    """
    dict_user = {}
    for idx, user in enumerate(user_list):
        if user.throughput != 0:
            dict_user[idx] = user.throughput
    sorted_dict = {k: v for k, v in sorted(dict_user.items(), key=lambda item: item[1])}
    idx = list(sorted_dict.keys())
    if len(idx) == 0:
        return 0
    else:
        return idx[0]


def function_simulation(run_i=0, n_episodes=5, ep_greedy=0, n_agents=16, frequency="1e09", mail=False, n_users=200,
                        s_render=0):
    """
    Simulation drone environment using Q-Learning
    """
    init_time = time.time()

    frequency_list = [float(item) for item in frequency.split(',')]
    agents = [Drone(frequency_list) for _ in range(n_agents)]
    for index, agent in enumerate(agents):
        agent.name = f'Drone_{index}'

    if ep_greedy == 0:  # TODO: e-greedy decay
        epsilon = 1
    else:  # TODO: e-greedy fixed value
        epsilon = ep_greedy

    env = MultiDroneEnv(agents, frequency=frequency_list, n_users=n_users)

    actions_name = []
    for action_name in agents[0].actions:
        actions_name.append(action_name.name)

    metric = Metric(run_i, n_episodes, actions_name)
    old_obs = env.reset()
    env.dir_sim = f'Run_{run_i}'

    env.epsilon = epsilon

    env.render(filename=f'Episode_0.png')

    l_rate = 0.9
    discount = 0.9

    num_iter_per_episode = 1000
    num_max_iter_same_rew = 20

    best_scenario = [0, 'best']
    num_max = 0
    iter_x_episode = []

    for episode in range(n_episodes):
        for iteration in count():

            if not ep_greedy:
                env.epsilon = np.exp(-iteration / 5)

            # TODO: Choice action
            actions_array = []  # TODO: Action selected
            actions_val_array = []  # TODO: Action validated
            for id_d, drone in enumerate(env.agents):
                action_ok, action_selected = drone.choice_action(old_obs[id_d], env.epsilon)
                actions_array.append(action_selected)
                actions_val_array.append(action_ok)

            reward, new_obs, done, _ = env.step(actions_val_array)

            # TODO: Learn agents
            for id_d, drone in enumerate(env.agents):
                drone.learn(old_obs[id_d], new_obs[id_d], [l_rate, discount, reward, actions_array[id_d]])

            # TODO: Select the best scenario
            actual_scenario = [reward, 'actual']
            both_scenario = [actual_scenario, best_scenario]
            s_f = sorted(both_scenario, key=itemgetter(0), reverse=True)

            # TODO: Update Criteria
            if s_f[0][1] == 'actual':
                best_scenario.clear()
                best_scenario = actual_scenario.copy()
                best_scenario[1] = 'best'
                num_max = 0
                for drone in env.agents:
                    drone.save_best()
                for user in env.user_list:
                    user.save_best()
            else:
                num_max += 1

            # TODO: Update observation spaces
            old_obs = new_obs.copy()

            # TODO: Stopping Criteria
            # First Condition
            if iteration == num_iter_per_episode - 1:
                num_max = 0
                break

            # Second Condition
            if num_max == num_max_iter_same_rew - 1:
                num_max = 0
                break

            # New Condition
            if done:
                num_max = 0
                break

        iter_x_episode.append(iteration)
        # TODO: Load best scenario
        save_pos = []
        for drone in env.agents:
            save_pos.append(drone.pos.copy())

        for drone in env.agents:
            drone.load_best()
        for user in env.user_list:
            user.load_best()
        num_max = 0

        # TODO: Update metrics
        zero_actions = (np.ones(len(env.agents), dtype='int') * agents[0].actions.stop).tolist()
        reward, new_obs, done, _ = env.step(zero_actions)
        idx_w_user = search_worse_user(env.user_list)

        metric.update(len(env.user_list), env.calc_users_connected, env.agents, env.all_freq,
                      env.user_list[idx_w_user].throughput, env.mean_sinr)

        # TODO: Update observation spaces
        old_obs = new_obs.copy()
        if s_render:
            if episode % 5 == 0:
                env.render(filename=f'Episode_{episode + 1}.png')  # TODO: Render image environment

        if episode == n_episodes - 1:
            env.render(filename=f'Episode_{episode + 1}.png')  # TODO: Render image environment

        # env.move_user()  # TODO: User movement

    metric.extra_metric(f'{env.dir_sim}', env.agents, n_episodes)
    metric.save_metric(run_i)
    show_iter(iter_x_episode, n_episodes, run_i)
    print(f'End Run {run_i:2d} -- Time:{(time.time() - init_time):.3f} s -- Users Connected {env.calc_users_connected}')
    if mail:
        if run_i % 10 == 0:
            mail_end_episode(run_i)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', help="Name of the folder where the simulations will be saved.", default='Paper')
    parser.add_argument('-e', '--episodes', help="Number of the episodes.", type=int, default=10)
    parser.add_argument('-r', '--run', help="Number of the independent run.", type=int, default=1)
    parser.add_argument('-g', '--greedy', help="Use e-greedy or e-greedy with decay", type=float, default=0.5)
    parser.add_argument('-d', '--drone', help="Number of drones", type=int, default=8)
    parser.add_argument('-u', '--users', help="Number of users", type=int, default=200)
    parser.add_argument('-m', '--mail', help='Send mail when simulation is end', type=int, default=0)
    parser.add_argument('-f', '--frequency', help="List with operations frequencies", type=str, default="1e09")
    parser.add_argument('-t', '--thread', help='Number thread', type=int, default=1)
    parser.add_argument('-s', '--show', help='Show render environment', type=int, default=0)
    args = parser.parse_args()

    if args.greedy == 0:
        print(f'\nActive e-greedy decay')
    else:
        print(f'\nActive e-greedy {args.greedy}')

    np.seterr(divide='ignore', invalid='ignore')

    main_chapter = os.getcwd()
    try:
        os.chdir(args.name)
    except:
        os.mkdir(args.name)
        os.chdir(args.name)

    now_chapter = os.getcwd()
    copy(main_chapter + f'/mapa.pickle', now_chapter + f'/mapa.pickle')
    Parallel(n_jobs=args.thread)(delayed(function_simulation)(i, args.episodes, args.greedy, args.drone, args.frequency,
                                                              args.mail, args.users, args.show)
                                 for i in range(args.run))
    fig_6(args.run)
    fig_7(args.run)
    fig_11(args.run)
    fig_12(args.run)
    fig_status(args.run)
    fig_sinr(args.run)
    fig_13(args.run)

    frames_path = 'Run_{i}/Episode_{j}.png'
    vid_name = 'Run_{i}/Run_{i}.mp4'

    if args.show:
        for i in range(args.run):
            with imageio.get_writer(vid_name.format(i=i), format='FFMPEG', mode='I', fps=1) as writer:
                writer.append_data(imageio.imread(frames_path.format(i=i, j=0)))
                for j in range(args.episodes):
                    if j % 5 == 0 or j == args.episodes - 1:
                        writer.append_data(imageio.imread(frames_path.format(i=i, j=j + 1)))

        for i in range(args.run):
            os.remove(frames_path.format(i=i, j=0))
            for j in range(args.episodes):
                if j % 5 == 0 or j == args.episodes - 1:
                    os.remove(frames_path.format(i=i, j=j + 1))

    lstFiles = []
    lstDir = os.walk(now_chapter)

    for root, dirs, files in lstDir:
        for file in files:
            (filename, extension) = os.path.splitext(file)
            if extension == ".npz":
                lstFiles.append(filename + extension)

    for file in lstFiles:
        os.remove(file)

    if args.mail:
        send_mail(args.name)
