import argparse

import matplotlib.pyplot as plt
import numpy as np
import rowan
import cvxpy as cp

import cfusdlog

def apply_motor_filter(action_scaled: np.ndarray,
                       prev_filtered_rpm_proxy: np.ndarray,
                       change: np.ndarray):
    dt = 0.004
    tau = 0.08 #T/4
    alpha = dt / tau  

    next_filtered_rpm_proxy = prev_filtered_rpm_proxy + alpha * (np.sqrt(action_scaled) - prev_filtered_rpm_proxy)
    filtered_thrust = np.square(next_filtered_rpm_proxy)
    return filtered_thrust, next_filtered_rpm_proxy

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("file_usd")
    args = parser.parse_args()

    data_usd = cfusdlog.decode(args.file_usd)

    start_time = np.inf
    for _,v in data_usd.items():
        start_time = min(start_time, v['timestamp'][0])

    # start_time = 0

    # make sure we have two events, one for when we switched to RL and one when we used back
    assert(data_usd['controllerChanged']['ctrl'][0] == 6)
    assert(data_usd['controllerChanged']['ctrl'][1] == 2)

    time_fF = (data_usd['fixedFrequency']['timestamp'] - start_time) / 1e3

    rl_start = (data_usd['controllerChanged']['timestamp'][0] - start_time) / 1e3
    rl_end = (data_usd['controllerChanged']['timestamp'][1] - start_time) / 1e3

    rl_idx_start = np.where(time_fF >= rl_start)[0][0]
    rl_idx_end = np.where(time_fF >= rl_end)[0][0]


    T = len(data_usd['fixedFrequency']['timestamp'])

    policy_in = np.array([
        data_usd['fixedFrequency']['ctrlrl.in0'],
        data_usd['fixedFrequency']['ctrlrl.in1'],
        data_usd['fixedFrequency']['ctrlrl.in2'],
        data_usd['fixedFrequency']['ctrlrl.in3'],
        data_usd['fixedFrequency']['ctrlrl.in4'],
        data_usd['fixedFrequency']['ctrlrl.in5'],
        data_usd['fixedFrequency']['ctrlrl.in6'],
        data_usd['fixedFrequency']['ctrlrl.in7'],
        data_usd['fixedFrequency']['ctrlrl.in8'],
        data_usd['fixedFrequency']['ctrlrl.in9'],        
        data_usd['fixedFrequency']['ctrlrl.in10'],
        data_usd['fixedFrequency']['ctrlrl.in11'],
        data_usd['fixedFrequency']['ctrlrl.in12'],
        data_usd['fixedFrequency']['ctrlrl.in13'],
        data_usd['fixedFrequency']['ctrlrl.in14'],
        data_usd['fixedFrequency']['ctrlrl.in15'],
        data_usd['fixedFrequency']['ctrlrl.in16'],
        data_usd['fixedFrequency']['ctrlrl.in17'],
        data_usd['fixedFrequency']['ctrlrl.in18'],
        data_usd['fixedFrequency']['ctrlrl.in19'], 
        data_usd['fixedFrequency']['ctrlrl.in20'],
        data_usd['fixedFrequency']['ctrlrl.in21'],
        data_usd['fixedFrequency']['ctrlrl.in22'],
        data_usd['fixedFrequency']['ctrlrl.in23'],
        data_usd['fixedFrequency']['ctrlrl.in24'],
        data_usd['fixedFrequency']['ctrlrl.in25'],
        data_usd['fixedFrequency']['ctrlrl.in26'],
        data_usd['fixedFrequency']['ctrlrl.in27'],
        ]).T[rl_idx_start:rl_idx_end]
    # print(policy_in.shape)
    print(policy_in[0, 24:28])
    
    policy_out = np.array([
        data_usd['fixedFrequency']['ctrlrl.out0'],
        data_usd['fixedFrequency']['ctrlrl.out1'],
        data_usd['fixedFrequency']['ctrlrl.out2'],
        data_usd['fixedFrequency']['ctrlrl.out3']]).T[rl_idx_start:rl_idx_end]
    
    policy_out = np.clip(policy_out, -1, 1)
    
    force_req_newtons = (policy_out + 1.0) / 2.0 * 0.118
    force_req_grams = force_req_newtons / 9.81 * 1000.0

    force_req_grams_filtered = np.zeros_like(force_req_grams)
    filtered_rpm_prev = np.sqrt(force_req_grams[0])
    # print(filtered_rpm_prev)
    # exit()

    # print(np.diff(time_fF))
    # exit()
    for i, (a, change) in enumerate(zip(force_req_grams, np.diff(force_req_grams, axis=0))):
        # print(i, a, change)
        # exit()
        force_req_grams_filtered[i], filtered_rpm_prev = apply_motor_filter(a, filtered_rpm_prev, change)
        # print(force_req_grams_filtered[i], filtered_rpm_prev)
    # exit()
        # filtered_thrust_hist.append(thrust_filt[0])
        # rpm_proxy_hist.append(filtered_rpm_prev[0])

    # filtered_thrust_hist = np.array(filtered_thrust_hist)
    # rpm_proxy_hist = np.array(rpm_proxy_hist)

    # motor fun

    rpm = np.array([
        data_usd['fixedFrequency']['rpm.m1'],
        data_usd['fixedFrequency']['rpm.m2'],
        data_usd['fixedFrequency']['rpm.m3'],
        data_usd['fixedFrequency']['rpm.m4']]).T
    
    pwm = np.array([
        data_usd['fixedFrequency']['motor.m1'],
        data_usd['fixedFrequency']['motor.m2'],
        data_usd['fixedFrequency']['motor.m3'],
        data_usd['fixedFrequency']['motor.m4'],
    ]).T
    vbat = np.array(data_usd['fixedFrequency']['pm.vbatMV']) / 1000.0

    print(rpm.shape)


    # exit()

    mass = 36.0 # g
    expected_thrust_per_rotor = mass / 4

    # fit kw's
    kws = np.zeros(4)
    for i in range(4):
        kw = cp.Variable()
        cost = cp.sum_squares(expected_thrust_per_rotor - kw * rpm[0:rl_idx_start,i]**2)
        prob = cp.Problem(cp.Minimize(cost), [])
        prob.solve()
        kws[i] = kw.value
    # print(kws)

    # fitted = kw.value * rpm**2
    # ax[1].plot(rpm, fitted, label='fit')
    # # ax[2].set_xlabel('rpm')
    # # ax[2].set_ylabel('fitted thrust [g]')
    # ax[1].legend()
    # ax[1].grid(True)

    # plt.show()


    # exit()
    
    kw = 2.2620546776378156e-08
    force_in_grams = kw * rpm**2
    for i in range(4):
        force_in_grams[:,i] = kws[i] * rpm[:,i]**2

    force_in_grams_from_pwm = 1.65049399e-09 *pwm**2 + 9.44396129e-05 * pwm -3.77748052e-01
    
    
    #-5.360718677769569 + pwm * 0.0005492858445116151


    fig, ax = plt.subplots(3, 2, sharex='all', sharey='all')
    # ax[0,1].plot(time_fF, rpm[:,0])
    # ax[0,1].set_ylabel(f"M1 [rpm]")
    # ax[1,1].plot(time_fF, rpm[:,1])
    # ax[1,1].set_ylabel(f"M2 [rpm]")
    # ax[1,0].plot(time_fF, rpm[:,2])
    # ax[1,0].set_ylabel(f"M3 [rpm]")
    # ax[0,0].plot(time_fF, rpm[:,3])
    # ax[0,0].set_ylabel(f"M4 [rpm]")


    ax[0,1].plot(time_fF, force_in_grams[:,0], label="rpm")
    ax[0,1].plot(time_fF[rl_idx_start:rl_idx_end], force_req_grams[:,0], label="desired")
    ax[0,1].plot(time_fF[rl_idx_start:rl_idx_end], force_req_grams_filtered[:,0], label="desired w/ filter")
    ax[0,1].plot(time_fF, force_in_grams_from_pwm[:,0], label="pwm")
    # ax[0,1].plot(time_fF, policy_out[:,0], label="policy")

    ax[0,1].set_ylabel(f"M1 [grams]")

    ax[1,1].plot(time_fF, force_in_grams[:,1], label="rpm")
    ax[1,1].plot(time_fF[rl_idx_start:rl_idx_end], force_req_grams[:,1], label="desired")
    ax[1,1].plot(time_fF[rl_idx_start:rl_idx_end], force_req_grams_filtered[:,1], label="desired w/ filter")
    ax[1,1].plot(time_fF, force_in_grams_from_pwm[:,1], label="pwm")
    ax[1,1].set_ylabel(f"M2 [grams]")
    
    ax[1,0].plot(time_fF, force_in_grams[:,2], label="rpm")
    ax[1,0].plot(time_fF[rl_idx_start:rl_idx_end], force_req_grams[:,2], label="desired")
    ax[1,0].plot(time_fF[rl_idx_start:rl_idx_end], force_req_grams_filtered[:,2], label="desired w/ filter")
    ax[1,0].plot(time_fF, force_in_grams_from_pwm[:,2], label="pwm")
    ax[1,0].set_ylabel(f"M3 [grams]")
    
    ax[0,0].plot(time_fF, force_in_grams[:,3], label="rpm")
    ax[0,0].plot(time_fF[rl_idx_start:rl_idx_end], force_req_grams[:,3], label="desired")
    ax[0,0].plot(time_fF[rl_idx_start:rl_idx_end], force_req_grams_filtered[:,3], label="desired w/ filter")
    ax[0,0].plot(time_fF, force_in_grams_from_pwm[:,3], label="pwm")
    ax[0,0].set_ylabel(f"M4 [grams]")

    ax[2,0].plot(time_fF[rl_idx_start:rl_idx_end], policy_in[:,21], label="r")
    ax[2,0].plot(time_fF[rl_idx_start:rl_idx_end], policy_in[:,22], label="p")
    ax[2,0].plot(time_fF[rl_idx_start:rl_idx_end], policy_in[:,23], label="y")

    ax[2,0].set_ylabel(f"omega")
    ax[2,0].legend()
    


    for x in range(2):
        for y in range(2):
            ax[x,y].axvline(x = rl_start, color = 'b')
            ax[x,y].axvline(x = rl_end, color = 'b')

    ax[0,0].legend()


    # ax[2,0].plot(time_fF, np.sum(force_in_grams, axis=1))
    # ax[2,0].set_ylabel(f"total thrust [grams]")

    plt.show()

    plt.plot(data_usd['fixedFrequency']['timestamp'], policy_in[:,21], label="roll")
    plt.plot(data_usd['fixedFrequency']['timestamp'], policy_in[:,22], label="p")
    plt.plot(data_usd['fixedFrequency']['timestamp'], policy_in[:,23], label="y")

    plt.show()