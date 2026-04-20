import multiprocessing as mp
import time
import numpy as np

def live_plot_worker(data_queue):
    import matplotlib.pyplot as plt
    
    plt.ion()
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    fig.suptitle('UR5e 6-Axis Tracking')
    
    ax_pos = axes[0]
    ax_vel = axes[1]
    ax_tau = axes[2]
    
    ax_pos.set_ylabel('Position (rad)')
    ax_pos.set_title('Joint Positions')
    ax_vel.set_ylabel('Velocity (rad/s)')
    ax_vel.set_title('Joint Velocities')
    ax_tau.set_ylabel('Current (A)')
    ax_tau.set_title('Actual Motor Currents')
    ax_tau.set_xlabel('Time (s)')
    
    lines = {}
    colors = ['r', 'g', 'b', 'c', 'm', 'y']
    for i in range(6):
        lines[f'q_des_{i}'], = ax_pos.plot([], [], linestyle='--', color=colors[i], label=f'J{i} Target')
        lines[f'q_act_{i}'], = ax_pos.plot([], [], linestyle='-', color=colors[i], label=f'J{i} Actual')
        
        lines[f'dq_des_{i}'], = ax_vel.plot([], [], linestyle='--', color=colors[i], label=f'J{i} Target')
        lines[f'dq_act_{i}'], = ax_vel.plot([], [], linestyle='-', color=colors[i], label=f'J{i} Actual')
        
        lines[f'tau_{i}'], = ax_tau.plot([], [], linestyle='-', color=colors[i], label=f'J{i} Curr')
        
    ax_pos.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize='small', ncol=2)
    ax_vel.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize='small', ncol=2)
    ax_tau.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize='small', ncol=2)
    
    fig.tight_layout()
    ax_pos.grid(True)
    ax_vel.grid(True)
    ax_tau.grid(True)

    t_data = []
    q_des_data = [[] for _ in range(6)]
    q_act_data = [[] for _ in range(6)]
    dq_des_data = [[] for _ in range(6)]
    dq_act_data = [[] for _ in range(6)]
    tau_data = [[] for _ in range(6)]

    last_draw_time = time.time()

    while True:
        try:
            # Drain queue
            while not data_queue.empty():
                item = data_queue.get_nowait()
                if item is None:  # Stop signal
                    plt.ioff()
                    plt.show()  # Hold plot open at the end
                    return
                
                t_now, q_des, q, dq_des, dq, tau = item
                t_data.append(t_now)
                for i in range(6):
                    q_des_data[i].append(q_des[i])
                    q_act_data[i].append(q[i])
                    dq_des_data[i].append(dq_des[i])
                    dq_act_data[i].append(dq[i])
                    tau_data[i].append(tau[i])

            # Keep only last 500 points (~10 seconds at 50Hz queue ingestion)
            if len(t_data) > 500:
                t_data = t_data[-500:]
                for i in range(6):
                    q_des_data[i] = q_des_data[i][-500:]
                    q_act_data[i] = q_act_data[i][-500:]
                    dq_des_data[i] = dq_des_data[i][-500:]
                    dq_act_data[i] = dq_act_data[i][-500:]
                    tau_data[i] = tau_data[i][-500:]

            # Draw at ~10 Hz
            if time.time() - last_draw_time > 0.1 and len(t_data) > 0:
                for i in range(6):
                    lines[f'q_des_{i}'].set_data(t_data, q_des_data[i])
                    lines[f'q_act_{i}'].set_data(t_data, q_act_data[i])
                    
                    lines[f'dq_des_{i}'].set_data(t_data, dq_des_data[i])
                    lines[f'dq_act_{i}'].set_data(t_data, dq_act_data[i])
                    
                    lines[f'tau_{i}'].set_data(t_data, tau_data[i])

                if len(t_data) > 1 and t_data[-1] > t_data[0]:
                    ax_pos.set_xlim(t_data[0], t_data[-1])
                    ax_vel.set_xlim(t_data[0], t_data[-1])
                    ax_tau.set_xlim(t_data[0], t_data[-1])
                
                # compute limits for position
                valid_q = [val for arr in q_des_data + q_act_data for val in arr]
                if valid_q:
                    min_q, max_q = min(valid_q), max(valid_q)
                    margin_q = max(abs(max_q - min_q) * 0.1, 0.01)
                    ax_pos.set_ylim(min_q - margin_q, max_q + margin_q)
                
                # compute limits for velocity
                valid_dq = [val for arr in dq_des_data + dq_act_data for val in arr]
                if valid_dq:
                    min_dq, max_dq = min(valid_dq), max(valid_dq)
                    margin_dq = max(abs(max_dq - min_dq) * 0.1, 0.01)
                    ax_vel.set_ylim(min_dq - margin_dq, max_dq + margin_dq)
                    
                # compute limits for torque
                valid_tau = [val for arr in tau_data for val in arr]
                if valid_tau:
                    min_tau, max_tau = min(valid_tau), max(valid_tau)
                    margin_tau = max(abs(max_tau - min_tau) * 0.1, 0.1)
                    ax_tau.set_ylim(min_tau - margin_tau, max_tau + margin_tau)

                fig.canvas.flush_events()
                last_draw_time = time.time()

            time.sleep(0.01)

        except KeyboardInterrupt:
            break
