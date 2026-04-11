%% POS_CTRL.m - 基于 SINDy 模型的电机位置控制 (性能对比版)
% 对比方案:
% 1. SINDy 补偿控制 (Feedback Linearization + PID)
% 2. 纯 PD 控制 (Pure PD tracking)

clear all; close all; clc;

% 路径设置
addpath('../utils');
datapath = '../DATA/MOTOR/';
SystemModel = 'SingleMotor';
ModelName = 'SINDYc';

% 1. 加载辨识出的 SINDy 模型
savefile = fullfile(datapath, ['EX_', SystemModel, '_SI_', ModelName, '.mat']);
if ~exist(savefile, 'file')
    error('未找到模型文件: %s。请先运行 EX_JOINT_SI_SINDYc.m', savefile);
end
load(savefile); % 加载 Model 结构体

% 提取模型关键参数
Xi = Model.Xi;
Nvar = Model.Nvar;
Nctrl = Model.Nctrl;
polyorder = Model.polyorder;

fprintf('============================================\n');
fprintf('SINDy 补偿 vs. 纯 PD 控制对比仿真\n');
fprintf('============================================\n');

% 2. 仿真参数
dt = 0.01;
T_total = 10;
tspan = 0:dt:T_total;
N = length(tspan);

% 3. 控制增益 (两组方案共用一套基础 PID 参数)
Kp = 10.0;
Ki = 0.0;
Kd = 5.0;

% 4. 仿真循环
% 存储两个方案的历程
% modes: 1 = SINDy Compensated, 2 = Pure PD
histories = cell(1, 2);

for mode = 1:2
    if mode == 1
        fprintf('>> 运行方案 1: SINDy 补偿控制...\n');
    else
        fprintf('>> 运行方案 2: 纯 PD 控制 (无模型补偿)...\n');
    end
    
    % 初始化
    x = [0.2; 0.0]; % 初始状态 [q; dq]
    error_int = 0;
    history = struct('t', tspan, 'x', zeros(Nvar, N), 'u', zeros(Nctrl, N), 'ref', zeros(1, N));
    history.x(:, 1) = x;
    
    for i = 1:N-1
        t = tspan(i);
        q = x(1);
        dq = x(2);
        
        % --- 5.1 目标轨迹 ---
        q_ref = 1.0*sin(t) + 0.5*sin(2*t);
        dq_ref = 1.0*cos(t) + 1.0*cos(2*t);
        ddq_ref = -1.0*sin(t) - 2.0*sin(2*t);
        
        % --- 5.2 误差计算 ---
        e = q_ref - q;
        de = dq_ref - dq;
        error_int = error_int + e * dt;
        
        % --- 5.3 期望加速度指令 v (基于误差) ---
        v = ddq_ref + Kp*e + Ki*error_int + Kd*de;
        
        % --- 5.4 控制量计算 ---
        if mode == 1
            % [方案 A] SINDy 动态补偿 (Feedback Linearization)
            % f(x): 固有动态 (u=0)
            Theta_auto = poolDataMotor([q, dq, 0], Nvar + Nctrl, polyorder);
            dq_dot_auto = Theta_auto * Xi(:, 2);
            
            % g(x): 控制增益 (单位 u 产生的变化)
            Theta_u1 = poolDataMotor([q, dq, 1], Nvar + Nctrl, polyorder);
            g_identified = (Theta_u1 - Theta_auto) * Xi(:, 2);
            
            % 抵消项 u = v - f / g,这样可以用同一套PD对比补偿的效果
            u = v - dq_dot_auto / g_identified;
        else
            % [方案 B] 纯 PD 控制
            % 直接将期望加速度 v 作为力矩输入 (假设增益为 1，且不补偿摩擦/阻尼)
            u = v;
        end
        
        % 限制控制力矩
        u = max(min(u, 50), -50);
        
        % --- 5.5 应用到物理系统 ---
        x = rk4u(@JointSys, x, u, dt, 1, t, []);
        
        % 记录
        history.x(:, i+1) = x;
        history.u(:, i) = u;
        history.ref(i) = q_ref;
    end
    histories{mode} = history;
end

fprintf('  仿真全部完成!\n');

% 6. 对比绘图
figure('Position', [100, 100, 1000, 800]);

% 子图 1: 角度跟踪对比
subplot(3, 1, 1);
plot(tspan, histories{1}.ref, 'k--', 'LineWidth', 1.5); hold on;
plot(tspan, histories{1}.x(1, :), 'b', 'LineWidth', 1.2);
plot(tspan, histories{2}.x(1, :), 'r', 'LineWidth', 1.2);
ylabel('角度 (rad)');
legend('参考轨迹 (Ref)', 'SINDy 补偿 (Compensated)', '纯 PD (Pure PD)');
title('位置跟踪性能对比');
grid on;

% 子图 2: 跟踪误差对比
subplot(3, 1, 2);
plot(tspan, histories{1}.ref - histories{1}.x(1, :), 'b', 'LineWidth', 1.2); hold on;
plot(tspan, histories{2}.ref - histories{2}.x(1, :), 'r', 'LineWidth', 1.2);
ylabel('误差 (rad)');
legend('SINDy 补偿误差', '纯 PD 误差');
title('跟踪误差对比 (误差越小越好)');
grid on;

% 子图 3: 控制输入 (力矩)
subplot(3, 1, 3);
plot(tspan, histories{1}.u, 'b', 'LineWidth', 1.2); hold on;
plot(tspan, histories{2}.u, 'r', 'LineWidth', 1.2);
xlabel('时间 (s)');
ylabel('力矩 (Nm)');
legend('SINDy 补偿力矩', '纯 PD 力矩');
title('控制力矩消耗');
grid on;

% 计算并显示 RMSE
rmse1 = rms(histories{1}.ref - histories{1}.x(1, :));
rmse2 = rms(histories{2}.ref - histories{2}.x(1, :));
fprintf('\n统计结果:\n');
fprintf('  SINDy 补偿控制 RMSE: %.6f rad\n', rmse1);
fprintf('  纯 PD 控制 RMSE:     %.6f rad\n', rmse2);
fprintf('  误差降低比率:        %.2f%%\n', (rmse2-rmse1)/rmse2 * 100);

% 保存对比图
saveas(gcf, fullfile('../FIGURES/MOTOR/', 'Control_Comparison_SINDy.png'));
