%% 1-DOF 单电机数据生成
%
% 输入变量:
%   InputSignalType
%   noise_level
%   DERIV_NOISE

%% 初始参数设置
x0  = [0.0, 0.0];                    % 初始状态 [q, dq]
n   = length(x0);                    % 状态数 = 2
nu  = 1;                             % 控制输入数 = 1

dt  = 0.005;                         % 时间步长 (200Hz)
options = odeset('RelTol',1e-8,'AbsTol',1e-8*ones(1,n));

if ~exist('noise_level', 'var'), noise_level = 0; end
if ~exist('DERIV_NOISE', 'var'), DERIV_NOISE = 0; end

%% 激励信号设计
T_total = 100;
tspan = 0:dt:T_total;
Ntrain = floor(length(tspan) / 3);
tau_max = 5.0;  % 最大力矩 [Nm]

fprintf('  生成单电机训练数据 (信号: %s)...\n', InputSignalType);

switch InputSignalType
    case 'sine2'
        forcing = @(x,t) tau_max * (sin(2.0*t) + 0.5*sin(4.0*t) + 0.5*sin(0.5*t));
    case 'chirp'
        forcing = @(x,t) tau_max * chirp(t, 0.1, T_total, 5);
    case 'noise'
        % 平滑后的随机力矩
        rng(42);
        u_raw = tau_max * randn(length(tspan), 1);
        [b_filt, a_filt] = butter(3, 0.05);
        u_raw = filtfilt(b_filt, a_filt, u_raw);
        forcing = @(x,t) interp1(tspan, u_raw, t, 'linear', 'extrap');
    otherwise
        forcing = @(x,t) tau_max * sin(t);
end

%% 仿真积分
tic;
[t,x] = ode45(@(t,x) JointSys(t,x,forcing(x,t),[]), tspan, x0, options);
fprintf('  积分完成, 耗时 %.2f 秒\n', toc);

% 提取控制输入
u = zeros(length(tspan), nu);
for i = 1:length(tspan)
    u(i) = forcing(0, tspan(i));
end

%% 添加测量噪声
if noise_level > 0
    rng(123, 'twister');
    for col = 1:n
        x(:,col) = x(:,col) + noise_level * std(x(:,col)) * randn(size(x,1), 1);
    end
    fprintf('  添加了 %.1f%% 测量噪声\n', noise_level*100);
end

%% 分割训练集和验证集
xv = x(Ntrain+1:end,:);
x  = x(1:Ntrain,:);
uv = u(Ntrain+1:end,:);
u  = u(1:Ntrain,:);
tv = t(Ntrain+1:end);
t  = t(1:Ntrain);
tspanv = tspan(Ntrain+1:end);
tspan  = tspan(1:Ntrain);

%% 可视化训练数据
figure('Name', 'Motor Data', 'Position', [100 100 800 300]);
subplot(1,3,1); plot(tspan, x(:,1), 'LineWidth', 1.5); title('Angle [rad]'); grid on;
subplot(1,3,2); plot(tspan, x(:,2), 'LineWidth', 1.5); title('Velocity [rad/s]'); grid on;
subplot(1,3,3); plot(tspan, u, 'r', 'LineWidth', 1.5); title('Torque [Nm]'); grid on;
sgtitle('Single Motor Training Data');
