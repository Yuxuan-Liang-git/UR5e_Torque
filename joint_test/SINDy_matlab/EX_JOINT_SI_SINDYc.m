% 单电机系统 - SINDy 系统辨识
%
% 状态: [q, dq] (2维)
% 控制: [tau] (1维)

clear all, close all, clc

%% 路径设置
figpath  = '../FIGURES/MOTOR/'; mkdir(figpath);
datapath = '../DATA/MOTOR/';   mkdir(datapath);
addpath('../utils');

SystemModel = 'SingleMotor';
Nvar = 2;    % [q, dq]
Nctrl = 1;   % [tau]

%% ==================== 参数配置 ====================
InputSignalType = 'sine2';        % 'sine2','chirp','noise'
noise_level = 0.0;                % 测量噪声比例
DERIV_NOISE = 0;                  % 1=TVRegDiff, 0=中心差分

% SINDy 参数
ModelName = 'SINDYc';
polyorder = 2;                    % 仅使用线性项，防止高阶误差积累导致仿真发散
usesine   = 0;                    % 单电机不需要三角项

% 稀疏化阈值 (基于 dx 归一化后的相对贡献值)
% q 方程 (精确 dq=dq), 小阈值 0.001
% dq 方程 (复杂动力学), 阈值 0.01~0.05
lambda_vec = [0.001, 0.001];

fprintf('============================================\n');
fprintf('1-DOF 单电机 SINDy 系统辨识\n');
fprintf('============================================\n');

%% ==================== 1. 生成数据 ====================
getTrainingData

%% ==================== 2. SINDy 辨识 ====================
trainSINDYc_Joint

%% ==================== 3. 训练集预测 ====================
fprintf('\n>> 训练集预测...\n');

p.ahat = Xi(:, 1:Nvar);
p.polyorder = polyorder;
p.usesine = usesine;
p.dt = dt;

[N, Ns] = size(x);
xSINDYc = zeros(Ns, N);
xSINDYc(:, 1) = x(1,:)';

for ct = 1:N-1
    % 使用标准的离散 SINDy 预测函数
    xSINDYc(:, ct+1) = rk4u(@sparseGalerkinControl_Motor, ...
        xSINDYc(:,ct), u(ct,:)', dt, 1, [], p);
end
xSINDYc = xSINDYc';

train_err = x - xSINDYc;
fprintf('  训练RMSE: q=%.6f rad, dq=%.6f rad/s\n', ...
    rms(train_err(:,1)), rms(train_err(:,2)));

%% ==================== 4. 验证集预测 ====================
fprintf('\n>> 验证集预测...\n');

xA = xv;
tA = tv;

[N_val, ~] = size(xA);
xB = zeros(Nvar, N_val);
xB(:,1) = x(end,:)';

for ct = 1:N_val-1
    xB(:, ct+1) = rk4u(@sparseGalerkinControl_Motor, ...
        xB(:,ct), uv(ct,:)', dt, 1, [], p);
end
xB = xB(:, 1:N_val)';
tB = tspanv;

%% ==================== 5. 可视化 ====================
fprintf('\n>> 可视化...\n');
VIZ_SI_Validation_Joint

%% ==================== 6. 保存 ====================
Model.name = 'SINDYc';
Model.polyorder = polyorder;
Model.usesine = usesine;
Model.Xi = Xi;
Model.dt = dt;
Model.Nvar = Nvar;
Model.Nctrl = Nctrl;
Model.lambda_vec = lambda_vec;
Model.SystemModel = SystemModel;

savefile = fullfile(datapath, ['EX_', SystemModel, '_SI_', ModelName, '.mat']);
save(savefile, 'Model');
fprintf('  模型已保存: %s\n', savefile);
