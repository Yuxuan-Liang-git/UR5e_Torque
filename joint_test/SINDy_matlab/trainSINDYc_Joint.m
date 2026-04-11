%% SINDYc 训练 - 单电机模型
% 使用标准 poolData 和 sparsifyDynamics

fprintf('\n========================================\n');
fprintf('SINDYc 训练 - 1-DOF 单电机模型\n');
fprintf('========================================\n');

Nstate = size(x, 2);   % 2
Nctrl  = size(u, 2);   % 1
NvarTotal = Nstate + Nctrl; % 3

fprintf('  状态: %d, 控制: %d, 总变量: %d\n', Nstate, Nctrl, NvarTotal);

%% 计算导数
if DERIV_NOISE == 0
    fprintf('  导数: 四阶中心差分\n');
    dx = zeros(length(x)-5, Nstate);
    for i = 3:length(x)-3
        for k = 1:Nstate
            dx(i-2,k) = (1/(12*dt))*(-x(i+2,k)+8*x(i+1,k)-8*x(i-1,k)+x(i-2,k));
        end
    end
    xaug = [x(3:end-3,:), u(3:end-3,:)];
else
    fprintf('  导数: TVRegDiff\n');
    tvr_iter = 20;
    tvr_alph = 0.00002;
    if noise_level > 0
        tvr_alph = tvr_alph * (1 + 100*noise_level);
    end
    dx = [];
    tic;
    for i = 1:Nstate
        dx_i = TVRegDiff(x(:,i), tvr_iter, tvr_alph, [], 'small', 1e12, dt, 0, 0);
        dx = [dx, dx_i(2:end)];
    end
    trim_s = 50; trim_e = 51;
    xt = [];
    for i = 1:Nstate
        xt_i = cumsum(dx(:,i))*dt;
        xt_i = xt_i - (mean(xt_i(trim_s:end-trim_e)) - mean(x(trim_s:end-trim_e,i)));
        xt = [xt, xt_i];
    end
    xt = xt(trim_s:end-trim_e,:);
    dx = dx(trim_s:end-trim_e,:);
    xaug = [xt, u(trim_s:end-trim_e,:)];
end

n_total = size(dx, 2);

%% 构建候选函数库
% 使用自定义单电机库，内含多种 tanh 和 sign 基函数
fprintf('  构建自定义电机库: polyorder=%d\n', polyorder);

clear Theta Xi
Theta = poolDataMotor(xaug, NvarTotal, polyorder);
Nlib = size(Theta, 2);

% 归一化自变量库
Theta_norm = zeros(Nlib, 1);
for i = 1:Nlib
    Theta_norm(i) = norm(Theta(:,i));
    if Theta_norm(i) > 0
        Theta(:,i) = Theta(:,i) ./ Theta_norm(i);
    end
end

% 归一化导数 (因差分计算得到的 dx_2 可能极大)
norm_dx = zeros(1, Nstate);
for i = 1:Nstate
    norm_dx(i) = norm(dx(:,i));
    if norm_dx(i) > 0
        dx(:,i) = dx(:,i) / norm_dx(i);
    end
end

% 稀疏回归 (使用 Ridge Regression 降低共线性)
alpha_ridge = 1e-4; 
Xi = (Theta'*Theta + alpha_ridge*eye(Nlib)) \ (Theta'*dx);

if exist('lambda_vec','var') && length(lambda_vec) >= Nstate
    for k = 1:10
        for ind = 1:Nstate
            smallinds = (abs(Xi(:,ind)) < lambda_vec(ind));
            Xi(smallinds, ind) = 0;
            biginds = ~smallinds;
            if any(biginds)
                Xi(biginds, ind) = (Theta(:,biginds)'*Theta(:,biginds) + alpha_ridge*eye(sum(biginds))) \ (Theta(:,biginds)'*dx(:,ind));
            end
        end
    end
else
    % Fallback if lambda_vec not defined
    Xi = sparsifyDynamics(Theta, dx, lambda, Nstate);
end

% 反归一化
for i = 1:Nstate
    if norm_dx(i) > 0
        Xi(:,i) = Xi(:,i) * norm_dx(i);
    end
end
for i = 1:Nlib
    if Theta_norm(i) > 0
        Xi(i,:) = Xi(i,:) ./ Theta_norm(i);
    end
end

% 保证与 utils 的兼容, 输出是库列宽 x 状态宽
Xi_ctrl = zeros(Nlib, Nctrl);
Xi = [Xi, Xi_ctrl];

%% 显示结果
str_vars = {'q', 'dq', 'u'};
yout = poolDataLISTMotor(str_vars, Xi, NvarTotal, polyorder);

nz = nnz(Xi(:,1:Nstate));
total_coeff = numel(Xi(:,1:Nstate));
fprintf('\n  稀疏率: %.1f%% (%d/%d 非零)\n', (1-nz/total_coeff)*100, nz, total_coeff);
for i = 1:Nstate
    fprintf('    状态 %d: %d 项\n', i, nnz(Xi(:,i)));
end

fprintf('========================================\n');
fprintf('SINDYc 训练完成\n');
fprintf('========================================\n');
