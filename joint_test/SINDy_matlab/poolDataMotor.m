function yout = poolDataMotor(yin, nVars, polyorder)
% 单电机专用候选函数库
% 包含多项式基函数和标准的库仑摩擦力基函数 sign(dq)
%
% yin = [q, dq, u] (对于单电机模型)
% dq 是第 2 列

% 基础多项式库 (不含全局 sin/cos)
yout = poolData(yin, nVars, polyorder, 0);

% --- 扩展标准的库仑摩擦力基函数 ---
dq = yin(:, 2);

% yout = [yout, tanh(10*dq)];
yout = [yout, sign(dq)];

end
