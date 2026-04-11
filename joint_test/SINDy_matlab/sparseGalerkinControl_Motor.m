function ykplus1 = sparseGalerkinControl_Motor(t, y, u, p)
% 离散时间 SINDy 模型预测
% 使用标准 poolData 构建候选库

if isfield(p, 'SelectVars') == 0
    p.SelectVars = 1:length(y);
end

polyorder = p.polyorder;
ahat      = p.ahat;

% 拼接状态和控制
y_sel = y(p.SelectVars)';  % 1 x Nstate
u_row = u(:)';             % 1 x Nctrl
xaug = [y_sel, u_row];     % 1 x (Nstate + Nctrl)
nVars = length(xaug);

% 使用自定义电机库
yPool = poolDataMotor(xaug, nVars, polyorder);
ykplus1 = (yPool * ahat)';
end
