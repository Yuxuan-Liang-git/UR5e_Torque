function dy = JointSys(t, y, u, p)
% JointSys (现已替换为单电机模型)
%
% 单电机带摩擦力动力学模型
% 状态: y = [q, dq] (角度, 角速度)
% 输入: u = tau (电机力矩)
%
% 动力学方程:
%   dq/dt = dq
%   ddq/dt = (1/J) * (tau - B*dq - Fc*tanh(10*dq))

q  = y(1);
dq = y(2);
tau = u;

% 电机参数
J = 1.0;      % 转动惯量 [kg*m^2]
B = 0.5;       % 粘性摩擦系数 [N*m*s/rad]
Fc = 0.3;     % 库仑摩擦系数 [N*m]

% 使用 tanh 近似 sign 函数避免 ODE 刚性问题
friction = B * dq + Fc * tanh(10 * dq);

ddq = (tau - friction) / J;

dy = [dq; ddq];
end
