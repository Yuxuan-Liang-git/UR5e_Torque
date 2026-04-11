function Xi = sparsifyDynamicsIndependent(Theta,dXdt,lambda,n)
% 独立稀疏化动力学系数 (每个状态独立阈值)
% 复制自 ../utils/sparsifyDynamicsIndependent.m, 移除了 keyboard 调试断点

% Initial guess using least-squares
Xi = Theta\dXdt;

% state-dependent lambda is sparsifying knob
for k=1:10
    for ind = 1:n    % n is state dimension
        % 找到贡献量小的系数强制置零
        smallinds = (abs(Xi(:,ind))<lambda(ind));   % find small coefficients
        Xi(smallinds,ind)=0;                        % and threshold
        % 用置零后新的Xi重新回归，找到新的系数
        biginds = ~smallinds;
        % Regress dynamics onto remaining terms to find sparse Xi
        Xi(biginds,ind) = Theta(:,biginds)\dXdt(:,ind);
    end
end
