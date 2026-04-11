function yout = poolDataLISTMotor(str_vars, ahat, nVars, polyorder)
% 单电机专用候选函数库的标签生成 (匹配带有 sign(dq) 的自定义库)

ind = 1;
labels{ind,1} = '1';
ind = ind + 1;

for i = 1:nVars
    labels{ind,1} = str_vars{i};
    ind = ind + 1;
end

if polyorder >= 2
    for i = 1:nVars
        for j = i:nVars
            labels{ind,1} = [str_vars{i}, '*', str_vars{j}];
            ind = ind + 1;
        end
    end
end

if polyorder >= 3
    for i = 1:nVars
        for j = i:nVars
            for k = j:nVars
                labels{ind,1} = [str_vars{i}, '*', str_vars{j}, '*', str_vars{k}];
                ind = ind + 1;
            end
        end
    end
end

% --- 标准库仑摩擦力项 ---
labels{ind,1} = 'tanh(10*dq)';
ind = ind + 1;

% 构建输出表格并即时打印非零项
fprintf('\n  辨识出的方程 (非零项):\n');
for j = 1:size(ahat, 2)
    fprintf('  --- %sdot ---\n', str_vars{j});
    count = 0;
    for k = 1:size(ahat,1)
        if abs(ahat(k,j)) > 1e-10
            if k <= length(labels)
                fprintf('    %+.6f * %s\n', ahat(k,j), labels{k});
            else
                fprintf('    %+.6f * [未知项_%d]\n', ahat(k,j), k);
            end
            count = count + 1;
        end
    end
    if count == 0
        fprintf('    (无非零项)\n');
    end
end

yout = [];
end
