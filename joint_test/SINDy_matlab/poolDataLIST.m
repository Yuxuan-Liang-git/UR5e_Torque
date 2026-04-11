function yout = poolDataLIST(yin,ahat,nVars,polyorder,usesine)
% 修复版: 正确处理多变量 sine/cosine 标签
% 原始版本对 usesine=1 且 nVars>1 时标签数不匹配 poolData 的列数
%
% 原 bug: poolData 为每个变量生成独立的 sin/cos 列,
%         但 poolDataLIST 只生成 "sin(k*yin)" 一个标签

n = size(yin,1);

ind = 1;
% poly order 0
yout{ind,1} = '1';
ind = ind+1;

% poly order 1
for i=1:nVars
    yout(ind,1) = yin(i);
    ind = ind+1;
end

if(polyorder>=2)
    for i=1:nVars
        for j=i:nVars
            yout{ind,1} = [yin{i},yin{j}];
            ind = ind+1;
        end
    end
end

if(polyorder>=3)
    for i=1:nVars
        for j=i:nVars
            for k=j:nVars
                yout{ind,1} = [yin{i},yin{j},yin{k}];
                ind = ind+1;
            end
        end
    end
end

if(polyorder>=4)
    for i=1:nVars
        for j=i:nVars
            for k=j:nVars
                for l=k:nVars
                    yout{ind,1} = [yin{i},yin{j},yin{k},yin{l}];
                    ind = ind+1;
                end
            end
        end
    end
end

if(polyorder>=5)
    for i=1:nVars
        for j=i:nVars
            for k=j:nVars
                for l=k:nVars
                    for m=l:nVars
                        yout{ind,1} = [yin{i},yin{j},yin{k},yin{l},yin{m}];
                        ind = ind+1;
                    end
                end
            end
        end
    end
end

% ===== 修复: 为每个变量生成独立的 sin/cos 标签 =====
if(usesine)
    for k=1:10
        for i=1:nVars
            yout{ind,1} = ['sin(',num2str(k),'*',yin{i},')'];
            ind = ind+1;
        end
        for i=1:nVars
            yout{ind,1} = ['cos(',num2str(k),'*',yin{i},')'];
            ind = ind+1;
        end
    end
end


output = yout;
newout(1) = {''};
for k=1:length(yin)
    newout{1,1+k} = [yin{k},'dot'];
end

for k=1:size(ahat,1)
    if k <= length(output)
        newout(k+1,1) = output(k);
    else
        newout{k+1,1} = ['?_', num2str(k)];
    end
    for j=1:length(yin)
        newout{k+1,1+j} = ahat(k,j);
    end
end
newout
