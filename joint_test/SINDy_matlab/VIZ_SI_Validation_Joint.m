%% 1-DOF 单电机验证结果可视化

figure('Name', 'Validation', 'Position', [100 100 1000 350]);

% Angle
subplot(1, 3, 1); hold on; box on;
plot([tA(1),tA(1)], [min([x(:,1);xA(:,1);xB(:,1)]), max([x(:,1);xA(:,1);xB(:,1)])], ':', 'Color', [0.4 0.4 0.4]);
plot([t;tA], [x(:,1);xA(:,1)], '-', 'Color', [0 0.447 0.741], 'LineWidth', 1);
plot(tB, xB(:,1), '--', 'Color', [0.850 0.325 0.098], 'LineWidth', 1.5);
xlabel('Time [s]'); ylabel('Angle q [rad]');
legend('True', ModelName, 'Location', 'best');
grid on;

% Velocity
subplot(1, 3, 2); hold on; box on;
plot([tA(1),tA(1)], [min([x(:,2);xA(:,2);xB(:,2)]), max([x(:,2);xA(:,2);xB(:,2)])], ':', 'Color', [0.4 0.4 0.4]);
plot([t;tA], [x(:,2);xA(:,2)], '-', 'Color', [0 0.447 0.741], 'LineWidth', 1);
plot(tB, xB(:,2), '--', 'Color', [0.850 0.325 0.098], 'LineWidth', 1.5);
xlabel('Time [s]'); ylabel('Velo dq [rad/s]');
grid on;

% Error
Nval = min(size(xA,1), size(xB,1));
err = xA(1:Nval,:) - xB(1:Nval,:);
subplot(1, 3, 3); hold on; box on;
plot(tB(1:Nval), err(:,1), 'Color', [0 0.447 0.741], 'LineWidth', 1);
plot(tB(1:Nval), err(:,2), 'Color', [0.850 0.325 0.098], 'LineWidth', 1);
xlabel('Time [s]'); ylabel('Error');
legend({'e_q', 'e_{dq}'}, 'Location', 'best');
grid on;

sgtitle('1-DOF Single Motor SINDy Validation', 'FontSize', 14, 'FontWeight', 'bold');
print('-dpng', '-r150', fullfile(figpath, ['EX_', SystemModel, '_Validation.png']));

fprintf('\n  验证集 RMSE:\n');
fprintf('  q_err  = %10.6f rad  (NRMSE: %.2f%%)\n', rms(err(:,1)), rms(err(:,1))/(max(xA(:,1))-min(xA(:,1)))*100);
fprintf('  dq_err = %10.6f rad/s (NRMSE: %.2f%%)\n', rms(err(:,2)), rms(err(:,2))/(max(xA(:,2))-min(xA(:,2)))*100);
