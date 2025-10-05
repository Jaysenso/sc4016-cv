% Training data
X = [3 3 1;
     1 1 1];

y = [1, -1];

iterations = 100;

% Run both algorithms
[w1, iter1] = perceptron_algorithm1(X, y, iterations);
[w2, iter2] = perceptron_algorithm2(X, y, iterations);

% Display results
fprintf('\nAlgorithm 1 Results:\n');
fprintf('Final weights: [%.0f, %.0f, %.0f]\n', w1);
fprintf('Iterations: %d\n', iter1);

fprintf('\nAlgorithm 2 Results:\n');
fprintf('Final weights: [%.4f, %.4f, %.4f]\n', w2);
fprintf('Iterations: %d\n', iter2);

% Visualize both algorithms
figure('Position', [100, 100, 1200, 500]);

subplot(1, 2, 1); plot_decision_boundary(X, y, w1, iter1, 'Algorithm 1 (Perceptron)', 'g');
subplot(1, 2, 2); plot_decision_boundary(X, y, w2, iter2, 'Algorithm 2 (Gradient Descent)', 'm');


function [weights, iterations] = perceptron_algorithm1(X, y, max_iter)
    [n_samples, n_features] = size(X);
    weights = zeros(n_features, 1);
    iterations = 0;
    alpha = 1;

    converged = false;
    while ~converged && iterations < max_iter
        converged = true;
        iterations = iterations + 1;

        for i = 1:n_samples
            if y(i) * (weights' * X(i,:)') <= 0
                weights = weights + alpha * y(i) * X(i,:)';
                converged = false;
            end
        end
    end
    fprintf('Algorithm 1 converged in %d iterations\n', iterations);
end

function [weights, iterations] = perceptron_algorithm2(X, y, max_iter)
    [n_samples, n_features] = size(X);
    weights = zeros(n_features, 1);
    alpha = 0.1; % learning rate
    iterations = 0;

    % Ensure y is a column vector for proper matrix operations
    y = y(:);
    
    for iter = 1:max_iter
        iterations = iter;
        total_error = 0;

        for i = 1:n_samples
            y_pred = weights' * X(i, :)';
            
            error_i = y(i) - y_pred;
            total_error = total_error + 0.5 * error_i^2;
            
            grad = -error_i * X(i, :)';
            
            % Update weights (stochastic gradient descent)
            weights = weights - alpha * grad;
        end
        
        % Stop if total error is small
        if total_error < 1e-3
            fprintf('Gradient descent converged in %d iterations\n', iterations);
            return;
        end
    end
    
    fprintf('Gradient descent reached max iterations (%d), not converged\n', iterations);
end

function plot_decision_boundary(X, y, weights, iterations, title_text, color)
    hold on;
    
    % Ensure y is a row vector for indexing
    y = y(:)';
    
    % Plot data points
    pos_idx = (y == 1);
    neg_idx = (y == -1);
    
    plot(X(pos_idx, 1), X(pos_idx, 2), 'ro', ...
        'MarkerSize', 14, 'LineWidth', 2, 'MarkerFaceColor', 'r');
    plot(X(neg_idx, 1), X(neg_idx, 2), 'bs', ...
        'MarkerSize', 14, 'LineWidth', 2, 'MarkerFaceColor', 'b');
    
    % Plot decision boundary
    x1_range = -2:0.1:5;
    
    if weights(2) ~= 0
        % Standard line: solve for x2
        x2_boundary = (-weights(1) * x1_range - weights(3)) / weights(2);
        plot(x1_range, x2_boundary, color, 'LineWidth', 3);
    elseif weights(1) ~= 0
        % Vertical line: solve for x1
        x1_boundary = -weights(3) / weights(1);
        plot([x1_boundary x1_boundary], [-2 5], color, 'LineWidth', 3);
    end
    
    xlabel('x_1', 'FontSize', 12);
    ylabel('x_2', 'FontSize', 12);
    
    if weights(2) ~= 0
        title(sprintf('%s\n%.2fx_1 + %.2fx_2 + %.2f = 0\nIterations: %d', ...
            title_text, weights(1), weights(2), weights(3), iterations), 'FontSize', 12);
    else
        title(sprintf('%s\n%.2fx_1 + %.2f = 0\nIterations: %d', ...
            title_text, weights(1), weights(3), iterations), 'FontSize', 12);
    end
    
    legend('Class -1', 'Class +1', 'Decision Boundary', 'Location', 'best');
    grid on;
    axis([-1 4 -1 4]);
    axis square;
    hold off;
end