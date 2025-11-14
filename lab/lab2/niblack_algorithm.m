% List of document images
cwd = "lab\lab2\imgs\";
doc_files = {'document01.bmp', 'document02.bmp', 'document03.bmp', 'document04.bmp'};
gt_files = {'document01-GT.tiff', 'document02-GT.tiff', 'document03-GT.tiff', 'document04-GT.tiff'};

% Store results for all documents
all_results = cell(4, 1);

% Process each document
for doc_idx = 1:4
    fprintf('\n=== Processing Document %d ===\n', doc_idx);
    
    % Load images
    img = imread(cwd + doc_files{doc_idx});
    if size(img, 3) == 3
        img = rgb2gray(img);
    end

    ground_truth = imread(cwd + gt_files{doc_idx});
    if size(ground_truth, 3) == 3
        ground_truth = rgb2gray(ground_truth);
    end
    
    % Normalize ground truth to binary (0 or 1)
    if max(ground_truth(:)) > 1
        ground_truth = ground_truth > 128;
    end
    ground_truth = double(ground_truth);
    
    % Define objective function for Bayesian optimization
    objective_func = @(params)niblack_objective_scalar(img, ground_truth, params.window_size, params.k);
    
    % Define optimization variables
    window_var = optimizableVariable('window_size', [31, 301], 'Type', 'integer');
    k_var = optimizableVariable('k', [-3.5, 3.5], 'Type', 'real');
    
    % Run Bayesian optimization
    results = bayesopt(objective_func, [window_var, k_var], ...
        'MaxObjectiveEvaluations', 250, ...
        'IsObjectiveDeterministic', true, ...
        'AcquisitionFunctionName', 'expected-improvement-plus', ...
        'ExplorationRatio', 0.4, ...
        'Verbose', 1, ...
        'PlotFcn', {@plotObjectiveModel});
    
    % Get optimal parameters
    optimal_window = results.XAtMinObjective.window_size;
    optimal_k = results.XAtMinObjective.k;
    optimal_error = results.MinObjective;
    
    % Compute final result with optimal parameters
    [final_error, binary_img, diff_img, error_percentile] = niblack_objective(img, ground_truth, optimal_window, optimal_k);
    
    % Store results
    all_results{doc_idx}.doc_idx = doc_idx;
    all_results{doc_idx}.doc_file = doc_files{doc_idx};
    all_results{doc_idx}.optimal_window = optimal_window;
    all_results{doc_idx}.optimal_k = optimal_k;
    all_results{doc_idx}.error = final_error;
    all_results{doc_idx}.error_percentile = error_percentile;
    all_results{doc_idx}.binary_img = binary_img;
    all_results{doc_idx}.diff_img = diff_img;
    all_results{doc_idx}.bayesopt_results = results;

    fprintf('\nOptimal parameters for Document %d:\n', doc_idx);
    fprintf('  Window size: %d\n', optimal_window);
    fprintf('  k value: %.4f\n', optimal_k);
    fprintf('  Error: %d pixels (%.2f%%)\n', final_error, error_percentile);
end

% Visualize results for all documents
fprintf('\n=== Summary of Optimal Parameters ===\n');
fprintf('%-8s %-12s %-12s %-15s %-15s\n', 'Doc', 'Window', 'k', 'Error', 'Error (%)');
fprintf('%s\n', repmat('-', 1, 70));

for doc_idx = 1:4
    result = all_results{doc_idx};
    fprintf('%-8d %-12d %-12.4f %-15d %-15.2f\n', ...
        result.doc_idx, result.optimal_window, result.optimal_k, ...
        result.error, result.error_percentile);
    
    % Load original images for visualization
    img = imread(cwd + doc_files{doc_idx});
    if size(img, 3) == 3
        img = rgb2gray(img);
    end

    ground_truth = imread(cwd + gt_files{doc_idx});
    if size(ground_truth, 3) == 3
        ground_truth = rgb2gray(ground_truth);
    end
    
    if max(ground_truth(:)) > 1
        ground_truth = ground_truth > 128;
    end
    
    % Create visualization figure (2x2 grid)
    figure('Name', sprintf('Document %d - Bayesian Optimization Results', doc_idx), ...
           'Position', [100, 100, 1200, 900]);
    
    % Original image
    subplot(2, 2, 1);
    imshow(img);
    title('Original Image', 'FontSize', 12, 'FontWeight', 'bold');
    
    % Optimized Niblack result
    subplot(2, 2, 2);
    imshow(result.binary_img);
    title(sprintf('Optimized Niblack\nW=%d, k=%.4f', ...
        result.optimal_window, result.optimal_k), ...
        'FontSize', 12, 'FontWeight', 'bold');
    
    % Ground truth
    subplot(2, 2, 3);
    imshow(ground_truth);
    title('Ground Truth', 'FontSize', 12, 'FontWeight', 'bold');
    
    % Error map
    subplot(2, 2, 4);
    imshow(result.diff_img, []);
    colormap(gca, 'hot');
    title(sprintf('Error Map\nError: %.2f%%', result.error_percentile), ...
        'FontSize', 12, 'FontWeight', 'bold');
    
    sgtitle(sprintf('Document %d: Bayesian Optimization Results', doc_idx), ...
            'FontSize', 14, 'FontWeight', 'bold');
end

%% Scalar objective function for Bayesian optimization (returns only error)
function error = niblack_objective_scalar(img, ground_truth, window_size, k)
    [error, ~, ~, ~] = niblack_objective(img, ground_truth, window_size, k);
end

%% Full objective function for Niblack thresholding (returns all outputs)
function [error, binary_img, diff_img, error_percentile] = niblack_objective(img, ground_truth, window_size, k)
    % Ensure window_size is odd
    if mod(window_size, 2) == 0
        window_size = window_size + 1;
    end
    
    % Compute local mean and std
    local_mean = imfilter(double(img), ones(window_size, window_size)/window_size^2, 'replicate');
    local_std = stdfilt(img, ones(window_size, window_size));
    
    % Compute Niblack threshold
    threshold_map = local_mean + k * local_std;
    binary_img = double(img) > threshold_map;
    
    % Compute error in both orientations
    diff_img_normal = abs(binary_img - ground_truth);
    error_normal = sum(diff_img_normal(:));
    
    diff_img_inverted = abs((1 - binary_img) - ground_truth);
    error_inverted = sum(diff_img_inverted(:));
    
    if error_inverted < error_normal
        error = error_inverted;
        binary_img = 1 - binary_img;
        diff_img = diff_img_inverted;
    else
        error = error_normal;
        diff_img = diff_img_normal;
    end
    
    error_percentile = 100 * error / numel(img);
end