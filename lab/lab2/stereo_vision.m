cwd = "lab\lab2\imgs\";

% Load and convert images to gray scale
left_corridor = imread(cwd + "corridorl.jpg");
right_corridor = imread(cwd + "corridorr.jpg");
ground_truth_corridor = imread(cwd + "corridor_disp.jpg");
analyze_disparity(left_corridor, right_corridor, ground_truth_corridor, ...
                 'lab\lab2\results\3d_stereo_vision\partc_corridor_disparity_result.png', ...
                 'Corridor');

left_triclops = imread(cwd + "triclopsi2l.jpg");
right_triclops = imread(cwd + "triclopsi2r.jpg");
ground_truth_triclops = imread(cwd + "triclopsid.jpg");
analyze_disparity(left_triclops, right_triclops,ground_truth_triclops, ...
                 'lab\lab2\results\3d_stereo_vision\partd_triclops_disparity_result.png', ...
                 'Triclopsi');

%% Disparity Loop
function analyze_disparity(left_img, right_img, ground_truth, output_filename, image_name)
    % Analyze and visualize disparity map for stereo image pair
    %
    % Parameters:
    %   left_img: Left image (can be RGB or grayscale)
    %   right_img: Right image (can be RGB or grayscale)
    %   ground_truth_path: Ground Truth image (can be RGB or grayscale)
    %   output_filename: Path to save the disparity result image
    %   image_name: Name for display in titles (e.g., 'Corridor', 'Triclops')
    
    % Set default parameters
    template_height = 11;
    template_width = 11;
    max_disparity = 15;
    
    % Convert to grayscale if needed
    if size(left_img, 3) == 3
        left_gray = rgb2gray(left_img);
    else
        left_gray = left_img;
    end
    
    if size(right_img, 3) == 3
        right_gray = rgb2gray(right_img);
    else
        right_gray = right_img;
    end
    % Load ground truth
    if size(ground_truth, 3) == 3
        ground_truth = rgb2gray(ground_truth);
    end
    
    % Compute disparity map
    fprintf('Computing disparity map for %s images\n', image_name);
    D = compute_disparity_map_fft(left_gray, right_gray, template_height, template_width, max_disparity);
    
    % Visualize results in 2x2 layout
    fig = figure('Position', [100, 100, 1200, 600]);
    
    % Top row: Input images and computed disparity
    subplot(2, 2, 1);
    imshow(left_gray);
    title(sprintf('%s - Left Image', image_name));
    
    subplot(2, 2, 2);
    imshow(right_gray);
    title(sprintf('%s - Right Image', image_name));

    subplot(2, 2, 3);
    imshow(ground_truth);
    title(sprintf('%s - Ground Truth Disparity', image_name));
    
    subplot(2, 2, 4);
    imshow(D, [0 15]);
    title(sprintf('%s - Computed Disparity', image_name));
    colorbar;
    
    % Save the result
    saveas(fig, output_filename);
    fprintf('Saved figure to: %s\n', output_filename);
    
    % Compute mean/max/min/std
    fprintf('\n--- Disparity Quality Analysis for %s ---\n', image_name);
    fprintf('Mean disparity: %.2f\n', mean(D(:)));
    fprintf('Max disparity: %.2f\n', max(D(:)));
    fprintf('Min disparity: %.2f\n', min(D(:)));
    fprintf('Std deviation: %.2f\n', std(D(:)));
    fprintf('\n');

end

%% Disparity map algorithm implementation using FFT 
function disparity_map = compute_disparity_map_fft(left_img, right_img, template_height, template_width, max_disparity)

    left = double(left_img);
    right = double(right_img);
    
    [h, w] = size(left);
    disparity_map = zeros(h, w);
    
    % Create template window
    window = ones(template_height, template_width);
    
    % Pre-compute using FFT-based convolution
    left_sq = left .^ 2;
    right_sq = right .^ 2;
    
    % Use fft2 for faster convolution
    term2 = ifft2(fft2(left_sq) .* fft2(window, h, w));
    term2 = real(term2);
    
    min_ssd = inf(h, w);
    
    for d = 0:max_disparity
        if d == 0
            shifted_right = right;
            shifted_right_sq = right_sq;
        else
            shifted_right = [zeros(h, d), right(:, 1:(w-d))];
            shifted_right_sq = [zeros(h, d), right_sq(:, 1:(w-d))];
        end
        
        % FFT-based convolutions
        term1 = ifft2(fft2(shifted_right_sq) .* fft2(window, h, w));
        term1 = real(term1);
        
        cross_corr = ifft2(fft2(left .* shifted_right) .* fft2(window, h, w));
        term3 = -2 * real(cross_corr);
        
        ssd = term1 + term2 + term3;
        
        mask = ssd < min_ssd;
        min_ssd(mask) = ssd(mask);
        disparity_map(mask) = d;
    end
end


