% List of document images
cwd = "lab\lab2\imgs\";
doc_files = {'document01.bmp', 'document02.bmp', 'document03.bmp', 'document04.bmp'};
gt_files = {'document01-GT.tiff', 'document02-GT.tiff', 'document03-GT.tiff', 'document04-GT.tiff'};

% Store results
errors = zeros(1, 4);

for i = 1:4
    % Read document image
    img = imread(cwd + doc_files{i});
    if size(img,3) == 3
        img = rgb2gray(img);
    end

    % Apply Otsu’s thresholding
    threshold = graythresh(img);
    binary_img = imbinarize(img, threshold);

    % Read ground truth
    ground_truth = imread(cwd + gt_files{i});
    if size(ground_truth,3) == 3
        ground_truth = rgb2gray(ground_truth);
    end
    if max(ground_truth(:)) > 1
        ground_truth = ground_truth > 128;
    end

    % Compute difference
    diff_img = abs(double(binary_img) - double(ground_truth));
    error = sum(diff_img(:));
    errors(i) = error;

    % Display all results for one document
    figure('Name', sprintf('Document %d Results', i), 'Position', [100, 100, 1200, 800]);
    set(gcf, 'WindowState', 'maximized');

    subplot(2,2,1);
    imshow(img);
    title(sprintf('Doc %d: Original', i));

    subplot(2,2,2);
    imshow(binary_img);
    title(sprintf('Otsu Result (T=%.3f)', threshold));

    subplot(2,2,3);
    imshow(ground_truth);
    title('Ground Truth');

    subplot(2,2,4);
    imshow(diff_img, []);
    title(sprintf('Error Map (Err=%d, %.2f%%)', error, 100*error/numel(img)));

    fprintf('Document %d:\n', i);
    fprintf('  Otsu threshold: %.4f\n', threshold);
    fprintf('  Segmentation error: %d pixels\n', error);
    fprintf('  Error percentage: %.2f%%\n\n', 100*error/numel(img));
end
