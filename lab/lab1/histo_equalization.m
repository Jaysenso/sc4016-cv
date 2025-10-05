% Read and convert image
Pc = imread("lab\lab1\imgs\mrt-train.jpg");
P = rgb2gray(Pc);

% Apply histogram equalization
P3 = histeq(P, 255);
P4 = histeq(P3, 255);

% Histogram comparison: 10 bins vs 256 bins
figure('Name', 'Histogram Analysis');
subplot(1, 2, 1);
imhist(P, 10);
title('Original (10 bins)');

subplot(1, 2, 2);
imhist(P, 256);
title('Original (256 bins)');

% Display original and equalized images
figure('Name', 'Image Comparison');
subplot(2, 2, 1);
imshow(P);
title('Original Image');

subplot(2, 2, 2);
imshow(P3);
title('Equalized Image');

subplot(2, 2, 4);
imshow(P4);
title('2nd Equalized Image');

% Histogram comparison: 10 bins vs 256 bins
figure('Name', 'Histogram Analysis');

subplot(2, 4, 1);
imhist(P, 10);
title('Original (10 bins)');

subplot(2, 4, 2);
imhist(P, 256);
title('Original (256 bins)');

subplot(2, 4, 3);
imhist(P3, 10);
title('Equalized (10 bins)');

subplot(2, 4, 4);
imhist(P3, 256);
title('Equalized (256 bins)');

% Show second equalization effect
subplot(2, 4, 7);
imhist(P4, 10);
title('2nd Equalization (10 bins)');

subplot(2, 4, 8);
imhist(P4, 256);
title('2nd Equalization (256 bins)');
figure;
% Show second equalization effect
subplot(1, 2, 1);
imhist(P3, 256);
title('P3 Equalization (256 bins)');

subplot(1, 2, 2);
imhist(P4, 256);
title('P4 Equalization (256 bins)');
% Display statistics
fprintf('Original image - Min: %d, Max: %d\n', min(P(:)), max(P(:)));
fprintf('After 1st equalization - Min: %d, Max: %d\n', min(P3(:)), max(P3(:)));
fprintf('After 2nd equalization - Min: %d, Max: %d\n', min(P4(:)), max(P4(:)));