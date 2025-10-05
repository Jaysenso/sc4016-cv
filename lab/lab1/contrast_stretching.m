

% Read the image
Pc = imread("lab\lab1\imgs\mrt-train.jpg");
P = rgb2gray(Pc);
whos Pc

% Obtain the min and max pixel intensities in the image
min_pixel = double(min(P(:))); 
max_pixel = double(max(P(:)));

% Contrast stretching formula
P2 = uint8(255 * (double(P) - min_pixel)/(max_pixel - min_pixel));

% Obtain the new min and max pixel intensities in the processed image
new_min_pixel = double(min(P2(:)));
new_max_pixel = double(max(P2(:)));

fprintf('Min: %d, Max: %d, New Min: %d, New Max: %d\n', min_pixel, max_pixel, new_min_pixel, new_max_pixel);


figure;
subplot(1,2,1);
imshow(P);
title("Original Image");

subplot(1,2,2);
imshow(P2);
title("Processed Image");
