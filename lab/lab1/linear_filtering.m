% Read and convert image
img_lib_gn = imread("lab\lab1\imgs\lib-gn.jpg");
img_lib_sp = imread("lab\lab1\imgs\lib-sp.jpg");

filter_size = 5;

% Show the images
figure;
subplot(1,2,1)
imshow(img_lib_gn);
title('img name: lib-gn');
subplot(1,2,2)
imshow(img_lib_sp);
title('img name: lib-sp')

% Demostrate Gaussian filters on the 2 diff images
CompareGaussian(img_lib_gn, filter_size)
CompareGaussian(img_lib_sp, filter_size)

% Demostrate Median Filtering on the 2 diff images
CompareMedian(img_lib_gn)
CompareMedian(img_lib_sp)

function CompareMedian(img)
    img_h1 = medfilt2(img, [3,3]);
    img_h2 = medfilt2(img, [5,5]);

    % Visualize results
    figure;
    subplot(2,2,1);
    imshow(img);
    title('Original Image');

    subplot(2,2,2)
    imshow(img_h1);
    title('Median Filtering (3x3)');

    subplot(2,2,4);
    imshow(img_h2);
    title('Median Filtering (5x5)');
end

function CompareGaussian(img, filter_size)
    % Apply Gaussian filters
    h1 = GaussianKernel(filter_size, 1.0);
    h2 = GaussianKernel(filter_size, 2.0);

    % Visualize kernels
    img_h1 = GaussianFilter(img, filter_size, 1.0);
    img_h2 = GaussianFilter(img, filter_size, 2.0);

    % Visualize gaussian filters
    figure;
    subplot(1,2,1);
    mesh(h1);
    title('Gaussian Kernel (σ=1)');

    subplot(1,2,2);
    mesh(h2);
    title('Gaussian Kernel (σ=2)');

    % Visualize results
    figure;
    subplot(2,2,1);
    imshow(img);
    title('Original Image');

    subplot(2,2,2)
    imshow(img_h1);
    title('Gaussian Filtering (σ=1)');

    subplot(2,2,4);
    imshow(img_h2);
    title('Gaussian Filtering (σ=2)');
end

function G = GaussianKernel(filter_size, sigma)
    half_size = floor(filter_size/2);
    [x,y] = meshgrid(-half_size:half_size, -half_size:half_size);

    % Gaussian Function
    G = exp(-(x.^2 + y.^2)/(2 * sigma^2));
    % Normalize so kernel sums to 1
    G = G / sum(G(:));

end

function img_out = GaussianFilter(img, filter_size, sigma)

    G = GaussianKernel(filter_size, sigma);
    img_out = conv2(double(img), G, "same");
    img_out = uint8(img_out);
end

