Pc = imread("lab\lab1\imgs\pck-int.jpg");
caged_primate = imread("lab\lab1\imgs\primatecaged.jpg");

img = im2gray(Pc);
img_caged = im2gray(caged_primate);

NoiseSuppression(img,7)
NoiseSuppression(img_caged,10);

function NoiseSuppression(img, sizeNeighborhood)
    [rows, cols] = size(img);

    % 2D Fourier Transform
    F = fft2(double(img));
    F_shift = fftshift(F);

    % Power spectrum
    S_shift = abs(F_shift).^2;

    % Display spectrum and select multiple peaks
    figure; imagesc(S_shift.^0.1); colormap('default'); title('Select peaks, press Enter when done');
    [x_peaks, y_peaks] = ginput;  % multiple clicks, press Enter to finish
    x_peaks = round(x_peaks);
    y_peaks = round(y_peaks);
    close;

    half_size = floor(sizeNeighborhood / 2);

    % Loop through all peaks
    for k = 1:length(x_peaks)
        % Original peak
        r_range = max(y_peaks(k)-half_size,1) : min(y_peaks(k)+half_size, rows);
        c_range = max(x_peaks(k)-half_size,1) : min(x_peaks(k)+half_size, cols);
        F_shift(r_range, c_range) = 0;

        % Symmetric peak
        r_sym = rows - y_peaks(k) + 1;
        c_sym = cols - x_peaks(k) + 1;
        r_range = max(r_sym-half_size,1) : min(r_sym+half_size, rows);
        c_range = max(c_sym-half_size,1) : min(c_sym+half_size, cols);
        F_shift(r_range, c_range) = 0;
    end

    % Inverse transform
    filtered_img = uint8(real(ifft2(ifftshift(F_shift))));

    % Display results
    figure;
    subplot(1,2,1); imshow(img); title('Original');
    subplot(1,2,2); imshow(filtered_img); title('Filtered');

    % Display filtered spectrum
    figure;
    imagesc((abs(F_shift).^2).^0.1); colormap('default'); title('Masked Power Spectrum');
end
