%% Execution time comparison
% Load the image
img = imread('hurricane-katrina.tif');
img = double(img);
% Wavelet parameters
wname = 'db4';
n_levels = 10;

% Initialize arrays to store results
ratios = zeros(1, n_levels);
max_abs_differences = zeros(1, n_levels);

for n = 1:n_levels
    % Define the anonymous functions for time measurement
    w1 = @() wavedec2(img, n, wname);
    w2 = @() wavefast(img, n, wname); % Assuming wavefast is available

    % Measure computation times
    t1 = timeit(w1);
    t2 = timeit(w2);

    % Store the ratio of computation times
    ratios(n) = t2 / t1;

    % Compute the wavelet decompositions
    coeffs1 = wavedec2(img, n, wname);
    coeffs2 = wavefast(img, n, wname);

    % Calculate the maximum absolute difference between decompositions
    max_abs_differences(n) = max(abs(coeffs1(:) - coeffs2(:)));
end

% Plot the ratios of computation times
figure;
plot(1:n_levels, ratios, '-o');
grid on;
title('Ratio of Computation Times (t2 / t1)');
xlabel('Decomposition Level (n)');
ylabel('Ratio (t2 / t1)');

% Plot the maximum absolute differences
figure;
plot(1:n_levels, max_abs_differences, '-o');
grid on;
title('Maximum Absolute Differences Between w1 and w2');
xlabel('Decomposition Level (n)');
ylabel('Maximum Absolute Difference');

%% Transform coefficient display
% Load the image
img = imread('zoneplate.tif');
img = double(img);

% Compute wavelet decomposition coefficients using Haar wavelets
[cm, sm] = wavefast(img, 2, 'haar');

% Display the coefficient matrices with different scale settings

% (a) Display with the default setting
wavedisplay(cm, sm);
title('Wavelet Coefficients Display: Default Scale');

% (b) Magnify default by the scale factor (scale = 8)
scale1 = 8;
wavedisplay(cm, sm, scale1);
title(['Wavelet Coefficients Display: Scale = ', num2str(scale1)]);

% (c) Magnify absolute values by abs(scale) with scale = -8
scale2 = -8;
wavedisplay(cm, sm, scale2);
title(['Wavelet Coefficients Display: Scale = ', num2str(scale2)]);


%% Wavelet directionality and edge
% Load the image
img = imread('test_pattern.tif');
img = double(img);

% Compute 4th order Symlets wavelet transform
[cm, sm] = wavefast(img, 1, 'sym4');

% (a) Display the wavelet coefficients
wavedisplay(cm, sm, -6);
title('Wavelet Coefficients Display: Horizontal, Vertical, Diagonal Directionality');

% (b) Zero coefficients using wavecut
[nc, ym] = wavecut('a', cm, sm);
wavedisplay(nc, sm, -6);
title('Wavelet Coefficients Display After Zeroing Coefficients');

% (c) Compute inverse wavelet transform and display the image
reconstructed = waveback(nc, sm, 'sym4');
abs_reconstructed = mat2gray(abs(reconstructed));
figure;
imshow(abs_reconstructed);
title('Reconstructed Image After Zeroing Coefficients');


%% Wavelet-based image smoothing
% Load the image
img = imread('hurricane-katrina.tif');

% Initialize variables
levels = 1:4; % Levels to zero
wname = 'bior6.8';
psnr_values = zeros(1, length(levels));

% Perform smoothing for each level
for level = levels
    % Compute wavelet decomposition
    [cm, sm] = wavefast(img, 4, wname);

    % Zero the detail coefficients and reconstruct
    [nc, g8] = wavezero(cm, sm, level, wname);

    % Display the smoothed image
    figure;
    imshow(mat2gray(g8));
    title(['Smoothed Image with Level ', num2str(level)]);

    % Compute and display the difference image
    diff_img = abs(double(img) - double(g8));
    figure;
    imshow(mat2gray(diff_img));
    title(['Difference Image with Level ', num2str(level)]);

    % Compute PSNR
    psnr_values(level) = psnr(uint8(g8), img);
    disp(psnr_values(level));
end

% Add Gaussian noise to the image
noisyImg = imnoise(img, 'gaussian', 0, 0.15);

% Repeat the process with noisy image
for level = levels
    % Compute wavelet decomposition
    [cm, sm] = wavefast(noisyImg, 4, wname);

    % Zero the detail coefficients and reconstruct
    [nc, g8] = wavezero(cm, sm, level, wname);

    % Display the smoothed image
    figure;
    imshow(mat2gray(g8));
    title(['Smoothed Noisy Image with Level ', num2str(level)]);

    % Compute and display the difference image
    diff_img = abs(double(noisyImg) - double(g8));
    figure;
    imshow(mat2gray(diff_img));
    title(['Difference Noisy Image with Level ', num2str(level)]);

    % Compute PSNR
    psnr_values(level) = psnr(uint8(g8), img);
    disp(psnr_values(level));
end

% Report PSNR values
[best_psnr, best_level] = max(psnr_values);
disp(['Best level for visual quality: ', num2str(best_level)]);
disp(['PSNR at best level: ', num2str(best_psnr)]);
