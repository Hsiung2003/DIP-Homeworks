%% Noise models
% Read the images
pepper = imread('peppers_gray.tif');
bridge = imread('walkbridge.tif');
woman = imread('woman_blonde.tif');

% Add Gaussian noise 
noisy_pepper = imnoise(pepper, 'gaussian');
noisy_bridge = imnoise(bridge, 'gaussian');
noisy_woman = imnoise(woman, 'gaussian');

% Display the images
figure;

subplot(3, 2, 1);
imshow(pepper);
title('Original Pepper image');
subplot(3, 2, 2);
imshow(noisy_pepper);
title('Pepper image with noise');
subplot(3, 2, 3);
imshow(bridge);
title('Original Walkbridge image');
subplot(3, 2, 4);
imshow(noisy_bridge);
title('Walkbridge image with noise');
subplot(3, 2, 5);
imshow(woman);
title('Original Woman image');
subplot(3, 2, 6);
imshow(noisy_woman);
title('Woman image with noise');
%%

% Add salt and pepper noise 
noisy_pepper = imnoise(pepper, 'salt & pepper');
noisy_bridge = imnoise(bridge, 'salt & pepper');
noisy_woman = imnoise(woman, 'salt & pepper');

% Display the images
figure;

subplot(3, 2, 1);
imshow(pepper);
title('Original Pepper image');
subplot(3, 2, 2);
imshow(noisy_pepper);
title('Pepper image with noise');
subplot(3, 2, 3);
imshow(bridge);
title('Original Walkbridge image');
subplot(3, 2, 4);
imshow(noisy_bridge);
title('Walkbridge image with noise');
subplot(3, 2, 5);
imshow(woman);
title('Original Woman image');
subplot(3, 2, 6);
imshow(noisy_woman);
title('Woman image with noise');
% Add Gaussian noise 
noisy_pepper = imnoise(pepper, 'speckle');
noisy_bridge = imnoise(bridge, 'speckle');
noisy_woman = imnoise(woman, 'speckle');

% Display the images
figure
subplot(3, 2, 1);
imshow(pepper);
title('Original Pepper image');
subplot(3, 2, 2);
imshow(noisy_pepper);
title('Pepper image with noise');
subplot(3, 2, 3);
imshow(bridge);
title('Original Walkbridge image');
subplot(3, 2, 4);
imshow(noisy_bridge);
title('Walkbridge image with noise');
subplot(3, 2, 5);
imshow(woman);
title('Original Woman image');
subplot(3, 2, 6);
imshow(noisy_woman);
title('Woman image with noise');

%% Gaussian noise reduction
% Read the image
original_image = imread('peppers_gray.tif');

% Add Gaussian noise with variances 0.05 and 0.2
noisy_image_005 = imnoise(original_image, 'gaussian', 0, 0.05);
noisy_image_02 = imnoise(original_image, 'gaussian', 0, 0.2);


% Function to calculate PSNR
calculate_psnr = @(original, restored) 10 * log10(255^2 / mean((double(original(:)) - double(restored(:))).^2));

% Arithmetic mean filter
h_avg = fspecial('average', [7, 7]);
restored_avg_005 = filter2(h_avg, noisy_image_005, 'same');
restored_avg_005_norm = mat2gray(restored_avg_005);

restored_avg_02 = filter2(h_avg, noisy_image_02, 'same');
restored_avg_02_norm = mat2gray(restored_avg_02);

psnr_avg_005 = calculate_psnr(original_image, restored_avg_005);
psnr_avg_02 = calculate_psnr(original_image, restored_avg_02);

% Gaussian lowpass filter
h_gauss = fspecial('gaussian', [9,9], 1);
restored_gauss_005 = imfilter(noisy_image_005, h_gauss, 'replicate');
restored_gauss_02 = imfilter(noisy_image_02, h_gauss, 'replicate');
psnr_gauss_005 = calculate_psnr(original_image, restored_gauss_005);
psnr_gauss_02 = calculate_psnr(original_image, restored_gauss_02);

% Median filter
restored_median_005 = medfilt2(noisy_image_005, [7 7]);
restored_median_02 = medfilt2(noisy_image_02, [7 7]);
psnr_median_005 = calculate_psnr(original_image, restored_median_005);
psnr_median_02 = calculate_psnr(original_image, restored_median_02);

% Wiener filter
restored_wiener_005 = wiener2(noisy_image_005, [5,5]);
restored_wiener_02 = wiener2(noisy_image_02, [5,5]);
psnr_wiener_005 = calculate_psnr(original_image, restored_wiener_005);
psnr_wiener_02 = calculate_psnr(original_image, restored_wiener_02);

% Alpha-trimmed mean filter
function output = alphatrim(input_image, kernel_size, d)   
    [rows, cols] = size(input_image);
    pad_size = floor(kernel_size / 2);
    padded_image = padarray(input_image, [pad_size, pad_size], 'replicate');
    output = zeros(size(input_image), 'like', input_image);
    
    for i = 1:rows
        for j = 1:cols
            % Extract the kernel window
            window = padded_image(i:i+kernel_size-1, j:j+kernel_size-1);
            % Flatten the window and sort the values
            sorted_values = sort(window(:));
            % Trim d/2 lowest and d/2 highest values
            trimmed_values = sorted_values(floor(d/2)+1:end-floor(d/2));
            % Compute the mean of the remaining values
            output(i, j) = mean(trimmed_values);
        end
    end
end

% Alpha-trimmed mean filter (user-defined function)
restored_alpha_005 = alphatrim(noisy_image_005, 7, 2); 
restored_alpha_02 = alphatrim(noisy_image_02, 7, 2); 
psnr_alpha_005 = calculate_psnr(original_image, restored_alpha_005);
psnr_alpha_02 = calculate_psnr(original_image, restored_alpha_02);

% display result
figure;
subplot(6, 2, 1);
imshow(noisy_image_005);
title('Noisy image 005');
subplot(6, 2, 2);
imshow(noisy_image_02);
title('Noisy image 02');
subplot(6, 2, 3);
imshow(restored_avg_005_norm);
title('Arithmetic mean restored 005');
subplot(6, 2, 4);
imshow(restored_avg_02_norm);
title('Arithmetic mean restored 02');
subplot(6, 2, 5);
imshow(restored_gauss_005);
title('Gaussian lowpass restored 005');
subplot(6, 2, 6);
imshow(restored_gauss_02);
title('Gaussian lowpass restored 02');
subplot(6, 2, 7);
imshow(restored_median_005);
title('Median restored 005');
subplot(6, 2, 8);
imshow(restored_median_005);
title('Median restored 005');
subplot(6, 2, 9);
imshow(restored_gauss_005);
title('Wiener restored 005');
subplot(6, 2, 10);
imshow(restored_gauss_02);
title('Wiener restored 02');
subplot(6, 2, 11);
imshow(restored_alpha_005);
title('Alpha restored 005');
subplot(6, 2, 12);
imshow(restored_alpha_02);
title('Alpha restored 02');

disp(['Arithmetic 005 PSNR: ', num2str(psnr_avg_005)]);
disp(['Arithmetic 02 PSNR: ', num2str(psnr_avg_02)]);
disp(['Gaussian 005 PSNR: ', num2str(psnr_gauss_005)]);
disp(['Gaussian 02 PSNR: ', num2str(psnr_gauss_02)]);
disp(['Median 005 PSNR: ', num2str(psnr_median_005)]);
disp(['Median 02 PSNR: ', num2str(psnr_median_02)]);
disp(['Wiener 005 PSNR: ', num2str(psnr_wiener_005)]);
disp(['Wiener 02 PSNR: ', num2str(psnr_wiener_02)]);
disp(['Alpha 005 PSNR: ', num2str(psnr_alpha_005)]);
disp(['Alpha 02 PSNR: ', num2str(psnr_alpha_02)]);


%% Salt-and-pepper noise reduction
% Read the image
image = imread('woman_blonde.tif');

% Add salt-and-pepper noise
noisy_image_01 = imnoise(image, 'salt & pepper', 0.1);
noisy_image_04 = imnoise(image, 'salt & pepper', 0.4);

calculate_psnr = @(original, restored) 10 * log10(255^2 / mean((double(original(:)) - double(restored(:))).^2));

% Alpha trimmed filter
function output = alphatrim(input_image, kernel_size, d)
    [rows, cols] = size(input_image);
    pad_size = floor(kernel_size / 2);
    padded_image = padarray(input_image, [pad_size, pad_size], 'replicate');
    output = zeros(size(input_image), 'like', input_image);
    
    for i = 1:rows
        for j = 1:cols
            window = padded_image(i:i+kernel_size-1, j:j+kernel_size-1);
            sorted_values = sort(window(:));
            trimmed_values = sorted_values(floor(d/2)+1:end-floor(d/2));
            output(i, j) = mean(trimmed_values);
        end
    end
end

% Midpoint filter
function output = midpoint(input_image, kernel_size)
    [rows, cols] = size(input_image);
    pad_size = floor(kernel_size / 2);
    padded_image = padarray(input_image, [pad_size, pad_size], 'replicate');
    output = zeros(size(input_image), 'like', input_image);
    
    for i = 1:rows
        for j = 1:cols
            window = padded_image(i:i+kernel_size-1, j:j+kernel_size-1);
            min_val = min(window(:));
            max_val = max(window(:));
            output(i, j) = (min_val + max_val) / 2;
        end
    end
end

% Outlier filter
function output = outlier(input_image, kernel_size, D)
    [rows, cols] = size(input_image);
    pad_size = floor(kernel_size / 2);
    padded_image = padarray(input_image, [pad_size, pad_size], 'replicate');
    output = zeros(size(input_image), 'like', input_image);
    
    for i = 1:rows
        for j = 1:cols
            window = padded_image(i:i+kernel_size-1, j:j+kernel_size-1);
            mean_val = mean(window(:));
            if abs(double(input_image(i, j)) - mean_val) > D
                output(i, j) = mean_val;
            else
                output(i, j) = input_image(i, j);
            end
        end
    end
end

% Adaptive median filter
function output = adpmedian(input_image, max_kernel_size)
    [rows, cols] = size(input_image);
    output = input_image;
    for i = 1:rows
        for j = 1:cols
            kernel_size = 3;
            while kernel_size <= max_kernel_size
                pad_size = floor(kernel_size / 2);
                padded_image = padarray(input_image, [pad_size, pad_size], 'replicate');
                window = padded_image(i:i+kernel_size-1, j:j+kernel_size-1);
                min_val = min(window(:));
                max_val = max(window(:));
                med_val = median(window(:));
                if med_val > min_val && med_val < max_val
                    if input_image(i, j) > min_val && input_image(i, j) < max_val
                        output(i, j) = input_image(i, j);
                    else
                        output(i, j) = med_val;
                    end
                    break;
                else
                    kernel_size = kernel_size + 2;
                end
            end
        end
    end
end


% Median Filter
median_filtered_01 = medfilt2(noisy_image_01, [5, 5]);
median_filtered_04 = medfilt2(noisy_image_04, [5, 5]);
psnr_median_01 = calculate_psnr(image, median_filtered_01);
psnr_median_04 = calculate_psnr(image, median_filtered_04);

% Alpha-Trimmed Mean Filter
alpha_trimmed_01 = alphatrim(noisy_image_01, 5, 2);
alpha_trimmed_04 = alphatrim(noisy_image_04, 5, 2);
psnr_alpha_01 = calculate_psnr(image, alpha_trimmed_01);
psnr_alpha_04 = calculate_psnr(image, alpha_trimmed_04);

% Midpoint Filter
midpoint_filtered_01 = midpoint(noisy_image_01, 7);
midpoint_filtered_04 = midpoint(noisy_image_04, 7);
psnr_midpoint_01 = calculate_psnr(image, midpoint_filtered_01);
psnr_midpoint_04 = calculate_psnr(image, midpoint_filtered_04);

% Outlier Filter
outlier_filtered_01 = outlier(noisy_image_01, 5, 20);
outlier_filtered_04 = outlier(noisy_image_04, 5, 20);
psnr_outlier_01 = calculate_psnr(image, outlier_filtered_01);
psnr_outlier_04 = calculate_psnr(image, outlier_filtered_04);

% Adaptive Median Filter
adp_median_filtered_01 = adpmedian(noisy_image_01, 5);
adp_median_filtered_04 = adpmedian(noisy_image_04, 5);
psnr_adp_01 = calculate_psnr(image, adp_median_filtered_01);
psnr_adp_04 = calculate_psnr(image, adp_median_filtered_04);

% Display results
figure;
subplot(2, 6, 1), imshow(noisy_image_01), title('Noisy (Density=0.1)');
subplot(2, 6, 2), imshow(median_filtered_01), title('Median');
subplot(2, 6, 3), imshow(alpha_trimmed_01), title('Alpha-Trimmed');
subplot(2, 6, 4), imshow(midpoint_filtered_01), title('Midpoint');
subplot(2, 6, 5), imshow(outlier_filtered_01), title('Outlier');
subplot(2, 6, 6), imshow(adp_median_filtered_01), title('Adaptive Median');

subplot(2, 6, 7), imshow(noisy_image_04), title('Noisy (Density=0.4)');
subplot(2, 6, 8), imshow(median_filtered_04), title('Median');
subplot(2, 6, 9), imshow(alpha_trimmed_04), title('Alpha-Trimmed');
subplot(2, 6, 10), imshow(midpoint_filtered_04), title('Midpoint');
subplot(2, 6, 11), imshow(outlier_filtered_04), title('Outlier');
subplot(2, 6, 12), imshow(adp_median_filtered_04), title('Adaptive Median');

disp(['Median 01 PSNR: ', num2str(psnr_median_01)]);
disp(['Median 04 PSNR: ', num2str(psnr_median_04)]);
disp(['Alpha trimmed 01 PSNR: ', num2str(psnr_alpha_01)]);
disp(['Alpha trimmed 04 PSNR: ', num2str(psnr_alpha_04)]);
disp(['Midpoint 01 PSNR: ', num2str(psnr_midpoint_01)]);
disp(['Midpoint 04 PSNR: ', num2str(psnr_midpoint_04)]);
disp(['Outlier 01 PSNR: ', num2str(psnr_outlier_01)]);
disp(['Outlier 04 PSNR: ', num2str(psnr_outlier_04)]);
disp(['Adp median 01 PSNR: ', num2str(psnr_adp_01)]);
disp(['Adp median 04 PSNR: ', num2str(psnr_adp_04)]);


%% Speckle noise reduction
% Read the image
image = imread('walkbridge.tif');

% Add speckle noise
noisy_image_01 = imnoise(image, 'speckle', 0.1); % Variance = 0.1
noisy_image_03 = imnoise(image, 'speckle', 0.3); % Variance = 0.3

calculate_psnr = @(original, restored) 10 * log10(255^2 / mean((double(original(:)) - double(restored(:))).^2));

% Alpha-trimmed mean filter
function output = alphatrim(input_image, kernel_size, d)
    [rows, cols] = size(input_image);
    pad_size = floor(kernel_size / 2);
    padded_image = padarray(input_image, [pad_size, pad_size], 'replicate');
    output = zeros(size(input_image), 'like', input_image);
    
    for i = 1:rows
        for j = 1:cols
            window = padded_image(i:i+kernel_size-1, j:j+kernel_size-1);
            sorted_values = sort(window(:));
            trimmed_values = sorted_values(floor(d/2)+1:end-floor(d/2));
            output(i, j) = mean(trimmed_values);
        end
    end
end

% Midpoint filter
function output = midpoint(input_image, kernel_size)
    [rows, cols] = size(input_image);
    pad_size = floor(kernel_size / 2);
    padded_image = padarray(input_image, [pad_size, pad_size], 'replicate');
    output = zeros(size(input_image), 'like', input_image);
    
    for i = 1:rows
        for j = 1:cols
            window = padded_image(i:i+kernel_size-1, j:j+kernel_size-1);
            min_val = min(window(:));
            max_val = max(window(:));
            output(i, j) = (min_val + max_val) / 2;
        end
    end
end

% Outlier filter
function output = outlier(input_image, kernel_size, D)
    [rows, cols] = size(input_image);
    pad_size = floor(kernel_size / 2);
    padded_image = padarray(input_image, [pad_size, pad_size], 'replicate');
    output = zeros(size(input_image), 'like', input_image);
    
    for i = 1:rows
        for j = 1:cols
            window = padded_image(i:i+kernel_size-1, j:j+kernel_size-1);
            mean_val = mean(window(:));
            if abs(double(input_image(i, j)) - mean_val) > D
                output(i, j) = mean_val;
            else
                output(i, j) = input_image(i, j);
            end
        end
    end
end

% Contraharmonic mean filter
function output = chmean(input_image, kernel_size, Q)
    % Convert the input image to double for calculations
    input_image = double(input_image);
    [rows, cols] = size(input_image);
    pad_size = floor(kernel_size / 2);
    padded_image = padarray(input_image, [pad_size, pad_size], 'replicate');
    output = zeros(size(input_image));
    
    for i = 1:rows
        for j = 1:cols
            % Extract the kernel window
            window = padded_image(i:i+kernel_size-1, j:j+kernel_size-1);
            % Calculate numerator and denominator
            numerator = sum(window(:).^(Q+1));
            denominator = sum(window(:).^Q);
            % Avoid division by zero
            if denominator == 0
                output(i, j) = input_image(i, j); % Keep original value if division by zero
            else
                output(i, j) = numerator / denominator;
            end
        end
    end
    
    % Convert output back to original image class
    output = uint8(output);
end

% Arithmetic Mean Filter
arithmetic_mean_01 = imfilter(noisy_image_01, fspecial('average', 3), 'replicate');
arithmetic_mean_03 = imfilter(noisy_image_03, fspecial('average', 3), 'replicate');
psnr_arithmetic_mean_01 = calculate_psnr(image, arithmetic_mean_01);
psnr_arithmetic_mean_03 = calculate_psnr(image, arithmetic_mean_03);

% Gaussian Lowpass Filter
gaussian_filter_01 = imfilter(noisy_image_01, fspecial('gaussian', 5, 1), 'replicate');
gaussian_filter_03 = imfilter(noisy_image_03, fspecial('gaussian', 5, 1), 'replicate');
psnr_gauss_01 = calculate_psnr(image, gaussian_filter_01);
psnr_gauss_03 = calculate_psnr(image, gaussian_filter_03);

% Wiener Filter
wiener_filter_01 = wiener2(noisy_image_01, [5, 5]);
wiener_filter_03 = wiener2(noisy_image_03, [5, 5]);
psnr_wiener_01 = calculate_psnr(image, wiener_filter_01);
psnr_wiener_03 = calculate_psnr(image, wiener_filter_03);

% Outlier Filter
outlier_filter_01 = outlier(noisy_image_01, 5, 20);
outlier_filter_03 = outlier(noisy_image_03, 7, 20);
psnr_outlier_01 = calculate_psnr(image, outlier_filter_01);
psnr_outlier_03 = calculate_psnr(image, outlier_filter_03);

% Alpha-Trimmed Mean Filter
alpha_trimmed_01 = alphatrim(noisy_image_01, 5, 2); 
alpha_trimmed_03 = alphatrim(noisy_image_03, 5, 2);
psnr_alpha_trimmed_01 = calculate_psnr(image, alpha_trimmed_01);
psnr_alpha_trimmed_03 = calculate_psnr(image, alpha_trimmed_03);

% Midpoint Filter
midpoint_filter_01 = midpoint(noisy_image_01, 5);
midpoint_filter_03 = midpoint(noisy_image_03, 5);
psnr_midpoint_01 = calculate_psnr(image, midpoint_filter_01);
psnr_midpoint_03 = calculate_psnr(image, midpoint_filter_03);

% Contraharmonic Mean Filter
contraharmonic_filter_01 = chmean(noisy_image_01, 3, 1.5); 
contraharmonic_filter_03 = chmean(noisy_image_03, 3, 1.5);
psnr_contraharmonic_01 = calculate_psnr(image, contraharmonic_filter_01);
psnr_contraharmonic_03 = calculate_psnr(image, contraharmonic_filter_03);

% Display results
figure;
subplot(2, 8, 1), imshow(noisy_image_01), title('Noisy (Density=0.1)');
subplot(2, 8, 2), imshow(arithmetic_mean_01), title('Arithmetic mean');
subplot(2, 8, 3), imshow(gaussian_filter_01), title('Gaussian lowpass');
subplot(2, 8, 4), imshow(wiener_filter_01), title('Wiener');
subplot(2, 8, 5), imshow(outlier_filter_01), title('Outlier');
subplot(2, 8, 6), imshow(alpha_trimmed_01), title('Alpha-trimmed mean');
subplot(2, 8, 7), imshow(midpoint_filter_01), title('Midpoint');
subplot(2, 8, 8), imshow(contraharmonic_filter_01), title('Contraharmonic mean');

subplot(2, 8, 9), imshow(noisy_image_03), title('Noisy (Density=0.3)');
subplot(2, 8, 10), imshow(arithmetic_mean_03), title('Arithmetic mean');
subplot(2, 8, 11), imshow(gaussian_filter_03), title('Gaussian lowpass');
subplot(2, 8, 12), imshow(wiener_filter_03), title('Wiener');
subplot(2, 8, 13), imshow(outlier_filter_03), title('Outlier');
subplot(2, 8, 14), imshow(alpha_trimmed_03), title('Alpha-trimmed mean');
subplot(2, 8, 15), imshow(midpoint_filter_03), title('Midpoint');
subplot(2, 8, 16), imshow(contraharmonic_filter_03), title('Contraharmonic mean');

disp(['Arithmetic mean 01 PSNR: ', num2str(psnr_arithmetic_mean_01)]);
disp(['Arithmetic mean 03 PSNR: ', num2str(psnr_arithmetic_mean_03)]);
disp(['Gaussian 01 PSNR: ', num2str(psnr_gauss_01)]);
disp(['Gaussian 03 PSNR: ', num2str(psnr_gauss_03)]);
disp(['Wiener 01 PSNR: ', num2str(psnr_wiener_01)]);
disp(['Wiener 03 PSNR: ', num2str(psnr_wiener_03)]);
disp(['Outlier 01 PSNR: ', num2str(psnr_outlier_01)]);
disp(['Outlier 03 PSNR: ', num2str(psnr_outlier_03)]);
disp(['ALpha-trimmed 01 PSNR: ', num2str(psnr_alpha_trimmed_01)]);
disp(['Alpha-trimmed 03 PSNR: ', num2str(psnr_alpha_trimmed_03)]);
disp(['Midpoint 01 PSNR: ', num2str(psnr_midpoint_01)]);
disp(['Midpoint 03 PSNR: ', num2str(psnr_midpoint_03)]);
disp(['Contraharmonic 01 PSNR: ', num2str(psnr_contraharmonic_01)]);
disp(['Contraharmonic 03 PSNR: ', num2str(psnr_contraharmonic_03)]);
