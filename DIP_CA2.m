%% Fourier transforms of images
%% Edge image
% Create the edge image
img = [zeros(256, 128) ones(256, 128)];

% Perform FFT and shift the zero frequency to the center
imgfft = fft2(img);
imgfft_shifted = fftshift(imgfft);

% Extract real and imaginary parts and spectrum
img_real = real(imgfft_shifted);   
img_imag = imag(imgfft_shifted);
spectrum = log(1 + abs(imgfft_shifted));

% Display results
figure;
subplot(1,4,1); imshow(img, []);
subplot(1,4,2); imshow(img_real, []);
subplot(1,4,3); imshow(img_imag, []);
subplot(1,4,4); imshow(spectrum, []);

%% Box
% Create a box image
img = zeros(256, 256);
img(78:178, 78:178) = 1;

% Perform FFT and shift the zero frequency to the center
imgfft = fft2(img);
imgfft_shifted = fftshift(imgfft);

% Extract real and imaginary parts and spectrum
img_real = real(imgfft_shifted);   
img_imag = imag(imgfft_shifted);
spectrum = log(1 + abs(imgfft_shifted));

% Display results
figure;
subplot(1,4,1); imshow(img, []);
subplot(1,4,2); imshow(img_real, []);
subplot(1,4,3); imshow(img_imag, []);
subplot(1,4,4); imshow(spectrum, []);

%% 45˚ rotated box
% Create a 45 degree box image
[x, y] = meshgrid(1:256, 1:256);
img= (x+y<329) & (x+y>182) & (x-y>-67) & (x-y<73);

% Perform FFT and shift the zero frequency to the center
imgfft = fft2(img);
imgfft_shifted = fftshift(imgfft);

% Extract real and imaginary parts and spectrum
img_real = real(imgfft_shifted);   
img_imag = imag(imgfft_shifted);
spectrum = log(1 + abs(imgfft_shifted));

% Display results
figure;
subplot(1,4,1); imshow(img, []);
subplot(1,4,2); imshow(img_real, []);
subplot(1,4,3); imshow(img_imag, []);
subplot(1,4,4); imshow(spectrum, []);


%% Circle
% Create a circle image
[x, y] = meshgrid(-128:127, -128:127);
z = sqrt(x.^2 + y.^2);
img = (z < 20);

% Perform FFT and shift the zero frequency to the center
imgfft = fft2(img);
imgfft_shifted = fftshift(imgfft);

% Extract real and imaginary parts and spectrum
img_real = real(imgfft_shifted);   
img_imag = imag(imgfft_shifted);
spectrum = log(1 + abs(imgfft_shifted));

% Display results
figure;
subplot(1,4,1); imshow(img, []);
subplot(1,4,2); imshow(img_real, []);
subplot(1,4,3); imshow(img_imag, []);
subplot(1,4,4); imshow(spectrum, []);


%% Ideal lowpass and highpass filtering
%% lowpass filtering
% Read the cameraman image
img = imread('cameraman.tif');

% Compute the Fourier Transform of the image
imgfft = fft2(double(img));
imgfft_shifted = fftshift(imgfft);

% Compute the spectrum for visualization
spectrum = log(1 + abs(imgfft_shifted));

% Define the meshgrid for circular masks
[rows, cols] = size(img);
[x, y] = meshgrid(-cols/2:cols/2-1, -rows/2:rows/2-1);
z = sqrt(x.^2 + y.^2);

% Lowpass filter with cutoff frequency of 5
cutoff_1 = (z < 5);
imglp_1 = imgfft_shifted .* cutoff_1;

% Spectrum after lowpass filtering with cutoff 5
spectrum_lp_1 = log(1 + abs(imglp_1));

% Inverse Fourier Transform to convert back to spatial domain
imglp_1_spatial = ifft2(ifftshift(imglp_1));
imglp_1_spatial = abs(imglp_1_spatial);

% Cutoff frequency of 30
cutoff_2 = (z < 30);
imglp_2 = imgfft_shifted .* cutoff_2;

% Spectrum after lowpass filtering with cutoff 30
spectrum_lp_2 = log(1 + abs(imglp_2));

% Inverse Fourier Transform for cutoff frequency 30
imglp_2_spatial = ifft2(ifftshift(imglp_2));
imglp_2_spatial = abs(imglp_2_spatial);

% Display results
figure;
subplot(2,4,1); imshow(img, []);
subplot(2,4,2); imshow(spectrum, []);
subplot(2,4,3); imshow(spectrum_lp_1, []);
subplot(2,4,4); imshow(imglp_1_spatial, []);

subplot(2,4,5); imshow(img, []);
subplot(2,4,6); imshow(spectrum, []);
subplot(2,4,7); imshow(spectrum_lp_2, []);
subplot(2,4,8); imshow(imglp_2_spatial, []);

%% Highpass filtering
% Read the cameraman image
img = imread('cameraman.tif');

% Compute the Fourier Transform of the image
imgfft = fft2(double(img));
imgfft_shifted = fftshift(imgfft);

% Compute the spectrum for visualization
spectrum = log(1 + abs(imgfft_shifted));

% Define the meshgrid for circular masks
[rows, cols] = size(img);
[x, y] = meshgrid(-cols/2:cols/2-1, -rows/2:rows/2-1);
z = sqrt(x.^2 + y.^2);

% Lowpass filter with cutoff frequency of 5
cutoff_1 = (z >= 5);
imghp_1 = imgfft_shifted .* cutoff_1;

% Spectrum after lowpass filtering with cutoff 5
spectrum_hp_1 = log(1 + abs(imghp_1));

% Inverse Fourier Transform to convert back to spatial domain
imghp_1_spatial = ifft2(ifftshift(imghp_1));
imghp_1_spatial = abs(imghp_1_spatial);

% Cutoff frequency of 30
cutoff_2 = (z >= 30);
imghp_2 = imgfft_shifted .* cutoff_2;

% Spectrum after lowpass filtering with cutoff 30
spectrum_hp_2 = log(1 + abs(imghp_2));

% Inverse Fourier Transform for cutoff frequency 30
imghp_2_spatial = ifft2(ifftshift(imghp_2));
imghp_2_spatial = abs(imghp_2_spatial);

% Display results
figure;
subplot(2,4,1); imshow(img, []);
subplot(2,4,2); imshow(spectrum, []);
subplot(2,4,3); imshow(spectrum_hp_1, []);
subplot(2,4,4); imshow(imghp_1_spatial, []);
subplot(2,4,5); imshow(img, []);
subplot(2,4,6); imshow(spectrum, []);
subplot(2,4,7); imshow(spectrum_hp_2, []);
subplot(2,4,8); imshow(imghp_2_spatial, []);


%% Gaussian filtering
% Read the lena image
img = imread('lena.bmp');

% Fourier Transform of the image
imgfft = fft2(double(img));
imgfft_shifted = fftshift(imgfft);

% Compute the spectrum for visualization
spectrum = log(1 + abs(imgfft_shifted));

% Define Gaussian Lowpass filters with sigma = 10 and sigma = 30
gaussian_lp_1 = fspecial('gaussian', size(img), 10);
gaussian_lp_2 = fspecial('gaussian', size(img), 30);

% Apply Lowpass Filter with sigma = 10
imgfft_lp_1 = imgfft_shifted .* gaussian_lp_1;
spectrum_lp_1 = log(1 + abs(imgfft_lp_1));
img_lp_1_spatial = ifft2(ifftshift(imgfft_lp_1));
img_lp_1_spatial = abs(img_lp_1_spatial);

% Apply Lowpass Filter with sigma = 30
imgfft_lp_2 = imgfft_shifted .* gaussian_lp_2;
spectrum_lp_2 = log(1 + abs(imgfft_lp_2));
img_lp_2_spatial = ifft2(ifftshift(imgfft_lp_2));
img_lp_2_spatial = abs(img_lp_2_spatial);

% Define Gaussian Highpass filters by subtracting the lowpass filter from 1
gaussian_hp_1 = 1 - gaussian_lp_1;
gaussian_hp_2 = 1 - gaussian_lp_2;

% Apply Highpass Filter with sigma = 10
imgfft_hp_1 = imgfft_shifted .* gaussian_hp_1;
spectrum_hp_1 = log(1 + abs(imgfft_hp_1));
img_hp_1_spatial = ifft2(ifftshift(imgfft_hp_1));
img_hp_1_spatial = abs(img_hp_1_spatial);

% Apply Highpass Filter with sigma = 30
imgfft_hp_2 = imgfft_shifted .* gaussian_hp_2;
spectrum_hp_2 = log(1 + abs(imgfft_hp_2));
img_hp_2_spatial = ifft2(ifftshift(imgfft_hp_2));
img_hp_2_spatial = abs(img_hp_2_spatial);


% Display results
figure;
subplot(4,4,1); imshow(img, []);
subplot(4,4,2); imshow(spectrum, []);
subplot(4,4,3); imshow(spectrum_lp_1, []);
subplot(4,4,4); imshow(img_lp_1_spatial, []);

subplot(4,4,5); imshow(img, []);
subplot(4,4,6); imshow(spectrum, []);
subplot(4,4,7); imshow(spectrum_lp_2, []);
subplot(4,4,8); imshow(img_lp_2_spatial, []);

subplot(4,4,9); imshow(img, []);
subplot(4,4,10); imshow(spectrum, []);
subplot(4,4,11); imshow(spectrum_hp_1, []);
subplot(4,4,12); imshow(img_hp_1_spatial, []);

subplot(4,4,13); imshow(img, []);
subplot(4,4,14); imshow(spectrum, []);
subplot(4,4,15); imshow(spectrum_hp_2, []);
subplot(4,4,16); imshow(img_hp_2_spatial, []);
