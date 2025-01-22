%% Spatial resolution
% Read the image 'lena.bmp' into the variable x
x = imread('lena.bmp');

% Resize the image by reducing it to 1/4 of its original size and then enlarging it by 4 times
% This simulates the effect of reducing and then restoring the image size
lena1 = imresize(imresize(x, 1/4), 4);

% Resize the image by reducing it to 1/8 of its original size and then enlarging it by 8 times
lena2 = imresize(imresize(x, 1/8), 8);

% Resize the image by reducing it to 1/16 of its original size and then enlarging it by 16 times
lena3 = imresize(imresize(x, 1/16), 16);

% Resize the image by reducing it to 1/32 of its original size and then enlarging it by 32 times
lena4 = imresize(imresize(x, 1/32), 32);

% Resize the image to 1/8 of its original size using nearest-neighbor interpolation
lena_nearest = imresize(x, 1/8, 'nearest');

% Resize the image to 1/8 of its original size using bilinear interpolation
lena_bilinear = imresize(x, 1/8, 'bilinear');

% Resize the image to 1/8 of its original size using bicubic interpolation
lena_bicubic = imresize(x, 1/8, 'bicubic');

% Display the resized images
figure, imshow(lena1);    
figure, imshow(lena2);    
figure, imshow(lena3);    
figure, imshow(lena4);    
figure, imshow(lena_nearest);   
figure, imshow(lena_bilinear);  
figure, imshow(lena_bicubic);   

%% Bit planes
% Read the image 'cameraman.tif' into the variable x
x = imread('cameraman.tif');

% Convert the image from integer type to double for arithmetic operations
xd = double(x);

% Isolate the least significant bit (LSB) plane (bit 0) using the mod function
c0 = mod(xd, 2);
c1 = mod(floor(xd/2), 2);
c2 = mod(floor(xd/4), 2);
c3 = mod(floor(xd/8), 2);
c4 = mod(floor(xd/16), 2);
c5 = mod(floor(xd/32), 2);
c6 = mod(floor(xd/64), 2);
c7 = mod(floor(xd/128), 2);

% Display each bit plane as an individual binary image
figure, imshow(c0);  
figure, imshow(c1);  
figure, imshow(c2);  
figure, imshow(c3);  
figure, imshow(c4);  
figure, imshow(c5);  
figure, imshow(c6);  
figure, imshow(c7);  

% Create a binary image 'ct' where pixels are 1 if the original intensity is greater than 127
ct = x > 127;

% Compare bit plane c7 with binary image ct, check if all elements are the same
all(c7(:) == ct(:)); 

% Reconstruct the image using only the 7th bit plane (MSB)
r1 = 2^7 * c7;

% Calculate the absolute difference between the original image and the reconstructed image using only the MSB
diff1 = abs(xd - r1);

% Reconstruct the image using the 7th and 6th bit planes
r2 = 2^7 * c7 + 2^6 * c6;

% Calculate the absolute difference between the original image and the reconstructed image using the 7th and 6th bit planes
diff2 = abs(xd - r2);

% Reconstruct the image using the 7th, 6th, 5th, and 4th bit planes
r3 = 2^7 * c7 + 2^6 * c6 + 2^5 * c5 + 2^4 * c4;

% Calculate the absolute difference between the original image and the reconstructed image using the 7th to 4th bit planes
diff3 = abs(xd - r3);

% Display the reconstructed images and the difference images
figure, imshow(r1);     
figure, imshow(r2);     
figure, imshow(r3);     
figure, imshow(diff1);  
figure, imshow(diff2);  
figure, imshow(diff3);  

%% Histogram Operation
%plot histogram
pollen_img = imread('pollen.tif');
figure;
subplot(2, 2, 1);
imshow(pollen_img);
subplot(2, 2, 2);
imhist(pollen_img);

%equalizing
equalized_pollen_img = histeq(pollen_img);
figure;
subplot(2, 2, 1);
imshow(equalized_pollen_img);
title('Equalized Pollen Image');
subplot(2, 2, 2);
imhist(equalized_pollen_img);
title('Histogram of Equalized Pollen Image');

%adjust
adjusted_pollen_img = imadjust(pollen_img, [0.05 0.6], [0 1]);
figure;
subplot(2, 2, 1);
imshow(adjusted_pollen_img);
subplot(2, 2, 2);
imhist(adjusted_pollen_img);

%plot histogram
aerial_img = imread('aerial.tif');
figure;
subplot(2, 2, 1);
imshow(aerial_img);
subplot(2, 2, 2);
imhist(aerial_img);

%equalizing
equalized_aerial_img = histeq(aerial_img);
figure;
subplot(2, 2, 1);
imshow(equalized_aerial_img);
subplot(2, 2, 2);
imhist(equalized_aerial_img);
%adjust
adjusted_aerial_img = imadjust(aerial_img, [0.6 0.98], [0 1]);
figure;
subplot(2, 2, 1);
imshow(adjusted_aerial_img);
subplot(2, 2, 2);
imhist(adjusted_aerial_img);

%% Transformations and registration
% Read the image 'peppers.bmp' into the variable f
f = imread('peppers.bmp');

% Define scaling factors for the x and y directions
sx = 0.85;
sy = 1.15;

% Define the affine transformation matrix for scaling
T1 = [sx 0 0
      0 sy 0
      0  0 1];

% Create an affine2d object for the scaling transformation
t1 = affine2d(T1);

% Apply the scaling transformation to the image
fs = imwarp(f, t1);

% Define the rotation angle (theta = pi/6)
theta = pi/6;

% Define the affine transformation matrix for rotation
T2 = [cos(theta) sin(theta) 0
     -sin(theta) cos(theta) 0
         0          0       1];

% Create an affine2d object for the rotation transformation
t2 = affine2d(T2);

% Apply the rotation transformation to the scaled image
fsr = imwarp(fs, t2);

% Define the shear factor (alpha = 0.75)
alpha = 0.75;

% Define the affine transformation matrix for shearing
T3 = [1    0    0
     alpha 1    0
      0    0    1];

% Create an affine2d object for the shearing transformation
t3 = affine2d(T3);

% Apply the shearing transformation to the scaled image
fss = imwarp(fs, t3);

% Display the original image and the transformed images (scaled, rotated, sheared)
figure;
subplot(2, 2, 1);
imshow(f);
title("Original Image (f)");

subplot(2, 2, 2);
imshow(fs);
title("Scaled Image (fs)");

subplot(2, 2, 3);
imshow(fsr);
title("Scaled and Rotated Image (fsr)");

subplot(2, 2, 4);
imshow(fss);
title("Scaled and Sheared Image (fss)");

% Perform image registration for the scaled image (fs) to the original image (f) using cpselect
cpselect(fs, f);
save('base_point.mat', 'base_points');
save('input_point.mat', 'input_points');

% Fit an affine transformation using the selected tie points
tietform = fitgeotrans(input_points, base_points, 'affine');

% Apply the transformation to register fs with f
fs2 = imwarp(fs, tietform, 'OutputView', imref2d(size(f)));

% Display the registered image, original image, and the difference between them
figure;
subplot(2, 2, 1);
imshow(fs2);
title("Registered Image (fs2, 4 pairs)");

subplot(2, 2, 2);
imshow(fs);
title("Target Image (fs)");

subplot(2, 2, 3);
imshow(f);
title("Original Image (f)");

subplot(2, 2, 4);
imshow(fs2 - f);
title("Difference Image (fs2 - f)");

% Perform image registration for the scaled and rotated image (fsr) to the original image (f) using cpselect
cpselect(fsr, f);
save('base_point.mat', 'base_points_2');
save('input_point.mat', 'input_points_2');

% Fit an affine transformation using the selected tie points
tietform2 = fitgeotrans(input_points_2, base_points_2, 'affine');

% Apply the transformation to register fsr with f
fsr2 = imwarp(fsr, tietform2, 'OutputView', imref2d(size(f)));

% Display the registered image, original image, and the difference between them
figure;
subplot(2, 2, 1);
imshow(fsr2);
title("Registered Rotated Image (fsr2, 4 pairs)");

subplot(2, 2, 2);
imshow(fsr);
title("Target Rotated Image (fsr)");

subplot(2, 2, 3);
imshow(f);
title("Original Image (f)");

subplot(2, 2, 4);
imshow(fsr2 - f);
title("Difference Image (fsr2 - f)");

% Perform image registration for the scaled and sheared image (fss) to the original image (f) using cpselect
cpselect(fss, f);
save('base_point.mat', 'base_points_3');
save('input_point.mat', 'input_points_3');

% Fit an affine transformation using the selected tie points
tietform3 = fitgeotrans(input_points_3, base_points_3, 'affine');

% Apply the transformation to register fss with f
fss2 = imwarp(fss, tietform3, 'OutputView', imref2d(size(f)));

% Display the registered image, original image, and the difference between them
figure;
subplot(2, 2, 1);
imshow(fss2);
title("Registered Sheared Image (fss2, 4 pairs)");

subplot(2, 2, 2);
imshow(fss);
title("Target Sheared Image (fss)");

subplot(2, 2, 3);
imshow(f);
title("Original Image (f)");

subplot(2, 2, 4);
imshow(fss2 - f);
title("Difference Image (fss2 - f)");

% Perform image registration for the scaled image (fs) with 8 tie points
cpselect(fs, f);
save('base_point.mat', 'base_points_4');
save('input_point.mat', 'input_points_4');

% Fit an affine transformation using 8 selected tie points
tietform4 = fitgeotrans(input_points_4, base_points_4, 'affine');

% Apply the transformation to register fs with f using 8 pairs
fs5 = imwarp(fs, tietform4, 'OutputView', imref2d(size(f)));

% Display the registered image, original image, and the difference between them
figure;
subplot(2, 2, 1);
imshow(fs5);
title("Registered Image (fs5, 8 pairs)");

subplot(2, 2, 2);
imshow(fs);
title("Target Image (fs)");

subplot(2, 2, 3);
imshow(f);
title("Original Image (f)");

subplot(2, 2, 4);
imshow(fs5 - f);
title("Difference Image (fs5 - f)");

% Perform image registration for the scaled and rotated image (fsr) with 8 tie points
cpselect(fsr, f);
save('base_point.mat', 'base_points_5');
save('input_point.mat', 'input_points_5');

% Fit an affine transformation using 8 selected tie points
tietform5 = fitgeotrans(input_points_5, base_points_5, 'affine');

% Apply the transformation to register fsr with f using 8 pairs
fs6 = imwarp(fsr, tietform5, 'OutputView', imref2d(size(f)));

% Display the registered image, original image, and the difference between them
figure;
subplot(2, 2, 1);
imshow(fs6);
title("Registered Rotated Image (fs6, 8 pairs)");

subplot(2, 2, 2);
imshow(fsr);
title("Target Rotated Image (fsr)");

subplot(2, 2, 3);
imshow(f);
title("Original Image (f)");

subplot(2, 2, 4);
imshow(fs6 - f);
title("Difference Image (fs6 - f)");

% Perform image registration for the scaled and sheared image (fss) with 8 tie points
cpselect(fss, f);
save('base_point.mat', 'base_points_6');
save('input_point.mat', 'input_points_6');

% Fit an affine transformation using 8 selected tie points
tietform6 = fitgeotrans(input_points_6, base_points_6, 'affine');

% Apply the transformation to register fss with f using 8 pairs
fs7 = imwarp(fss, tietform6, 'OutputView', imref2d(size(f)));

% Display the registered image, original image, and the difference between them
figure;
subplot(2, 2, 1);
imshow(fs7);
title("Registered Sheared Image (fs7, 8 pairs)");

subplot(2, 2, 2);
imshow(fss);
title("Target Sheared Image (fss)");

subplot(2, 2, 3);
imshow(f);
title("Original Image (f)");

subplot(2, 2, 4);
imshow(fs7 - f);
title("Difference Image (fs7 - f)");
