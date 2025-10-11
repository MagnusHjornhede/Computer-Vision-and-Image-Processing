%% ================================
%  - Computer Vision and Robotics -
%  - Image Processing Tasks - 
%  - Magnus H
% =================================

clear; clc; close all;

% Set up project folder structure (relative paths for GitHub)
root = fullfile(pwd, 'CaseStudy1_ImageProcessing');
img_dir = fullfile(root, 'images');
if ~exist(img_dir, 'dir')
    mkdir(img_dir);
end

%% **************** Task 1 — Basic Image Operations ****************

% Load Barbara image from .mat file
barbara_mat = load('barbara.mat');
barbara = barbara_mat.barbara;
imshow(barbara);
imwrite(barbara, fullfile(img_dir, 'barbara_from_mat.png'));

% Load from .png using MATLAB and Peter Corke’s toolbox
barbara_png = imread('barbara.png');
imwrite(barbara_png, fullfile(img_dir, 'barbara_from_png_MATLAB.png'));

if exist('iread', 'file')  % check if MVTB is installed
    barbara_pcv = iread('barbara.png');
    imwrite(barbara_pcv, fullfile(img_dir, 'barbara_from_png_PeterCorke.png'));
end

% Convert to grayscale and save
gray_barbara = rgb2gray(barbara_png);
imwrite(gray_barbara, fullfile(img_dir, 'grayscale_barbara.png'));

% Convert types and save examples
img_double = im2double(gray_barbara);
imwrite(img_double, fullfile(img_dir, 'double_gray_barbara.png'));

img_uint8 = im2uint8(img_double);
imwrite(img_uint8, fullfile(img_dir, 'uint8_gray_barbara.png'));

% Resize images
imwrite(imresize(gray_barbara, 0.5), fullfile(img_dir, 'gray_half_barbara.png'));
imwrite(imresize(gray_barbara, 0.25), fullfile(img_dir, 'gray_quarter_barbara.png'));

% Adjust brightness (gamma < 1 brightens)
brighter_img = imadjust(gray_barbara, [], [], 0.7);
imwrite(brighter_img, fullfile(img_dir, 'brighter_gray_barbara.png'));

% Visualize histograms
figure; imhist(gray_barbara);
saveas(gcf, fullfile(img_dir, 'original_histogram.png'));

figure; imhist(brighter_img);
saveas(gcf, fullfile(img_dir, 'brightened_histogram.png'));

close all;


%% **************** Task 2 — Image Restoration ****************

% Load flower image
flower_data = load('flower.mat');
flower = flower_data.flower;
imwrite(flower, fullfile(img_dir, 'flower_original.png'));

flower_gray = im2gray(im2double(flower));

% Gaussian Noise (SNR = 20 dB)
SNR_dB = 20;
SNR_linear = 10^(SNR_dB / 10);
signal_var = var(flower_gray(:));
noise_var = signal_var / SNR_linear;

noisy_gauss = imnoise(flower_gray, 'gaussian', 0, noise_var);
imwrite(noisy_gauss, fullfile(img_dir, 'flower_noisy_gauss.png'));

% Average and Median Filters
h_avg = fspecial('average', [5 5]);
imwrite(imfilter(noisy_gauss, h_avg), fullfile(img_dir, 'flower_filtered_avg_gauss.png'));
imwrite(medfilt2(noisy_gauss, [3 3]), fullfile(img_dir, 'flower_filtered_median_gauss.png'));

% Salt & Pepper Noise (20%)
noisy_sp = imnoise(flower_gray, 'salt & pepper', 0.2);
imwrite(noisy_sp, fullfile(img_dir, 'flower_noisy_saltpepper.png'));
imwrite(imfilter(noisy_sp, h_avg), fullfile(img_dir, 'flower_filtered_avg_saltpepper.png'));
imwrite(medfilt2(noisy_sp, [3 3]), fullfile(img_dir, 'flower_filtered_median_saltpepper.png'));

% Combined noise → sequential filtering
noisy_combined = imnoise(noisy_gauss, 'salt & pepper', 0.2);
imwrite(noisy_combined, fullfile(img_dir, 'flower_noisy_combined.png'));

filtered_combined = medfilt2(imfilter(noisy_combined, h_avg), [3 3]);
imwrite(filtered_combined, fullfile(img_dir, 'flower_filtered_combined.png'));


%% **************** Task 3 — Edge Detection ****************

% Load clock image
clock_data = load('clock.mat');
clock_img = clock_data.clock;
imwrite(clock_img, fullfile(img_dir, 'clock_original.png'));

% Sobel
sobel_edges = edge(clock_img, 'sobel');
imwrite(sobel_edges, fullfile(img_dir, 'clock_sobel_edges.png'));
imwrite(imbinarize(im2double(sobel_edges)), fullfile(img_dir, 'clock_sobel_binary.png'));

% Laplacian of Gaussian (LoG)
log_edges = edge(clock_img, 'log');
imwrite(log_edges, fullfile(img_dir, 'clock_log_edges.png'));
imwrite(imbinarize(im2double(log_edges)), fullfile(img_dir, 'clock_log_binary.png'));

% Canny
canny_edges = edge(clock_img, 'canny');
imwrite(canny_edges, fullfile(img_dir, 'clock_canny_edges.png'));
imwrite(imbinarize(im2double(canny_edges)), fullfile(img_dir, 'clock_canny_binary.png'));

disp('✅ Lab 1 processing complete. Results saved in images/.');
