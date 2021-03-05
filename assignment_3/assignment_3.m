%{ 
    Assignment #3 - PCA and a Spring-Mass System
    AMATH482 - Computational Methods For Data Science - Feb. 24th, 2021
    Taught by Professor Jason J. Bramburger (Winter 2021)
    Written By: Clarisa Leu-Rodriguez - email: cleu@uw.edu
%}
 
%% Import Data
 
clear all; close all; clc; 
 
% Test #1: Ideal Case
load('cam1_1.mat');
load('cam2_1.mat');
load('cam3_1.mat');
 
% Test #2: Noisy Case
load('cam1_2.mat');
load('cam2_2.mat');
load('cam3_2.mat');
 
% Test #3: Horizontal Displacement
load('cam1_3.mat');
load('cam2_3.mat');
load('cam3_3.mat');
 
% Test #4: Horizontal Displacement and Rotation
load('cam1_4.mat');
load('cam2_4.mat');
load('cam3_4.mat');
 
% Constants used throughout script:
% Video data is of dimensions 480 x 640 x 3 x num_frames
% That is - the width of each frame is 640 pixels, the height is 480
% pixels, the pixels have an associated RGB value, and the number of frames
% vary between each set of data.
frame_width = size(vidFrames1_1, 2);
frame_height = size(vidFrames1_1, 1);  % Frame width and height are constant
                                       % between all datasets - however
                                       % frame width varies.
 
 
%% Test #1: Ideal Case - Extract Time Series Data
% Find filter which crops out background well through trial/error - in addition
% to good value to set as the threshold for grayscale values to look at.
 
filter_cam_1 = zeros(frame_height, frame_width);
% Specify x-bound/y-bound to keep - these are the bounds for the paint can
% in the video.
x_bound_filt_1 = 280:1:400;
y_bound_filt_1 = 140:1:450;
filter_cam_1(y_bound_filt_1, x_bound_filt_1) = 1;
time_series_dat_cam_1 = get_time_series_dat(vidFrames1_1, filter_cam_1, 240, 0); 
 
filter_cam_2 = zeros(frame_height, frame_width);
% Specify x-bound/y-bound to keep - these are the bounds for the paint can
% in the video.
x_bound_filt_2 = 230:1:360;
y_bound_filt_2 = 80:1:405;
filter_cam_2(y_bound_filt_2, x_bound_filt_2) = 1;
time_series_dat_cam_2 = get_time_series_dat(vidFrames2_1, filter_cam_2, 240, 0);
 
filter_cam_3 = zeros(frame_height, frame_width);
% Specify x-bound/y-bound to keep - these are the bounds for the paint can
% in the video.
x_bound_filt_3 = 220:1:500;
y_bound_filt_3 = 220:1:380;
filter_cam_3(y_bound_filt_3, x_bound_filt_3) = 1;
time_series_dat_cam_3 = get_time_series_dat(vidFrames3_1, filter_cam_3, 240, 0);
 
 
%% Test #1: Ideal Case - Perform PCA
% Data should be of the same dimension for PCA - so align data.
dat = align_dat(time_series_dat_cam_1, time_series_dat_cam_2, time_series_dat_cam_3, 21);
% Subtract the mean from each row.
mean_val = mean(dat, 2);
dat = dat - repmat(mean_val, 1, size(dat, 2));
 
% Calculate SVD & Energies. Also scale by 1 / sqrt(# of time points) to align in
% space).
[U, S, V] = svd(dat' ./ sqrt(size(dat, 2)));
lambdas = diag(S).^2;
energies_1 = lambdas ./ sum(lambdas);  % Save energies to plot later.
 
% Project principal components to compare with original.
Y = dat' * V;
 
 
%% Test #1: Ideal Case - Figures
% Plot change in position from original data.
figure();
subplot(2, 1, 1);
x_axis = 1:size(dat, 2);
plot(x_axis, dat(2, :), x_axis, dat(1, :), 'Linewidth', 2);
ylabel("Change in Position"); xlabel("Time (frame number)"); 
title("Test #1: Ideal Case - Change in Position for Camera #1 (Original)");
legend("Z Direction", "X-Y Plane");
grid on; set(gca, 'Fontsize', 13);
 
% Plot projection.
subplot(2, 1, 2);
plot(x_axis, Y(:, 1), x_axis, Y(:, 2), 'Linewidth', 2);
ylabel("Change in Position"); xlabel("Time (frame number)"); 
title("Test #1: Ideal Case - Principal Components");
legend("Principal Component #1", "Principal Component #2");
grid on; set(gca, 'Fontsize', 13);
 
 
%% Test #2: Noisy Case - Extract Time Series Data
% Find filter which crops out background well through trial/error - in addition
% to good value to set as the threshold for grayscale values to look at.
filter_cam_1 = zeros(frame_height, frame_width);
% Specify x-bound/y-bound to keep - these are the bounds for the paint can
% in the video.
x_bound_filt_1 = 290:1:415;
y_bound_filt_1 = 160:1:425;
filter_cam_1(y_bound_filt_1, x_bound_filt_1) = 1;
time_series_dat_cam_1 = get_time_series_dat(vidFrames1_2, filter_cam_1, 240, 0); 
 
filter_cam_2 = zeros(frame_height, frame_width);
% Specify x-bound/y-bound to keep - these are the bounds for the paint can
% in the video.
x_bound_filt_2 = 165:1:425;
y_bound_filt_2 = 50:1:475;
filter_cam_2(y_bound_filt_2, x_bound_filt_2) = 1;
time_series_dat_cam_2 = get_time_series_dat(vidFrames2_2, filter_cam_2, 240, 0);
 
 
filter_cam_3 = zeros(frame_height, frame_width);
% Specify x-bound/y-bound to keep - these are the bounds for the paint can
% in the video.
x_bound_filt_3 = 235:1:495;
y_bound_filt_3 = 200:1:380;
filter_cam_3(y_bound_filt_3, x_bound_filt_3) = 1;
time_series_dat_cam_3 = get_time_series_dat(vidFrames3_2, filter_cam_3, 240, 0);
 
 
%% Test #2: Noisy Case - Perform PCA
% Data should be of the same dimension for PCA - so align data.
dat = align_dat(time_series_dat_cam_1, time_series_dat_cam_2, time_series_dat_cam_3, 19);
% Subtract the mean from each row.
mean_val = mean(dat, 2);
dat = dat - repmat(mean_val, 1, size(dat, 2));
 
% Calculate SVD & Energies. Also scale by 1 / sqrt(# of time points) to align in
% space).
[U, S, V] = svd(dat' ./ sqrt(size(dat, 2)));
lambdas = diag(S).^2;
energies_2 = lambdas ./ sum(lambdas);  % Save energies to plot later.
 
% Project principal components to compare with original.
Y = dat' * V;
 
 
%% Test #2: Noisy Case - Figures
% Plot change in position from original data.
figure();
subplot(2, 1, 1);
x_axis = 1:size(dat, 2);
plot(x_axis, dat(2, :), x_axis, dat(1, :), 'Linewidth', 2);
ylabel("Change in Position"); xlabel("Time (frame number)"); 
title("Test #2: Noisy Case - Change in Position for Camera #1 (Original)");
legend("Z Direction", "X-Y Plane");
grid on; set(gca, 'Fontsize', 13);
 
% Plot projection.
subplot(2, 1, 2);
plot(x_axis, Y(:, 1), x_axis, Y(:, 2), x_axis, Y(:, 3), ...
        'Linewidth', 2);
ylabel("Change in Position"); xlabel("Time (frame number)"); 
title("Test #2: Noisy Case - Principal Components");
legend("Principal Component #1", "Principal Component #2", ...
    "Principal Component #3");
grid on; set(gca, 'Fontsize', 13);
 
 
%% Test #3: Horizontal Displacement - Extract Time Series Data
% Find filter which crops out background well through trial/error - in addition
% to good value to set as the threshold for grayscale values to look at.
 
filter_cam_1 = zeros(frame_height, frame_width);
% Specify x-bound/y-bound to keep - these are the bounds for the paint can
% in the video.
x_bound_filt_1 = 265:1:415;
y_bound_filt_1 = 225:1:440;
filter_cam_1(y_bound_filt_1, x_bound_filt_1) = 1;
time_series_dat_cam_1 = get_time_series_dat(vidFrames1_3, filter_cam_1, 240, 0); 
 
filter_cam_2 = zeros(frame_height, frame_width);
% Specify x-bound/y-bound to keep - these are the bounds for the paint can
% in the video.
x_bound_filt_2 = 165:1:425;
y_bound_filt_2 = 140:1:425;
filter_cam_2(y_bound_filt_2, x_bound_filt_2) = 1;
time_series_dat_cam_2 = get_time_series_dat(vidFrames2_3, filter_cam_2, 250, 0);
 
filter_cam_3 = zeros(frame_height, frame_width);
% Specify x-bound/y-bound to keep - these are the bounds for the paint can
% in the video.
x_bound_filt_3 = 225:1:490;
y_bound_filt_3 = 160:1:360;
filter_cam_3(y_bound_filt_3, x_bound_filt_3) = 1;
time_series_dat_cam_3 = get_time_series_dat(vidFrames3_3, filter_cam_3, 240, 0);
 
 
%% Test #3: Horizontal Displacement - Perform PCA
% Data should be of the same dimension for PCA - so align data.
dat = align_dat(time_series_dat_cam_1, time_series_dat_cam_2, time_series_dat_cam_3, 14);
% Subtract the mean from each row.
mean_val = mean(dat, 2);
dat = dat - repmat(mean_val, 1, size(dat, 2));
 
% Calculate SVD & Energies. Also scale by 1 / sqrt(# of time points) to align in
% space).
[U, S, V] = svd(dat' ./ sqrt(size(dat, 2)));
lambdas = diag(S).^2;
energies_3 = lambdas ./ sum(lambdas);  % Save energies to plot later.
 
% Project principal components to compare with original.
Y = dat' * V;
 
 
%% Test #3: Horizontal Displacement - Figures
% Plot change in position from original data.
figure();
subplot(2, 1, 1);
x_axis = 1:size(dat, 2);
plot(x_axis, dat(2, :), x_axis, dat(1, :), 'Linewidth', 2);
ylabel("Change in Position"); xlabel("Time (frame number)"); 
title("Test #3: Horizontal Displacement - Change in Position for Camera #1 (Original)");
legend("Z Direction", "X-Y Plane");
grid on; set(gca, 'Fontsize', 13);
 
% Plot projection.
subplot(2, 1, 2);
plot(x_axis, Y(:, 1), x_axis, Y(:, 2), x_axis, Y(:, 3), 'Linewidth', 2);
ylabel("Change in Position"); xlabel("Time (frame number)"); 
title("Test #3: Horizontal Displacement - Principal Components");
legend("Principal Component #1", "Principal Component #2", "Principal Component #3");
grid on; set(gca, 'Fontsize', 13);
 
 
%% Test #4: Horizontal Displacement & Rotation - Extract Time Series Data
% Find filter which crops out background well through trial/error - in addition
% to good value to set as the threshold for grayscale values to look at.
 
filter_cam_1 = zeros(frame_height, frame_width);
% Specify x-bound/y-bound to keep - these are the bounds for the paint can
% in the video.
x_bound_filt_1 = 275:1:470;
y_bound_filt_1 = 225:1:440;
filter_cam_1(y_bound_filt_1, x_bound_filt_1) = 1;
time_series_dat_cam_1 = get_time_series_dat(vidFrames1_4, filter_cam_1, 230, 0); 
 
filter_cam_2 = zeros(frame_height, frame_width);
% Specify x-bound/y-bound to keep - these are the bounds for the paint can
% in the video.
x_bound_filt_2 = 175:1:450;
y_bound_filt_2 = 110:1:390;
filter_cam_2(y_bound_filt_2, x_bound_filt_2) = 1;
time_series_dat_cam_2 = get_time_series_dat(vidFrames2_4, filter_cam_2, 250, 0);
 
filter_cam_3 = zeros(frame_height, frame_width);
% Specify x-bound/y-bound to keep - these are the bounds for the paint can
% in the video.
x_bound_filt_3 = 245:1:490;
y_bound_filt_3 = 130:1:320;
filter_cam_3(y_bound_filt_3, x_bound_filt_3) = 1;
time_series_dat_cam_3 = get_time_series_dat(vidFrames3_4, filter_cam_3, 220, 0);
 
 
%% Test #4: Horizontal Displacement & Rotation - Perform PCA
% Data should be of the same dimension for PCA - so align data.
dat = align_dat(time_series_dat_cam_1, time_series_dat_cam_2, time_series_dat_cam_3, 15);
% Subtract the mean from each row.
mean_val = mean(dat, 2);
dat = dat - repmat(mean_val, 1, size(dat, 2));
 
% Calculate SVD & Energies. Also scale by 1 / sqrt(# of time points) to align in
% space).
[U, S, V] = svd(dat' ./ sqrt(size(dat, 2)));
lambdas = diag(S).^2;
energies_4 = lambdas ./ sum(lambdas);  % Save energies to plot later.
 
% Project principal components to compare with original.
Y = dat' * V;
 
 
%% Test #4: Horizontal Displacement & Rotation - Figures
% Plot change in position from original data.
figure();
subplot(2, 1, 1);
x_axis = 1:size(dat, 2);
plot(x_axis, dat(2, :), x_axis, dat(1, :), 'Linewidth', 2);
ylabel("Change in Position"); xlabel("Time (frame number)"); 
title("Test #4: Horizontal Displacement & Rotation - Change in Position for Camera #1 (Original)");
legend("Z Direction", "X-Y Plane");
grid on; set(gca, 'Fontsize', 13);
 
% Plot projection.
subplot(2, 1, 2);
plot(x_axis, Y(:, 1), x_axis, Y(:, 2), x_axis, Y(:, 3), 'Linewidth', 2);
ylabel("Change in Position"); xlabel("Time (frame number)"); 
title("Test #4: Horizontal Displacement & Rotation - Principal Components");
legend("Principal Component #1", "Principal Component #2", "Principal Component #3");
grid on; set(gca, 'Fontsize', 13);
 
 
%% Plot Energies For All Tests to Compare
figure();
x_axis = 1:6;
subplot(2, 2, 1);
plot(x_axis, energies_1, 'ko', 'LineWidth', 2);
xlabel("Dimension"); ylabel("Percentage of Energy Captured in Dimension");
title("Test #1: Ideal Case - Energies");
grid on; set(gca, 'Fontsize', 12);
 
subplot(2, 2, 2);
plot(x_axis, energies_2, 'ko', 'LineWidth', 2);
xlabel("Dimension"); ylabel("Percentage of Energy Captured in Dimension");
title("Test #2: Noisy Case - Energies");
grid on; set(gca, 'Fontsize', 12);
 
subplot(2, 2, 3);
plot(x_axis, energies_3, 'ko', 'LineWidth', 2);
xlabel("Dimension"); ylabel("Percentage of Energy Captured in Dimension");
title("Test #3: Horizontal Displacement - Energies");
grid on; set(gca, 'Fontsize', 12);
 
subplot(2, 2, 4);
plot(x_axis, energies_4, 'ko', 'LineWidth', 2);
xlabel("Dimension"); ylabel("Percentage of Energy Captured in Dimension");
title("Test #4: Horizontal Displacement & Rotation - Energies");
grid on; set(gca, 'Fontsize', 12);
 

