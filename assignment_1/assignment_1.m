%{ 
    Assignment #1 - A Submarine Problem
    AMATH482 - Computational Methods For Data Science -January 27th, 2021
    Taught by Professor Jason J. Bramburger (Winter 2021)
    Written By: Clarisa Leu-Rodriguez - email: cleu@uw.edu
%}
 
clear all; close all; clc
 
load subdata.mat  % Imports the data as the 262144x49 (space by time) matrix 
                  % called subdata. 
 
L = 10;  % Spatial domain
n = 64;  % Fourier modes
realizations = 49;  % The number of "time slices" in our data.
 
% We assume the domain is periodic so the last point is the same as the 
% first.
x2 = linspace(-L, L, n+1); x = x2(1:n); y = x; z = x;
k = (2*pi / (2*L))*[0:(n/2 - 1) -n/2:-1]; ks = fftshift(k);
[X, Y, Z] = meshgrid(x, y, z);  % X, Y, Z - spatial domain
[Kx, Ky, Kz] = meshgrid(ks, ks, ks);  % Kx, Ky, Kz - frequency domain
 
x_pos_noisy = zeros(1, realizations);
y_pos_noisy = zeros(1, realizations); z_pos_noisy = zeros(1, realizations);
unt_ave = zeros(n, n, n);
for j = 1:realizations
    un(:, :, :) = reshape(subdata(:, j), n, n, n);
    [max_val_dat_noise, k] = max(abs(un(:)));
    [max_val_x, max_val_y, max_val_z] = ind2sub(size(un), k);
    x_pos_noisy(j) = X(max_val_x, max_val_y, max_val_z);
    y_pos_noisy(j) = Y(max_val_x, max_val_y, max_val_z);
    z_pos_noisy(j) = Z(max_val_x, max_val_y, max_val_z);
    unt_ave = unt_ave + fftshift(fftn(un));
end
unt_ave = abs(unt_ave) ./ realizations;  % Average the spectrum
 
% Plot the position of the max sonar reading for each time slice in 
% the noisy data to compare to denoised data later.
figure(1)
plot3(x_pos_noisy, y_pos_noisy, z_pos_noisy, 'k-', 'LineWidth', 3);
set(gcf, 'position', [400, 300, 600, 500]);
title('Position of Max Sonar Reading from Noisy Data in 3-Dimensions Over Time', ...
    'Fontsize', 20);
xlabel('X-Axis');  ylabel('Y-Axis');  zlabel('Z-Axis');
grid on, hold on
% Plot the last coordinate
plot3(x_pos_noisy(realizations), y_pos_noisy(realizations), ... 
    z_pos_noisy(realizations), 'ro');
fprintf('Ending Position From Noisy Data is: (%f, %f, %f). \n', ... 
    x_pos_noisy(realizations), y_pos_noisy(realizations), ... 
    z_pos_noisy(realizations));
 
% Look at the isosurface of the noisy data.
figure(2)
set(gcf, 'position', [400, 300, 600, 500]);
isosurface(X, Y, Z, abs(un), 0.4, 'r'); grid on;
title('Acoustic Data (Noisy)','Fontsize', 20); xlabel('X-Axis');
ylabel('Y-Axis'); zlabel('Z-Axis');
axis([-20 20 -20 20 -20 20]);
 
% Find the peak in our data within the frequency domain by normalizing.
max_val = max(unt_ave(:));
unt_ave = unt_ave ./ max_val;
 
% Find frequency signature of submarine.
i = find(unt_ave == 1);
Kx0 = Kx(i);  Ky0 = Ky(i);  Kz0 = Kz(i);
fprintf('Frequency Signature of Submarine is: (%f, %f, %f). \n', Kx0, ... 
            Ky0, Kz0);
 
% Remove noise from data and center around frequency
tau = 0.5;  % Pick a bandwidth which makes data look "smoothest"
filter = exp(-tau.*((Kx-Kx0).^2 + (Ky-Ky0).^2 + (Kz-Kz0).^2));
 
x_pos = zeros(1, realizations); y_pos = zeros(1, realizations);
z_pos = zeros(1, realizations);
for j = 1:realizations
    un_filter(:, :, :) = fftshift(fftn(reshape(subdata(:, j), n, n, n)));
    % Apply filter
    unt_filter(:, :, :) = un_filter.*filter;
    sub(:, :, :) = ifftn(unt_filter);
    % Find the max value for each time slice and get positional
    % coordinates to graph.
    [max_val_dat, k] = max(abs(sub(:)));
    [max_val_x, max_val_y, max_val_z] = ind2sub(size(sub), k);
    x_pos(j) = X(max_val_x, max_val_y, max_val_z);
    y_pos(j) = Y(max_val_x, max_val_y, max_val_z);
    z_pos(j) = Z(max_val_x, max_val_y, max_val_z);
end
 
% Plot the submarines position over time.
figure(3)
plot3(x_pos, y_pos, z_pos, 'k-', 'LineWidth', 3);
set(gcf, 'position', [400, 300, 600, 500]);
title('Submarine Movement in 3-Dimensions Over Time', 'Fontsize', 20);
xlabel('X-Axis');  ylabel('Y-Axis');  zlabel('Z-Axis');
grid on, hold on
% Plot the ending position of the submarine.
plot3(x_pos(realizations), y_pos(realizations), z_pos(realizations), 'ro');
fprintf('Ending Position of Submarine is: (%f, %f, %f). \n', ... 
    x_pos(realizations), y_pos(realizations), z_pos(realizations));
 
 
% Output the submarines positional coordinates.
for j = 1:realizations
    fprintf('Time Slice: %f, Position: (%f, %f, %f)\n', j, x_pos(j), ...
        y_pos(j), z_pos(j));
end

