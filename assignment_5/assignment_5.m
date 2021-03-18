%{ 
    Assignment #5 - Background Subtraction in Video Streams
    AMATH482 - Computational Methods For Data Science - Mar. 17th, 2021
    Taught by Professor Jason J. Bramburger (Winter 2021)
    Written By: Clarisa Leu-Rodriguez - email: cleu@uw.edu
%}


%% Import Videos
clear all; close all; clc;  % Clear and refresh workspace.

% We use the low resolution versions for ease of computation.
% Uncomment depending on which video you want to perform DMD on.
%video = VideoReader('ski_drop_low.mp4'); vid_title = "Ski Drop";
video = VideoReader('monte_carlo_low.mp4'); vid_title = "Monte Carlo";


%% Extract Data for DMD
video_frames = read(video);
num_frames = video.NumFrames;
frame_width = video.Width;
frame_height = video.Height;
dt = 1; t = 1:num_frames;

% Construct struct needed to use frame2im() function.
for j = 1:num_frames
    vid_dat(j).cdata = video_frames(:, :, :, j);
    vid_dat(j).colormap = [];
end

X = zeros(frame_height * frame_width, num_frames);  % Each frame is a column in X.
for j = 1:num_frames
    x = frame2im(vid_dat(j));
    % Show original video
    % imshow(x);
    x = rgb2gray(x);
    % Flip contrast of the image (visually works better with DMD).
    x = imcomplement(x);
    x = im2double(x);
    X(: , j) = reshape(x, [frame_height * frame_width, 1]);
end

% Create DMD Matrices from video data.
X1 = X(:, 1:end-1); X2 = X(:, 2:end);


%% DMD: Perform SVD & Truncate U, Sigma, and V to Lower Rank
[U, Sigma, V] = svd(X1, 'econ');  % SVD of X1

% Plot Singular Value Spectrum & Energy From SVD
figure();
subplot(2, 1, 1);
plot(diag(Sigma), 'ko', 'LineWidth', 2);
title('Singular Value Spectrum - ' + vid_title); xlabel('Mode'); 
ylabel('Singular Value'); grid on; set(gca, 'fontsize', 12);
subplot(2, 1, 2);
plot(diag(Sigma) / sum(diag(Sigma)), 'ko', 'LineWidth', 2);
title('Energies of Singular Values - ' + vid_title); xlabel('Mode'); 
ylabel('Energy Captured'); grid on; set(gca, 'fontsize', 12);

% Truncate to rank with 90% energy
energy = 0;
total_energy = sum(diag(Sigma));
threshold = 0.9; 
r = 0;
while energy <= threshold
    r = r + 1;
    energy = energy + Sigma(r,r) / total_energy;
end

% Low-rank approximation of U, Sigma, and V
U_r = U(:, 1:r);
Sigma_r = Sigma(1:r, 1:r);
V_r = V(:, 1:r);


%% DMD: Computation of ~S & DMD Modes
S = U_r' * X2 * V_r / Sigma_r;
[eV, D] = eig(S);  % Compute eigenvalues and eigenvectors.
mu = diag(D);
omega = log(mu) / dt;
Phi = U_r * eV;

% Plot Omegas
figure();
plot(real(omega), imag(omega), 'r.', 'Markersize', 20);
title("\omega Values - " + vid_title);
xlabel('Re(\omega)'); ylabel('Im(\omega)');
grid on; set(gca, 'fontsize', 12);


%% DMD: Create DMD Solution
y0 = Phi\X1(:, 1);  % Pseudoinverse to get initial conditions.
% We pick the smallest omega which summarizes the background.
[min_omega, min_index] = min(abs(omega));
X_dmd = y0(min_index).*Phi(:, min_index).*exp(omega(min_index).*t);


%% DMD: Create Sparse and Nonsparse
X_sparse = X - abs(X_dmd);
R = X_sparse.*(X_sparse < 0);
X_background = R + abs(X_dmd);
X_foreground = X_sparse - R;
X_reconstructed = X_foreground + X_background;


%% DMD: Play back video frame by frame
figure();
for j = 1:num_frames
    % Reshape & flip contrast back.
    background = reshape(X_background(:, j), [frame_height, frame_width]);
    background = imcomplement(background);
    foreground = reshape(X_foreground(:, j), [frame_height, frame_width]);
    foreground = imcomplement(foreground);
    reconstructed = reshape(X_reconstructed(:, j), [frame_height, frame_width]);
    reconstructed = imcomplement(reconstructed);
    
    % Plot Background, Foreground, and Reconstructed
    subplot(3, 1, 1);
    imshow(im2uint8(background));
    title('Background - ' + vid_title); set(gca, 'fontsize', 12);

    subplot(3, 1, 2);
    imshow(im2uint8(foreground));
    title('Foreground - ' + vid_title); set(gca, 'fontsize', 12);
    
    subplot(3, 1, 3);
    imshow(im2uint8(reconstructed));
    title('Sum of Background & Foreground - ' + vid_title);
    set(gca, 'fontsize', 12);
    
    drawnow;
end

