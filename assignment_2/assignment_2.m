%{ 
    Assignment #2 - Rock & Roll and the Gabor Transform
    AMATH482 - Computational Methods For Data Science - Feb. 10th, 2021
    Taught by Professor Jason J. Bramburger (Winter 2021)
    Written By: Clarisa Leu-Rodriguez - email: cleu@uw.edu

    This MATLAB script is written with the intentions of being ran in
    sections, in order to explore the Gabor transform with a Gaussian
    window function.
    Please read each respective section for instructions on how to use/modify.
%}

%% Clear and refresh workspace.
clear; clc; close all;


%% Import Data
[y_gnr, Fs_gnr] = audioread('GNR.m4a');
y_gnr = y_gnr';
tr_gnr = length(y_gnr) / Fs_gnr; % record time in seconds.
t_gnr = (1:length(y_gnr)) / Fs_gnr;

[y_floyd, Fs_floyd] = audioread('Floyd.m4a');
y_floyd = y_floyd';
% Uncomment to analyze whole piece & get rid of the last data point so it 
% is of even length for plotting.
% y_floyd = y_floyd(1:end-1);

% We look at a smaller sample of the Pink Floyd song as it is 
% computationally expensive to analyze the whole clip at once.
len_floyd = floor(length(y_floyd) / 4);
% Modify the length of the floyd clip here.
% e.g. y_floyd((2*len_floyd + 1) : (3*len_floyd)) analyzes 30-45 seconds.
y_floyd = y_floyd(1 : len_floyd); 
tr_floyd = length(y_floyd) / Fs_floyd; % record time in seconds.
t_floyd = (1:length(y_floyd)) / Fs_floyd;


%% Plot unfiltered music signals in time/frequency space
close all;
figure();
% Plot time.
subplot(2, 2, 1);
plot((1:length(y_gnr)) / Fs_gnr, y_gnr, 'k', 'Linewidth', 2);
xlabel('Time (s)'); ylabel('Amplitude');
title('Guns N'' Roses - Sweet Child O'' Mine (Time Domain)');
grid on;
set(gca, 'Fontsize', 14);

subplot(2, 2, 2);
plot((1:length(y_floyd)) / Fs_floyd, y_floyd, 'k', 'Linewidth', 2);
xlabel('Time (s)'); ylabel('Amplitude');
title('Pink Floyd - Comfortably Numb (Time Domain)');
grid on;
set(gca, 'Fontsize', 14);

% Plot Frequency.
n_gnr = length(y_gnr);
L_gnr = tr_gnr;
% Scale k vector by 1/L because of Hz.
k_gnr = (1/L_gnr).*[0:n_gnr/2-1 -n_gnr/2:-1];
ks_gnr = fftshift(k_gnr);
yt_gnr = fft(y_gnr);

subplot(2, 2, 3);
plot(ks_gnr, abs(fftshift(yt_gnr))/max(abs(yt_gnr)), 'r', 'Linewidth', 2);
xlabel('Frequency (k)'); ylabel('fft(gnr data)');
title('Guns N'' Roses - Sweet Child O'' Mine (Frequency Domain)');
grid on;
set(gca, 'Fontsize', 14);

n_floyd = length(y_floyd);
L_floyd = tr_floyd;
 % Scale k vector by 1/L because of Hz.
k_floyd = (1/L_floyd).*[0:n_floyd/2-1 -n_floyd/2:-1];
ks_floyd = fftshift(k_floyd);
yt_floyd = fft(y_floyd);

subplot(2, 2, 4);
plot(ks_floyd, abs(fftshift(yt_floyd))/max(abs(yt_floyd)), 'r', 'Linewidth', 2);
xlabel('Frequency (k)'); ylabel('fft(floyd data)');
title('Pink Floyd - Comfortably Numb (Frequency Domain)');
grid on;
set(gca, 'Fontsize', 14);


%% Play song clip - GNR
 p8_gnr = audioplayer(y_gnr, Fs_gnr); playblocking(p8_gnr);
 
 
%% Play song clip - Pink Floyd
 p8_floyd = audioplayer(y_floyd, Fs_floyd); playblocking(p8_floyd);


%% Part #1 - Explore GNR - Gaussian Window Function - Find Good Width.
% This section can also be modified to find a good width for the Gaussian
% window function using the Pink Floyd clip.

close all;
time_step = 0.2;  % Modify time step here.
tau_gnr = 0:time_step:L_gnr;  
a_gnr_vals = [1 10 100 1000];  % Explore different a values for Gaussian window.
figure();
for i = 1:length(a_gnr_vals)
    a_gnr = a_gnr_vals(i);
    ygt_spec_gnr_gabor = zeros(length(tau_gnr), n_gnr);
    for j = 1:length(tau_gnr)
        % Using a Gaussian Window Filter.
        g = exp(-a_gnr*(t_gnr-tau_gnr(j)).^2);
        yg = g.*y_gnr;
        ygt = fft(yg);
        ygt_spec_gnr_gabor(j, :) = abs(fftshift(ygt));
    end
    % Plot spectragram.
    subplot(2, 2, i);
    pcolor(tau_gnr, ks_gnr, ygt_spec_gnr_gabor.');
    shading interp;
    title({['GNR - Sweet Child O'' Mine' ] 
        ['Gabor Transform with Gaussian Window']
        ['a = ' num2str(a_gnr) ' and \Delta\tau = ' num2str(time_step)]
        });
    colormap(hot);
    ylabel('Frequency (Hz)');  xlabel('Time (s)');
    set(gca,'ylim',[0, 2000],'Fontsize', 14);
end


%% Part #1 - Explore GNR - Gaussian Window Function - Find Good Time Step.
% This section can also be modified to find a good time step for the Gaussian
% window function using the Pink Floyd clip.

close all;
% Parameters for Gaussian Window Filter Function.
time_step_vals = [0.1 0.2 0.5 1];  % Explore different time steps.
a_gnr = 100;  % Modify width here.
figure();
for i = 1:length(time_step_vals)
    time_step = time_step_vals(i);
    tau_gnr = 0:time_step:L_gnr;
    ygt_spec_gnr_gabor = zeros(length(tau_gnr), n_gnr);
    for j = 1:length(tau_gnr)
        % Using a Gaussian Window Filter.
        g = exp(-a_gnr*(t_gnr-tau_gnr(j)).^2);
        yg = g.*y_gnr;
        ygt = fft(yg);
        ygt_spec_gnr_gabor(j, :) = abs(fftshift(ygt));
    end
    % Plot spectragram.
    subplot(2, 2, i);
    pcolor(tau_gnr, ks_gnr, ygt_spec_gnr_gabor.');
    shading interp;
    title({['GNR - Sweet Child O'' Mine' ] 
        ['Gabor Transform with Gaussian Window']
        ['a = ' num2str(a_gnr) ' and \Delta\tau = ' num2str(time_step)]
        });
    colormap(hot);
    ylabel('Frequency (Hz)');  xlabel('Time (s)');
    set(gca,'ylim',[0, 2000],'Fontsize', 14);
end

 
%% Part #1 - Explore GNR - Gaussian Window Function Generate Final Plot.
close all;
time_step = 0.2;  % Modify time step here.
tau_gnr = 0:time_step:L_gnr;
a_gnr = 100;  % Modify width here.
ygt_spec_gnr = zeros(length(tau_gnr), n_gnr);
ygt_spec_gnr_gabor = zeros(length(tau_gnr), n_gnr);
gnr_guitar_notes = zeros(1, length(tau_gnr));

for j = 1:length(tau_gnr)
    % Using a Gaussian Window Filter.
    g = exp(-a_gnr*(t_gnr-tau_gnr(j)).^2);
    yg = g.*y_gnr;
    ygt = fft(yg);
    ygt_spec_gnr_gabor(j, :) = abs(fftshift(ygt));
     
    % Find maximum frequency for each time slice - this is the note played.
    [~, i] = max(abs(ygt));
    gnr_guitar_notes(j) = k_gnr(i);
    
    % We then can use a Gaussian filter to filter around the maximum frequency
    % in the frequency domain for a cleaner looking spectragram if we want.
    gaussian_tau = 0.1;  % We pick a tau here which looks best.
    gaussian_filter = exp(-gaussian_tau*(k_gnr-gnr_guitar_notes(j)).^2);
    ygt_filtered = gaussian_filter.*ygt;
    ygt_spec_gnr(j, :) = abs(fftshift(ygt_filtered));
end

% Plot clean spectragrams (Gabor here).
figure();
pcolor(tau_gnr, ks_gnr, ygt_spec_gnr_gabor.');
shading interp;
title({['Guns N'' Roses - Sweet Child O'' Mine' ] 
    ['Spectragram (Guitar) - Gabor Transform with Gaussian Window']
    ['a = ' num2str(a_gnr) ' and \Delta\tau = ' num2str(time_step)]
    });
colormap(hot);
ylabel('Frequency (Hz)');  xlabel('Time (s)');
set(gca,'ylim',[0, 2000],'Fontsize', 14);

% Plot clean spectragrams (filtered, finished version - Gabor and Gaussian
% filter in freq. space).
figure();
pcolor(tau_gnr, ks_gnr, ygt_spec_gnr.');
shading interp;
title({['Guns N'' Roses - Sweet Child O'' Mine' ] 
    ['Spectragram (Guitar) - Clean']
    });
colormap(hot);
% Show notes on y-axis (pulled off visially by looking at spectragram).
yticks([277.18, 311.13, 369.99, 415.30])
yline(277.18, 'w'); yline(311.13, 'w'); yline(369.99, 'w'); yline(415.30, 'w');
yticklabels({'D♭ - 277.18','E♭ - 311.13', 'G♭ - 369.99', 'A♭ - 415.3'})
ylabel('Musical Note and Frequency (Hz)');  xlabel('Time (s)');
% We set the y-limits/range on our spectragram to be within the audible
% range of the fundamental frequencies of an acoustic guitar.
set(gca,'ylim',[100, 500],'Fontsize', 14);


%% Part #2 - Explore Pink Floyd - Comfortably Numb (Bass)
close all;
time_step = 0.1;  % Modify time step here.
tau_floyd = 0:time_step:L_floyd;
a_floyd_bass = 100;  % Modify width here.

ygt_spec_floyd_bass_gabor = zeros(length(tau_floyd), n_floyd);
ygt_spec_floyd_bass = zeros(length(tau_floyd), n_floyd);
% Save the bass notes for Part #3.
floyd_bass_notes = zeros(1, length(tau_floyd));  
% Frequencies determined by looking at spectragram.
fund_freq_bass = [82.407, 92.499, 97.999, 110.0, 123.47];  
for j = 1:length(tau_floyd)  
    % Using a Gaussian Window Filter.
    g_bass = exp(-a_floyd_bass*(t_floyd-tau_floyd(j)).^2);
    yg_bass = g_bass.*y_floyd;
    ygt_bass = fft(yg_bass);
    ygt_spec_floyd_bass_gabor(j, :) = abs(fftshift(ygt_bass));
    
    % Find maximum frequency for each time slice - this is the note played.
    [max_val, i] = max(abs(ygt_bass));
    % Make sure note is within first harmonic to a certain degree of error.
    err = 3;
    music_note = get_fundamental_freq(k_floyd(i), fund_freq_bass, err);
    if (music_note == 0)
        % Not perfect - but if get_fundamental_freq doesn't return we set
        % the music note for this time slice of this bass to be the
        % previous note.
        music_note = floyd_bass_notes(j-1);
    end
    % We know it is this note for the first few based on the spectragram.
    if (j == 1 || j == 2)
        % Hacky.
        music_note = fund_freq_bass(end);
    end
    floyd_bass_notes(j) = music_note;
    
    % Use a gaussian filter to filter around the fundamental frequency of 
    % the bass note we determined above.
    gaussian_tau = 0.1;
    gaussian_filter = exp(-gaussian_tau*(k_floyd-music_note).^2);
    ygt_bass_filtered = gaussian_filter.*ygt_bass;
    ygt_spec_floyd_bass(j, :) = abs(fftshift(ygt_bass_filtered));
end

% Plot spectragram (Gabor transform with Gaussian window).
figure();
pcolor(tau_floyd, ks_floyd, ygt_spec_floyd_bass_gabor.');
shading interp;
title({['Pink Floyd - Comfortably Numb' ] 
    ['Spectragram (Bass) - Gabor Transform with Gaussian Window']
    ['a = ' num2str(a_floyd_bass) ' and \Delta\tau = ' num2str(time_step)]
    });
colormap(hot);
ylabel('Frequency (Hz)');  xlabel('Time (s)');
set(gca,'ylim',[0, 1400],'Fontsize', 14);

% Plot the Spectragram of filtered bass line.
figure();
pcolor(tau_floyd, ks_floyd, ygt_spec_floyd_bass.');
shading interp;
title({['Pink Floyd - Comfortably Numb' ] 
    ['Spectragram (Bass) - Clean']
    });
colormap(hot);
ylabel('Musical Note and Frequency (Hz)');  xlabel('Time (s)');
yticks([82.407, 92.499, 97.999, 110.0, 123.47]);
yticklabels({'E - 82.407', 'G♭ - 92.499', 'G - 97.999', 'A - 110.0', 'B - 123.47'});
yline(82.407, 'w'); yline(92.499, 'w'); yline(97.999, 'w'); yline(110.0, 'w'); yline(123.47, 'w');
% We set the y-limits/range on our spectragram to be within the audible
% range of the fundamental frequencies of a bass.
set(gca,'ylim', [60, 160],'Fontsize', 14);


%% Part #3 - Explore Pink Floyd - Comfortably Numb (Guitar)
a_floyd_guitar = 1000;  % Modify width here, we use the same time_step in Part #2.
ygt_spec_floyd_guitar_gabor = zeros(length(tau_floyd), n_floyd);
ygt_spec_floyd_guitar = zeros(length(tau_floyd), n_floyd);
floyd_guitar_notes = zeros(1, length(tau_floyd));  % Save the guitar notes.

% First we want to remove the bass line from the song along with its
% overtones from the data.
freq_dat = fftshift(fft(y_floyd));  % Transform to freq space.
err = 3;  % Specified error to remove frequency.
for j = 1:length(floyd_bass_notes)
    freq = floyd_bass_notes(j);  % Bass note to remove.
    if (freq == 0)
        continue;
    end
    % First harmonic.
    freq_dat(abs(freq_dat) <= (freq - err) & abs(freq_dat) >= (freq + err)) = 0;
    % Second harmonic.
    freq_dat(abs(freq_dat) <= (2*freq - err) & abs(freq_dat) >= (2*freq + err)) = 0;
    % Third harmonic.
    freq_dat(abs(freq_dat) <= (3*freq - err) & abs(freq_dat) >= (3*freq + err)) = 0;
    % Fourth harmonic.
    freq_dat(abs(freq_dat) <= (4*freq - err) & abs(freq_dat) >= (4*freq + err)) = 0;
    % Fifth harmonic.
    freq_dat(abs(freq_dat) <= (5*freq - err) & abs(freq_dat) >= (5*freq + err)) = 0;
    % Sixth harmonic.
    freq_dat(abs(freq_dat) <= (6*freq - err) & abs(freq_dat) >= (6*freq + err)) = 0;
end
% Transform back to time space to do Gabor.
filtered_floyd_dat = ifft(ifftshift(freq_dat));

% Do Gabor + Gaussian filter on cleaned data.
for j = 1:length(tau_floyd)
    g_guitar = exp(-a_floyd_guitar*(t_floyd-tau_floyd(j)).^2);
    yg_guitar = g_guitar.*filtered_floyd_dat;
    ygt_guitar = fft(yg_guitar);
    ygt_spec_floyd_guitar_gabor(j, :) = abs(fftshift(ygt_guitar));

    % Find maximum frequency for each time slice - this is the note played.
    [~, i] = max(abs(ygt_guitar));
    floyd_guitar_notes(j) = abs(k_floyd(i));
    
    % Use a gaussian filter to filter around the maximum frequency.
    gaussian_tau = 0.1;
    gaussian_filter = exp(-gaussian_tau*(k_floyd-floyd_guitar_notes(j)).^2);
    ygt_guitar_filtered = gaussian_filter.*ygt_guitar;    
    ygt_spec_floyd_guitar(j, :) = abs(fftshift(ygt_guitar_filtered));
end

% Plot spectragram (Gabor transform with Gaussian window).
figure();
pcolor(tau_floyd, ks_floyd, ygt_spec_floyd_bass_gabor.');
shading interp;
title({['Pink Floyd - Comfortably Number' ] 
    ['Spectragram (Guitar) - Gabor Transform with Gaussian Window']
    ['a = ' num2str(a_floyd_guitar) ' and \Delta\tau = ' num2str(time_step)]    
    });
colormap(hot);
ylabel('Frequency (Hz)');  xlabel('Time (s)');
set(gca,'ylim',[300, 1000],'Fontsize', 14);

% Plot the Spectragram of filtered guitar.
figure();
pcolor(tau_floyd, ks_floyd, ygt_spec_floyd_guitar.');
shading interp;
title({['Pink Floyd - Comfortably Number' ] 
    ['Spectragram (Guitar) - Clean']
    });
colormap(hot);
ylabel('Musical Note and Frequency (Hz)');  xlabel('Time (s)');
yticks([329.63, 369.99, 493.88, 587.33]);
yticklabels({'E - 329.63', 'G♭ - 369.99', 'B - 493.88', 'D - 587.33'});
yline(329.63, 'w'); yline(369.99, 'w'); yline(587.33, 'w'); yline(493.88, 'w');
set(gca,'ylim',[300, 700],'Fontsize', 14);


%% Bonus: Play Pink Floyd Sample Without Bass Notes
p8_floyd_guitar = audioplayer(filtered_floyd_dat, Fs_floyd); 
playblocking(p8_floyd_guitar);


