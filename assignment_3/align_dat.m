%{ 
    Helper function where given positional data from three different
    cameras - crops each time series data as to be better aligned in time with
    eachother, wrt. to the oscillation they are all capturing.
    In addition - in order to perform PCA, each matrix must have the same 
    dimensions - so we determine the matrix with the smallest number of
    frames and trim the other two to that size.
    Inputs are the three time series we wish to align in time and crop and
    the number of frames to find the lowest y-coordinate within in order to 
    try and align the data within the same oscillation.
    Outputs all the time series combined in order to perform PCA.
%}

function [cropped_dat] = align_dat (time_series_dat_1, time_series_dat_2, time_series_dat_3, num)
    [~, i] = min(time_series_dat_1(1:num, 2));
    time_series_dat_1  = time_series_dat_1(i:end, :);

    [~, i] = min(time_series_dat_2(1:num, 2));
    time_series_dat_2  = time_series_dat_2(i:end, :);

    [~, i] = min(time_series_dat_3(1:num, 2));
    time_series_dat_3  = time_series_dat_3(i:end, :);
    
    % Find minimum frame length and resize.
    min_frame = length(time_series_dat_1);
    if (length(time_series_dat_2) < min_frame)
        min_frame = length(time_series_dat_2);
    end
    if (length(time_series_dat_3) < min_frame)
        min_frame = length(time_series_dat_3);
    end
    
    % Set return value.
    cropped_dat = [time_series_dat_1(1:min_frame, :)'; ... 
        time_series_dat_2(1:min_frame, :)'; time_series_dat_3(1:min_frame, :)'];
end
