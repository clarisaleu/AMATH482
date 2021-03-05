%{ 
    Helper function to gather the time series data of the paint can (i.e. the
    mass), given the video data of the spring-mass system.

    Takes in the video frames (vid_frames) data, a function to crop the
    video as to remove any background "noise", a gray_scale_val to filter
    the video by (value from 0 to 255) as to isolate the light on the paint can,
    and a boolean value play_vid where if asserted - plays the passed in
    video as well as shows the gray scale filtering/crop function in action.
    
    Outputs a matrix (which is of size num_frames by 2) containing the (x, y) 
    coordinates of where we suspect the paint can to be located - 
    based on the given crop function and gray scale value used to filter each frame by.

    * Note, you will need the image processing toolbox in MATLAB to use this
    function. *
%}

function [time_series_dat] = get_time_series_dat(vid_frames, ...
                                crop_vid_func, gray_scale_val, play_vid)
    num_frames = size (vid_frames, 4);
    time_series_dat = zeros(num_frames, 2);
    for i = 1:num_frames
        % For each frame, construct a struct with two fields:
        %   - cdata which is the image data stored as an array of uint8 values,
        %   - colormap which will be empty as the video frames contain
        %     true (RGB) images. This is so we can use the function
        %     frame2im - which will return the image data associated with the
        %     passed in the video frame.
        frame_dat.cdata = vid_frames(:, :, :, i);
        frame_dat.colormap = [];
        image_dat = frame2im(frame_dat);  % Get image data for frame.
        
        % Convert to grayscale as it is easier to track the light on
        % top of the paint can (i.e. - we set a threshold for grayscale values
        % to let in where there are 256 different shades of gray with 0 being
        % black and 255 being white).
        image_dat_bw = rgb2gray(image_dat);
        
        % Pick threshold which looks best at picking out the light on top
        % of the paint can. In addition, crop the background of the video
        % with passed in filter function.
        cropped_dat = double(image_dat_bw).*crop_vid_func;
        filtered_dat = cropped_dat > gray_scale_val;
        
        % Find coordinates of bright spots - average them and set return
        % value which corresponds to where we suspect the can to be for this 
        % slice in time. Averaging should be okay here if crop function
        % correctly crops out any background light reflected by the chair,
        % etc.
        indices = find (filtered_dat);
        [Y, X] = ind2sub(size(filtered_dat), indices);
        time_series_dat(i, 1) = mean(X);
        time_series_dat(i, 2) = mean(Y);
        
        % Play video if specified.
        if (play_vid)
            % Original video.
            subplot(2,2,1);
            imshow(uint8(frame_dat.cdata)); drawnow;
            
            % Black and white video.
            subplot(2,2,2);
            imshow(uint8(image_dat_bw)); drawnow;
            
            % Black and white video with crop.
            subplot(2,2,3);
            imshow(uint8(cropped_dat)); drawnow;
            
            % Black and white video with crop and filtered gray scale values.
            % filtered_dat consists of ones and zeros, where values greater 
            % than gray_scale_val will be one & values less than or equal 
            % to gray_scale_val will be zero. We multiply by 255 to amplify
            % these values and make them white.
            subplot(2,2,4);
            imshow(uint8(255 * filtered_dat)); drawnow;
            hold on;
            % Show time point used in PCA.
            plot(mean(X),mean(Y),'r*')
            hold off;
        end
    end
    
    if (play_vid)
        % Close video at end.
        close all;
    end
end
