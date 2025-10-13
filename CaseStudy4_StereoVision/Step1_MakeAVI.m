% Paths to the stereo image datasets 
imageDirCam0 = 'D:\AI\LP5\R7020E Computer Vision and Image Processing\Lab\Dpth estimation dataset\MH_01_Easy_cam\cam0';
imageDirCam1 = 'D:\AI\LP5\R7020E Computer Vision and Image Processing\Lab\Dpth estimation dataset\MH_01_Easy_cam\cam1';

% Load the image files
fileNamesCam0 = dir(fullfile(imageDirCam0, '*.png'));
fileNamesCam1 = dir(fullfile(imageDirCam1, '*.png'));

% Get the number of images in each directory
numImagesCam0 = length(fileNamesCam0);
numImagesCam1 = length(fileNamesCam1);

% Ensure both directories have the same number of images
if numImagesCam0 ~= numImagesCam1
    warning('The number of images in cam0 and cam1 directories do not match.');
    
    % If cam0 has more images, remove the extra ones
    if numImagesCam0 > numImagesCam1
        fileNamesCam0 = fileNamesCam0(1:numImagesCam1);
    else
        % If cam1 has more images, remove the extra ones
        fileNamesCam1 = fileNamesCam1(1:numImagesCam0);
    end
end

% Now, the number of images should match
numImages = min(numImagesCam0, numImagesCam1);

% Set up video writers for left and right videos
outputVideoLeft = VideoWriter('left_camera_video.avi');
outputVideoRight = VideoWriter('right_camera_video.avi');

outputVideoLeft.FrameRate = 10;  % Set frame rate for left camera video
outputVideoRight.FrameRate = 10; % Set frame rate for right camera video

% Open the video writers
open(outputVideoLeft);
open(outputVideoRight);

% Loop through the images and add them to the video files
for i = 1:numImages
    % Read the left and right stereo images
    frameLeft = imread(fullfile(imageDirCam0, fileNamesCam0(i).name));
    frameRight = imread(fullfile(imageDirCam1, fileNamesCam1(i).name));

    % Write each frame to the video
    writeVideo(outputVideoLeft, frameLeft);
    writeVideo(outputVideoRight, frameRight);
end

% Close the video writers
close(outputVideoLeft);
close(outputVideoRight);

disp('AVI files created: left_camera_video.avi and right_camera_video.avi');
