% Load the stereoParameters object (calibration data)
% load('stereo_camera_calibration_200.mat');  % stereoParams contains the calibration parameters
load('stereo_calibration_params_with_Q.mat');



% Define the video files for the left and right cameras
videoFileLeft = 'left_camera_video.avi';
videoFileRight = 'right_camera_video.avi';

% Create VideoReader objects for both left and right videos
readerLeft = VideoReader(videoFileLeft);
readerRight = VideoReader(videoFileRight);

% Create a video player for displaying the results
player = vision.VideoPlayer('Position', [20, 200, 740, 560]);

% Create a 3D point cloud player for visualizing 3D points
player3D = pcplayer([-3, 3], [-3, 3], [0, 8], 'VerticalAxis', 'y', 'VerticalAxisDir', 'down');

% Initialize frame counter
frameCounter = 0;

% Loop through each frame in the video
while hasFrame(readerLeft) && hasFrame(readerRight) && isOpen(player3D)
    frameCounter = frameCounter + 1;
    
    % Read the current frames from both the left and right videos
    frameLeft = readFrame(readerLeft);
    frameRight = readFrame(readerRight);
    
    % Rectify the frames using the stereo parameters
    [frameLeftRect, frameRightRect, reprojectionMatrix] = ...
        rectifyStereoImages(frameLeft, frameRight, stereoParams);

    % Convert the rectified frames to grayscale for disparity calculation
    frameLeftGray = im2gray(frameLeftRect);
    frameRightGray = im2gray(frameRightRect);
    
    % Compute the disparity map using Semi-Global Matching (SGM)
    disparityMap = disparitySGM(frameLeftGray, frameRightGray, 'DisparityRange', [0, 64]);

    % Reconstruct the 3D scene from the disparity map
    points3D = reconstructScene(disparityMap, reprojectionMatrix);
    points3D = points3D ./ 1000;  % Convert from millimeters to meters

    % Reshape and filter valid 3D points for point cloud creation
    [rows, cols, ~] = size(points3D);
    points3D = reshape(points3D, [], 3);  % Flatten the 3D points to a Nx3 matrix
    validIdx = isfinite(points3D(:, 1)) & isfinite(points3D(:, 2)) & isfinite(points3D(:, 3));
    points3D = points3D(validIdx, :);

    % Create the point cloud with color information from the left rectified frame
    colorImage = reshape(frameLeftRect, [], 3);  % Flatten the color image
    colorImage = colorImage(validIdx, :);  % Filter colors based on valid 3D points
    ptCloud = pointCloud(points3D, 'Color', colorImage);

    % Visualize the point cloud for every 10th frame
    if mod(frameCounter, 10) == 0
        view(player3D, ptCloud);
    end

    % Detect people in the left rectified image (if needed)
    peopleDetector = peopleDetectorACF();
    bboxes = detect(peopleDetector, frameLeftGray);

    if ~isempty(bboxes)
        % Find the centroids of detected people
        centroids = [round(bboxes(:, 1) + bboxes(:, 3) / 2), ...
                     round(bboxes(:, 2) + bboxes(:, 4) / 2)];

        % Ensure the centroids are within the valid bounds of the disparity map
        [rows, cols] = size(disparityMap);  % Get disparity map dimensions
        centroids(:, 1) = min(max(centroids(:, 1), 1), cols);  % Clamp x-coordinates
        centroids(:, 2) = min(max(centroids(:, 2), 1), rows);  % Clamp y-coordinates

        % Find the 3D world coordinates of the centroids
        centroidsIdx = sub2ind([rows, cols], centroids(:, 2), centroids(:, 1));  % Convert to linear indices
        X = points3D(:, 1);
        Y = points3D(:, 2);
        Z = points3D(:, 3);

        % Make sure centroidsIdx is within bounds
        centroidsIdx = min(centroidsIdx, length(X));  % Ensure the indices are within bounds
        centroids3D = [X(centroidsIdx), Y(centroidsIdx), Z(centroidsIdx)];

        % Find the distances from the camera to each centroid
        dists = sqrt(sum(centroids3D .^ 2, 2));

        % Display the detected people and their distances every 10th frame
        if mod(frameCounter, 10) == 0
            labels = dists + " meters";
            annotatedFrame = insertObjectAnnotation(frameLeftRect, 'rectangle', bboxes, labels);
            figure;
            imshow(annotatedFrame);
            title('Detected People and Distances');
        end
    end

    % Display the current frame with any detections every 10th frame
    if mod(frameCounter, 10) == 0
        step(player, frameLeftRect);  % Display the left rectified frame
    end
end

% Release resources for the video player
release(player);
