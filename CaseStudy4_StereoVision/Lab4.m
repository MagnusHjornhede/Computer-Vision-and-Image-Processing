% Load the stereoParameters object (calibration data)
load('stereo_camera_calibration_200.mat');  % stereoParams contains the calibration parameters

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

% Loop through each frame in the video
while hasFrame(readerLeft) && hasFrame(readerRight)
    % Read the current frames from both the left and right videos
    frameLeft = readFrame(readerLeft);
    frameRight = readFrame(readerRight);
    
    % Rectify the frames using the stereo parameters
    [frameLeftRect, frameRightRect, reprojectionMatrix] = ...
        rectifyStereoImages(frameLeft, frameRight, stereoParams);

    % Display the rectified frames as a stereo anaglyph (optional)
    figure;
    imshow(stereoAnaglyph(frameLeftRect, frameRightRect));
    title('Rectified Video Frames');

    % Convert the rectified frames to grayscale for disparity calculation
    frameLeftGray = im2gray(frameLeftRect);
    frameRightGray = im2gray(frameRightRect);
    
    % Compute the disparity map using Semi-Global Matching (SGM)
    disparityMap = disparitySGM(frameLeftGray, frameRightGray, 'DisparityRange', [0, 64]);

    % Display the disparity map
    figure;
    imshow(disparityMap, [0, 64]);
    title('Disparity Map');
    colormap jet;
    colorbar;

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

    % Visualize the point cloud
    view(player3D, ptCloud);

    % Detect people in the left rectified image (if needed)
    peopleDetector = peopleDetectorACF();
    bboxes = detect(peopleDetector, frameLeftGray);

    if ~isempty(bboxes)
        % Find the centroids of detected people
        centroids = [round(bboxes(:, 1) + bboxes(:, 3) / 2), ...
                     round(bboxes(:, 2) + bboxes(:, 4) / 2)];

        % Find the 3D world coordinates of the centroids
        centroidsIdx = sub2ind(size(disparityMap), centroids(:, 2), centroids(:, 1));
        X = points3D(:, 1);
        Y = points3D(:, 2);
        Z = points3D(:, 3);
        centroids3D = [X(centroidsIdx), Y(centroidsIdx), Z(centroidsIdx)];

        % Find the distances from the camera to each centroid
        dists = sqrt(sum(centroids3D .^ 2, 2));

        % Display the detected people and their distances
        labels = dists + " meters";
        annotatedFrame = insertObjectAnnotation(frameLeftRect, 'rectangle', bboxes, labels);
        
        % Display the annotated frame
        figure;
        imshow(annotatedFrame);
        title('Detected People and Distances');
    end
    
    % Display the current frame with any detections
    step(player, frameLeftRect);  % Display the left rectified frame
end

% Release resources when done
release(player);
release(player3D);
