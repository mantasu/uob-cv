%% Lab 3: Hough Transform

close all; 
clear all;

%% First we will read the image and find edges
% Read image
I = imread('cluttera2.jpg');

% The image is RGB and we need to convert to grayscale
I = rgb2gray(I);

% Apply your favourite edge detector: Here I am using Built-in Canny
BW = edge(I,'canny');

% Calculate the Hough Transform
[H,T,R] = hough(BW);

% Show original image and the calculated edges
figure;
subplot(1,2,1)
imshow(I);
title('original');

subplot(1,2,2)
imshow(BW);
title('Edges');

%% Hough Transform: Will will use the the built in function
% Calculate Hough Peaks (number of votes per line as in Lecture
P  = houghpeaks(H,10,'threshold',ceil(0.3*max(H(:))));
x = T(P(:,2));
y = R(P(:,1));

% show the Hough Map and the identified peaks (features)
figure;
imshow(H,[],'XData',T,'YData',R,'InitialMagnification','fit');
xlabel('\theta'), ylabel('\rho');
axis on, axis normal, hold on;
title('Hough Transform');
plot(x,y,'s','color','red');

%% Now we will find the lines
% Find lines and plot them
lines = houghlines(BW,T,R,P,'FillGap',5,'MinLength',7);

figure;
imshow(I,[]);
hold on
max_len = 0;
for k = 1:length(lines)
    xy = [lines(k).point1; lines(k).point2];
    plot(xy(:,1),xy(:,2),'LineWidth',2,'Color','green');

    % plot beginnings and ends of lines
    plot(xy(1,1),xy(1,2),'x','LineWidth',2,'Color','yellow');
    plot(xy(2,1),xy(2,2),'x','LineWidth',2,'Color','red');

    % determine the endpoints of the longest line segment
    len = norm(lines(k).point1 - lines(k).point2);
    if ( len > max_len)
        max_len = len;
        xy_long = xy;
    end
end

% highlight the longest line segment
plot(xy_long(:,1),xy_long(:,2),'LineWidth',2,'Color','cyan');
