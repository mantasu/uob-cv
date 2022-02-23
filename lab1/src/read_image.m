function image = read_image(base_dir, file)

% read_image(base_dir, file) - reads a gif file from the directory base_dir
%                        base_dir should be a string, and if it is the
%                        current directory it should be ''
% It also converts the image from unit8 to double

image = imread([base_dir, file]);

image = double(image);