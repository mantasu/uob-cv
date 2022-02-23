function m = magnitude(x, y)
%MAGNITUDE Computes the magnitude of 2 gradient matrices
%   This method takes in 2 gradient matrices (in x and y directions) and
%   computes the magnitude vector (matrix): the square root of the sum of
%   the 2 squared matrices.
m = sqrt(x.^2 + y.^2);
end

