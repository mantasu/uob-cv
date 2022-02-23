function m_a = magnitude_approximation(x, y)
%MAGNITUDE_APPROXIMATION Adds up 2 absolute matrices.
%   This method take in 2 gradient matrices in 2 directions and
%   approximates their magnitude by taking the absolute versions of each of
%   the matrices and summing them together.
m_a = abs(x) + abs(y);
end

