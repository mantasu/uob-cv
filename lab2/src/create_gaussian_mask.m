function gaussian_mask = create_gaussian_mask(mean, sigma, size, factor)
%GAUSSIAN_MASK Creates a Gaussian mask
%   Given a size of a square matrix, range factor, a mean and a standard
%   deviation, it produces a Gaussian mask filter.
if nargin < 4
    % Set default factor
    factor = 1;
end

% Create the range and create a Gaussian mask
range = [-floor(size/2):floor(size/2)] * factor;
gaussian_mask = N(mean, sigma, range)' * N(mean, sigma, range);

end

