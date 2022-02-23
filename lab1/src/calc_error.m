function [mean_abs_err, mean_rel_err] = calc_error(measured, expected)
%CALC_ERROR Returns the mean error
%   Calculates the mean absolute and relative errors between 2 matrices.

% Calculate the epsilon for stability
epsilon = ones(size(measured)) * 1e-8;

% Calculate the element-wise absolute and relative errors
absolute_err = abs(measured - expected);
relative_err = absolute_err ./ max(epsilon, abs(measured));

% Caclulate the mean absolute and relative errors
mean_abs_err = mean(absolute_err, "all");
mean_rel_err = mean(relative_err, "all");
end

