function gaussian_map_2_filters = apply_2_1d_gaussians(filter1, filter2, image)
gaussian_map_2_filters = conv2(filter1, filter2, image);
end