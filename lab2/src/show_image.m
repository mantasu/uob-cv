function show_image(img, name)
% SHOW_IMAGE Image

% Create a figure

set(gca,'xtick',[]);
set(gca,'ytick',[]);

% Get colormap and show
colormap(gray);
imagesc(img);

% Set proper axis
axis image;
axis off;
f = gcf;

% Save if needed
if nargin > 1
    savepath = strcat("figures/", name, ".jpg");
    exportgraphics(f, savepath, 'Resolution', 300);
end

end