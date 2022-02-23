<h1 align="center">Computer Vision Notes and Exercises</h1>
<p align="center"><b>Computational Vision & Imaging</b></p>
<p align="center"><i>University of Birmingham - Spring 2022</i></p>

## Overview
This repository contains my notes and lab exercises for the _Computational Vision & Imaging_ module provided by the University of Birmingham. Lab exercises contain **MATLAB** code which is also rendered as **HTML** document for each lab session for visual reasons. This is my personal practice of writing **LaTeX** documents and coding in **MATLAB**.

> **Note**: `notes.md` file should be checked with an enhanced markdown previewer, such as VS code (using extension [Markdown All in One](https://marketplace.visualstudio.com/items?itemName=yzhang.markdown-all-in-one)) for colors and latex equations.

## Solutions
### Labs
* [Lab 1](lab1/matlab.html): Edge Detectors. (_Done_)
* [Lab 2](lab2/matlab.html): Noise Removal. (_Done_)

## Examples
<details><summary><b>Lab question example</b></summary>
<br>

**Question**: what happens to the edges in the heavily blurred case after applying gradient operators, such as _Sobel_ operators?
<br>

**Answer**: thicker edges stand out more and thinner edges almost disappear. This is because thin edges may sometimes be confused with noise which gets averaged out after heavily blurying an image. Strong edges, on the other hand, remain standing out (i.e., they are surrounded by more values indicating an edge region) therefore the gradient operator captures the changes and emphasizes them.

</details>

<details><summary><b>Lab code example</b></summary>
<br>

**Task**: combine the idea of the _Laplacian_ operator with the idea of _Gaussian_ smoothing.
<br>

```matlab
% Get LoG filter, apply to image and apply zerocross
log_filter = conv2(laplacian, gaussian_filter_5x5);
I_in = conv2(log_filter, shakey);
I_out = edge(I_in, 'zerocross');

% Show the result
show_image(I_out, "log_zerocross")
```

</details>
