<style>
  .{
    background-color: black;
  }
  h1, h2, h3 {
    margin-top: 1em;
    font-weight: bold;
    color: #ddffff;
  }
  strong {
    color: lightblue;
  }
  em {
    color: #aaccff;
  }
</style>

# Visual Perception

## Computational Vision

**Computational Vision** - the process of discovering from images what is present in the world and where. It is _challenging_ because it tries to recover lost information when reversing the imaging process (imperfect imaging prcess, discretized continuous world)

**Applications**:
* Automated navigation with obstacle avoidance
* Scene recognition and target detection
* Document processing
* Human computer interfaces

## Human Vision

Captured photons release energy which is used as electrical signals for us to see. Pupils dilate to accept as much light as needed to see clearly. _Rod_ cells (~`120m`) are responsible for vision at low light leves, _cones_ (~`6m`) are active at higher levels and are capable of color vision and have high spacial acuity. There's a `1-1` relationship of _cones_ and neurons so we are abe to resolve better, meanwhile many _rods_ converge to one neuron.

**Receptive field** - area on which light must fall for neuron to be stimulated. A **receptive field** of a **Ganglion cell**  is formed from all _photoreceptors_ that synapse with it.

**Ganglion cells** - cells located in retina that process visual information that begins as light entering the eye and transmit it to the brain. There are `2` types of them.
* _On-center_ - fire when light is on centre
* _Off-center_ - fire when light is around centre

**Ganglion cells** allow transmition of information about contrast. The size of the **receptive field** controls the spatial frequency information (e.g., small ones are stimulated by high frequencies for fine detail). **Ganglion cells** don't work binary (fire/not fire), the change the firing rate when there's light.

**Trichromatic coding** - any color can be reproduced using 3 primary colors (red, blue, green).

> Retina contains only 8% of blue _cones_ and equal proportion of red and green ones - they allow to discriminate colors `2nm` in difference and allow to match multiple wavelenghts to a single color (does not include blending though)

## Maths

Wave frequency $f$ (`Hz`) and energy $E$ (`J`) can be calculated as follows ($h=6.623\times 10^{34}$ - Plank's constant, $c=2.998\times 10^8$ - speed of light):

$$f=\frac{c}{\lambda}$$

$$E=hf$$

> Perceivable electromagnetic radiadiation wavelengths are within `380` to `760` nm

**Focal length** $f$ (`m`) - distance from lens to the point $F$ where the system converges the light. The power of lens (how much the lens reduces the real world to the image in plane) is just $\frac{1}{f}$ (`D`) (~`59D` for human eye)

![Lens Formula and Magnification](https://s3.amazonaws.com/bucketeer-6a6b5dd7-82e9-48dd-b3be-ec23fe6cc180/notes/images/000/000/060/original/lens-formula.jpg?1583773717)

> If the image plane is curved (e.g., back of an eye), then as the angle from optical centre to real world object $\tan\theta=\frac{h}{u}=\frac{h'}{v}$ gets larger (when it gets closer to the lens), it is approximated worse.


# Edge Detection and Filters
## Edge Detection
**Intensity image** - a matrix whose values correspond to pixels with intensities within some range. (`imagesc` in _matlab_ displays intensity image).

**Colormap** - a matrix which maps every pixel to usually (most popular are _RGB_ maps) `3` values to represent a single color. They're averaged to convert it to intensity value.

**Image** - a function $f(x, y)$ mapping coordinates to intensity. We care about the rate of change in intensity in `x` and `y` directions - this is captured by the _gradien of intensity_ Jacobian vector:

$$\nabla f(x, y)=\begin{pmatrix}{\partial f} / {\partial x} \\ {\partial f} / {\partial y}\end{pmatrix}$$

Such gradient has `x` and `y` component, thus it has _magnitude_ and _direction_:

$$M(\nabla f)=\sqrt{(\nabla_xf)^2+(\nabla_yf)^2}$$

$$\alpha=\tan^{-1}\left(\frac{\nabla_yf}{\nabla_xf}\right)$$

> Edge detection is useful for feature extraction for recognition algorithms

## Detection Process

Edge dectiptors:
* **Edge direction** - perpendicular to the direction of the highest change in intensity (to _edge normal_)
* **Edge strength** - contrast along the normal
* **Edge position** - image position where the edge is located

Edge detection steps:
1. **Smoothening** - remove noise
2. **Enhancement** - apply differentiation
3. **Thresholding** - determine how intense are edges
4. **Localization** - determine edge location

Optimal edge detection criteria:
* _Detecting_ - minimize false positives and false negatives
* _Localising_ - detected edges should be close to true edges
* _Single responing_ - minimize local maxima around the true edge

## First order Edge Filters

To approximate the gradient at the centre point of `2`-by-`2` pixel area, for change in `x` - we sum the differences between values in rows; for change in `y` - the differences between column values. We can achieve the same by summing weighted pixels with horizontal and vertical weight matrices (by applying _cross-correlation_):

$$W_x=\begin{bmatrix}-1 & 1 \\ -1 & 1\end{bmatrix};\ W_y=\begin{bmatrix}1 & -1 \\ 1 & -1\end{bmatrix}$$

There are other ways to approximate the gradient (not necessarily at `2`-by-`2` regions): _Roberts_ and _Sobel_ (very popular):

$$W_x=\begin{bmatrix}1 & 0 \\ 0 & -1\end{bmatrix};\ W_y=\begin{bmatrix}0 & -1 \\ 1 & 0\end{bmatrix}$$

$$W_x=\begin{bmatrix}-1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1\end{bmatrix};\ W_y=\begin{bmatrix}1 & 2 & 1 \\ 0 & 0 & 0 \\ -1 & -2 & -1\end{bmatrix}$$

After applying filters to compute the gradient matrices $G_x$ and $G_y$, we can calculate the magnitude between the `2` to get the final output: $M=\sqrt{G_x^2 + G_y^2}$. Sometimes it's approximated by magnitude values.

Note that given an **intensity image** $I$ of dimensions $H\times W$ and a filter (kernel) $K$ of dimensions $(2N+1)\times (2M+1)$ the _cross-correlation_ at pixel $h, w$ is expressed as (odd-by-odd filters are more common as we can superimpose the maps onto the original images we want to compare):

$$(I\otimes K)_{h,w}=\sum_{n=-N}^N\sum_{m=-M}^MI_{h+n,w+m}K_{N+n,M+m}$$

> We usually set a threshold for calculated gradient map to distinguish where the edge is.

If we use noise smoothing filter, instead of looking for an image gradient after applying the noise filter, we can take the derivative of the noise filter and then convolve it (because mathematically it's the same):

$$\nabla_x(h\star f)=(\nabla_xh)\star f$$

## Second order Edge Filters

> We can apply **Laplacian Operator** - by applying the second derivative we can identify where the rate of change in intensity crosses `0`, which shows exact edge.

**Laplacian** - sum of second order derivatives (dot product):

$$\nabla f=\nabla^2 f=\nabla \cdot \nabla f$$

$$\nabla^2I=\frac{\partial^2I}{\partial x^2}+\frac{\partial^2I}{\partial y^2}$$

For a finite difference approximation, we need a filter that is at least the size of `3`-by-`3`. For change in `x`, we take the difference between the differences involving the center and adjacent pixels for that row, for change in `y` - involving centre and adjacent pixels in that column. I. e., in `3`-by-`3` case:

$$(\nabla_{x^2}^2I)_{h,w}=(I_{h,w+1}-I_{h,w}) - (I_{h,w} - I_{h,w-1})=I_{h,w-1}-2I_{h,w}+I_{h,w+1}$$

$$(\nabla_{y^2}^2I)_{h,w}=I_{h-1,w}-2I_{h,w}+I_{h+1,w}$$

We just add the double derivative matrices together to get a final output. Again, we can calculate weights for these easily to represent the whole process as a _cross-correlation_ (a more popular one is the one that accounts for diagonal edges):

$$W=\begin{bmatrix}0 & 1 & 0 \\ 1 & -4 & 1 \\ 0 & 1 & 0\end{bmatrix};\ W=\begin{bmatrix}1 & 4 & 1 \\ 4 & -20 & 4 \\ 1 & 4 & 1\end{bmatrix}$$

## Noise Removal
We use a uniform filter (e.g., in `3`-by-`3` case all filter values are $\frac{1}{9}$) to average random noise - the bigger the filter, the more details we lose but the less noise the image has due to its smoothness. More popular filters are _Gaussian_ filters with more weight on middle points. _Gaussian_ filter can be generated as follows:

$$H_{ij}= \frac{1}{2\pi\sigma^2}\exp \left(-\frac{(i-(k+1))^2+(j-(k+1))^2}{2\sigma^2} \right) ; 1 \leq i, j \leq (2k + 1)$$

> **Laplacian of Gaussian** - _Laplassian_ + _Gaussian_ which smoothens the image (necessary before _Laplassian_ operation) with _Gaussian_ filter and calculates the edge with **Laplassian Operator**

Note the noise supression-localization tradeoff: larger mask size reduces noise but adds uncertainty to edge location. Also note that the smoothness for _Gaussian_ filters depends on $\sigma$.

## Canny Edge Detector

> Canny has shown that the first derivative of the _Gaussian_ provides an operator that optimizes signal-to-noise ratio and localiztion

Algorithm:
1. Compute the image gradients $\nabla_x f = f * \nabla_xG$ and $\nabla_y f = f * \nabla_yG$ where $G$ is the _Gaussian_ function of which the kernels can be found by:
    * $\nabla_xG(x, y)=-\frac{x}{\sigma^2}G(x, y)$
    * $\nabla_yG(x, y)=-\frac{y}{\sigma^2}G(x, y)$
2. Compute image gradient _magnitude_ and _direction_
3. Apply **non-maxima** suppression
4. Apply **hysteresis** thresholding

**Non-maxima suppression** - checking if gradient magnitude at a pixel location along the gradient direction (_edge normal_) is local maximum and setting to `0` if it is not

**Hysteresis thresholding** - a method that uses `2` threshold values, one for certain edges and one for certain non-edges (usually $t_h = 2t_l$). Any value that falls in between is considered as an edge if neighboring pixels are a strong edge.

> For **edge linking** high thresholding is used to start curves and low to continue them. Edge direction is also utilized.

# Scale Invariant Feature Transform

**SIFT** - an algorithm to detect and match the local features in images

**Invariance Types** (and how to achieve them):
* **Illumination** - luminosity changes
    * Difference based metrics
* **Scale** - image size change
    * _Pyramids_ - average pooling with stride `2` multiple times
    * _Scale Space_ - apply _Pyramids_ but take **DOG**s (_Differences of Gaussians_) in between and keep features that are repeatedly persistent
* **Rotation** - roll the image along the `x` axis
    * Rotate to most dominant gradient direction (found by histograms)
* **Affine** -
* **Perspective** - 

# Questions
1. **Introduction**
    * Why is computational vision challenging? Some applications.