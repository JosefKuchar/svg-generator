= Bezier representation

The vector output used throughout this work is based on a hierarchical
representation consisting of shapes, paths, and individual Bezier segments.
A shape corresponds to one filled graphical object and is assigned a single
RGB color and opacity value. Each shape contains one or more paths, and each
path consists of a sequence of cubic Bezier curves. In the implementation,
one curve is stored as a tuple
$((x_0, y_0), (x_1, y_1), (x_2, y_2), (x_3, y_3))$,
where $(x_0, y_0)$ is the start point, $(x_1, y_1)$ and $(x_2, y_2)$ are the
two control points, and $(x_3, y_3)$ is the endpoint. This convention is used
for geometric manipulation and SVG export.

For learning, the hierarchical SVG structure is converted into a flat sequence
of segment descriptors. Each segment is represented by a 13-dimensional vector
$ s = (x_0, y_0, x_1, y_1, x_2, y_2, r, g, b, alpha, f_p, f_s, f_r) $
where $(x_0, y_0)$ denotes the start point of the segment,
$(x_1, y_1)$ and $(x_2, y_2)$ are the control points,
$(r, g, b)$ is the color, $alpha$ is opacity,
$f_p$ indicates the beginning of a new SVG path element,
$f_s$ indicates the beginning of a new subpath within the current path,
and $f_r$ is a validity flag distinguishing real segments from padding.
The endpoint is omitted from the learned representation, because it is implied
by the start point of the following segment. For the last segment in a path,
the endpoint is defined by the start point of the first segment, which closes
the contour explicitly.

All quantities are normalized to the interval $[-1, 1]$. Let the original
raster image have width $W$ and height $H$. Coordinates are normalized with
respect to the image center
$ c_x = W / 2 quad c_y = H / 2 $
and the isotropic scale factor
$ lambda = 2 / max(W, H) $
The normalized coordinates are therefore
$
  tilde(x) = (x - c_x) lambda, quad
  tilde(y) = (y - c_y) lambda
$
This choice preserves aspect ratio and maps the larger image dimension to the
full interval $[-1, 1]$. Color channels originally stored in $[0, 255]$ are
mapped linearly to $[-1, 1]$, and opacity values from $[0, 1]$ are mapped by
the same affine transformation. Binary structural flags are represented as
$+1$ for true and $-1$ for false, which keeps every output dimension on a
common numerical scale.

Since the number of segments varies across examples, the model operates on a
fixed-length tensor of shape $(N, 13)$, where $N$ is the maximum number of
segments allowed for one sample. If an SVG contains fewer than $N$ segments,
the remaining rows are padded with zeros and their validity flag is set to
$-1$. If an SVG contains more than $N$ segments, the representation is
truncated to the first $N$ segments. The validity flag thus serves two
purposes: it masks padded positions during learning and enables the decoder to
ignore non-existent segments during reconstruction.

The inverse mapping reconstructs the hierarchical vector structure from the
predicted tensor. First, all rows with $f_r \leq 0$ are discarded. The
remaining normalized coordinates, colors, and opacity values are denormalized
back to image space and original attribute ranges. The flags $f_p$ and $f_s$
are thresholded at zero and used to determine whether a segment starts a new
shape or a new subpath. Because color and opacity are predicted per segment,
the final attribute of a reconstructed shape is obtained by averaging these
values over all its constituent segments. Finally, each path is closed by
connecting the endpoint of every segment to the start point of the next one,
with the last segment connected back to the first. This yields a compact
sequence representation that is convenient for neural prediction while still
preserving the topology required for valid SVG reconstruction.

= SVG Conversion to Bezier Representation

The source dataset contains SVG files whose graphical content may be expressed
using a heterogeneous set of primitives, transformations, and grouping
constructs. Before these data can be used for training, each SVG must be
converted into a uniform representation compatible with the tensor encoding
described above. The conversion procedure implemented in `parsing.py` therefore
maps every supported graphical element to a collection of filled cubic Bezier
paths together with a shared color and opacity.

The conversion begins with structural simplification. SVG files are first
processed externally in Inkscape in batch mode. During this stage, all objects
are converted to paths and strokes are expanded into filled outlines. This step
removes many forms of SVG variability and ensures that the subsequent parser
operates on explicit geometric contours rather than on higher-level drawing
commands. Several classes of samples are excluded before conversion, namely
SVGs containing gradient definitions, masks, or embedded style blocks. These
constructs are not supported by the present representation, which assumes a
single solid fill color and a scalar opacity for each shape.

After preprocessing, the SVG document is parsed recursively. Group and root
nodes are traversed until individual drawable shapes are reached. For each
shape, the fill color is extracted as an RGB triplet and opacity is read from
the `opacity` or `fill-opacity` attribute, with percentage values converted to
the unit interval. The fill rule is also preserved, because it determines the
topological interpretation of nested contours.

Each shape is then rewritten as a path object and reified so that all geometric
commands are made explicit. The resulting path is decomposed segment by
segment, and every segment is converted to cubic Bezier form. This conversion
is exact for native cubic Bezier segments. Straight lines and closing commands
are represented as degenerate cubic curves whose control points lie on the line
between the endpoints at one-third and two-thirds of its length. Quadratic
Bezier segments are elevated to cubic form by the standard degree-elevation
formula
$
  c_1 = p_0 + frac(2, 3) (q_1 - p_0), quad
  c_2 = p_2 + frac(2, 3) (q_1 - p_2)
$
where $p_0$ and $p_2$ are the original endpoints and $q_1$ is the quadratic
control point. Elliptic arcs are approximated by the parser library as one or
more cubic Bezier segments and are stored in the same format. Consequently, all
supported SVG geometry is reduced to a single primitive type.

An additional normalization step is applied when the original SVG uses the
`evenodd` fill rule. The internal representation and SVG export assume the
non-zero winding rule, so contour orientations must be adjusted to preserve the
same filled region. The implemented procedure first splits the curve sequence
into contiguous subpaths, then estimates the nesting depth of each subpath by
testing whether a representative point lies inside other subpaths. Subpaths at
even depth are treated as outer boundaries and subpaths at odd depth as holes.
Their orientation is then reversed when necessary so that outer contours are
clockwise and holes are counter-clockwise in SVG image coordinates. This makes
the geometry compatible with the non-zero rule without changing the visual
appearance of the shape.

Once all segments have been converted to cubic Bezier curves and, if needed,
their winding order has been normalized, the curve list is partitioned into
Bezier paths. A new path is started whenever the start point of the current
curve does not coincide with the endpoint of the previous one. Each resulting
subpath is stored as one `BezierPath`, and the collection of all subpaths with
their common color and opacity forms one `BezierShape`. The final output of the
parser is therefore a list of shapes in the same hierarchical form that is
subsequently transformed into the fixed-length tensor representation used for
training.

= Synthetic data generator

In addition to SVG data collected from external sources, this work uses a
synthetic data generator implemented in `synthetic.py`. Its purpose is to
produce a large number of geometrically valid training examples directly in the
target Bezier representation. This provides precise control over scene
complexity, guarantees compatibility with the representation used by the model,
and makes it possible to generate effectively unlimited training data without
additional annotation or SVG cleaning.

The generator produces scenes composed of multiple filled shapes on a square
canvas. Each scene contains a random number of objects sampled from a prescribed
interval. Every object is represented as one `BezierShape` with one or more
closed `BezierPath` contours, a solid RGB color, and an opacity value. Shape
opacity is set to one in most cases, while a smaller subset of shapes is drawn
with reduced opacity in order to expose the model to moderate transparency
variation.

The available shapes are divided into three categories:

- Primitive shapes: circles, ellipses, squares, rectangles, triangles,
  pentagons, hexagons, and stars. These objects provide simple closed contours
  with analytically controlled geometry.
- Organic shapes: smooth, rough, and spiky blobs generated from perturbed
  radial samples. These objects introduce irregular boundaries and more varied
  local curvature.
- Compound shapes: L-shapes, crosses, arrows, crescents, ring sectors, rounded
  rectangles, trapezoids, and parallelograms. These objects increase geometric
  diversity by introducing concavity, varying thickness, and mixed straight and
  curved boundaries.

This design was chosen to cover both analytically simple contours and shapes
with more varied topology and curvature.

All generated geometry is expressed as cubic Bezier curves. Straight polygonal
edges are represented by degenerate cubic segments whose control points lie on
the corresponding line segment. Circles and ellipses are approximated by four
cubic Bezier segments using the standard constant $kappa = 0.5522847498$.
Rounded rectangles and related figures combine linear segments with cubic arc
approximations. Organic blobs are generated differently: first, a set of angles
is distributed around a circle with random angular perturbation; second, a
radius is sampled independently for each angle; third, the resulting contour
points are connected by a closed chain of cubic Bezier segments obtained from a
Catmull-Rom-style tangent construction. The handle length is scaled by a
smoothness parameter, which allows the generator to control whether the blob is
smooth, rough, or spiky.

For each sampled shape, geometric parameters such as size, aspect ratio,
rotation, and contour detail are drawn from random intervals that depend on the
shape category. The object center is then sampled under a margin constraint so
that the entire shape remains inside the canvas with high probability. An
approximate extent function is used for this purpose. After construction, the
outer contour is explicitly oriented counter-clockwise. This establishes a
consistent winding convention for all synthesized objects.

Some shapes may additionally contain holes. Hole generation is only attempted
for shape types for which an internal contour can be inserted reliably.
Candidate hole contours are sampled inside the bounding box of the outer shape,
scaled to a random fraction of its size, and randomly offset away from the
exact center. The hole itself may again be circular, polygonal, blob-like, or
rounded-rectangular. In contrast to the outer boundary, the hole contour is
forced to clockwise orientation. This opposite winding is necessary because the
exported SVGs use the non-zero fill rule, under which opposite contour
directions produce cut-out regions.

A complete synthetic scene is constructed by repeatedly sampling shapes until
either the requested number of shapes is reached or the global segment budget is
exhausted. The segment budget is important because the downstream model expects
a fixed maximum number of Bezier segments per sample. Instead of generating a
scene first and truncating it afterwards, the generator stops adding shapes as
soon as the next object would exceed the allowed number of segments. This
ensures that every produced scene is valid without altering already generated
geometry.

The dataset interface is implemented by the `SyntheticBezierDataset` class,
which generates samples on the fly. For index $i$ in epoch $e$, the random seed
is chosen deterministically as
$ s_(i,e) = s_0 + i + e N $,
where $s_0$ is a base seed and $N$ is the virtual dataset length. Consequently,
the same epoch is reproducible, while different epochs expose the model to new
synthetic scenes. Each generated scene is converted to the tensor
representation using `shapes_to_tensor`, serialized back to SVG, rasterized to
an RGB image, and finally processed by the DINOv3 image processor. The dataset
therefore returns the same pair as the real dataset, namely a tensor of Bezier
segments and a corresponding conditioning raster image. This makes the
synthetic generator a drop-in replacement for supervised training and
qualitative sampling.

= Model architecture

The predictive model is implemented in `model.py` as a conditional flow-matching
transformer. Its input consists of two parts: a sequence of noisy Bezier-segment
descriptors and a raster conditioning image. The output is a sequence of the
same length and dimensionality as the Bezier input, interpreted as a velocity
field in representation space. The architecture therefore operates directly on
the continuous tensor representation introduced above and predicts how a noisy
sample should move toward a valid vector graphic conditioned on the raster
image.

The conditioning branch is based on a pretrained DINOv3 visual encoder,
specifically `facebook/dinov3-vits16-pretrain-lvd1689m`. The encoder is kept
frozen throughout training and is used only to extract a sequence of visual
features from the conditioning raster image. Concretely, the model takes the
last hidden state of DINOv3 and linearly projects it to the internal hidden
dimension of the transformer. This yields a sequence of conditioning tokens that
serve as keys and values in cross-attention. Freezing the image encoder reduces
the number of trainable parameters and stabilizes optimization, while still
providing semantically rich image descriptors.

The Bezier branch processes a tensor of segment descriptors of shape
$(B, N, D)$, where $B$ is batch size, $N$ is the maximum number of segments,
and $D = 13$ is the segment dimensionality. Each segment vector is projected by
a learned linear layer into a hidden space of dimension $H$. The scalar flow
time $t in [0, 1]$ is embedded separately using sinusoidal features followed by
a multilayer perceptron. The resulting time embedding is then used to modulate
all transformer blocks through adaptive layer normalization.

The backbone itself is a stack of transformer blocks of DiT type. Each block
contains three sublayers:

- RoPE self-attention over the Bezier token sequence.
- Cross-attention from Bezier tokens to image-conditioning tokens.
- A position-wise feed-forward network.

Self-attention uses rotary positional embeddings applied to the query and key
vectors. This gives the model information about the order of segments within
the sequence while preserving the attention-based formulation. Cross-attention
does not use rotary embeddings; instead, it lets each Bezier token attend to
the visual features extracted from the raster image. In this way, the model can
combine geometric context from the partially denoised vector sequence with
semantic and structural cues present in the conditioning image.

Each transformer block is modulated by the time embedding using adaptive layer
normalization with gating. More precisely, the time embedding is passed through
a small modulation network that predicts, for each of the three sublayers, a
shift vector, a scale vector, and a residual gate. If $x$ denotes a normalized
token representation, the modulation takes the form
$ mod(x) = x dot (1 + gamma) + beta $,
where $beta$ and $gamma$ are functions of the time embedding. The gated residual
connection then controls how strongly the output of the corresponding sublayer
is injected back into the main stream. This design allows the network to adapt
its computation continuously as a function of flow time, which is essential for
learning a time-dependent vector field.

After the stacked transformer blocks, the model applies one final adaptive
normalization step conditioned on time and then projects the hidden
representation back to the original Bezier-segment dimension. The final linear
projection is initialized with zeros, so the network initially predicts a near
zero velocity field. This is a common stabilization strategy in diffusion-like
and flow-based transformer models, because it avoids large uncontrolled updates
at the beginning of training.

Training follows the rectified-flow formulation. Let $x_1$ denote a ground
truth Bezier tensor sampled from the dataset and let $x_0$ be Gaussian noise of
the same shape. A scalar time $t$ is sampled for each training example from a
logit-normal distribution obtained by applying the sigmoid function to a
standard normal sample. The noisy intermediate point is then constructed by
linear interpolation
$ x_t = t x_1 + (1 - t) x_0 $.
The target velocity is defined as
$ v^ast = x_1 - x_0 $.
Given $x_t$, $t$, and the image-conditioning tokens, the network predicts a
velocity field $v_theta(x_t, t, c)$ and is optimized using the mean squared
error objective
$ L = ||v_theta(x_t, t, c) - v^ast||_2^2 $.
In the current implementation, this loss is evaluated over the full sequence,
including padded positions.

To support classifier-free guidance, the model uses conditioning dropout during
training. With a fixed probability, the image-conditioning sequence is replaced
by a learned null token broadcast across the conditioning length. This teaches
the network both conditional and unconditional velocity fields within a single
set of parameters. During inference, the two predictions can be combined as
$ v = v_u + w (v_c - v_u) $,
where $w$ is the guidance scale. When $w = 1$, standard conditional sampling is
recovered.

Sampling is performed by solving the learned ordinary differential equation from
noise toward data. The process starts from an initial sample
$ x(0) ~ N(0, I) $.
The model then integrates the velocity field from $t = 0$ to $t = 1$ using the
classical fourth-order Runge-Kutta method with a fixed number of time steps. In
each integration step, the transformer is evaluated one or more times to obtain
the required intermediate velocities. The final state is interpreted as a
predicted Bezier tensor, which is subsequently converted back to vector shapes
and rendered as SVG. This sampling procedure is deterministic for fixed initial
noise, fixed conditioning, and fixed integration parameters.
