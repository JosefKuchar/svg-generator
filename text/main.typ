#import "@git/fi-muni-thesis:0.1.0": fithesis

#show: fithesis.with(
  title: [Generative Neural Models for Scalable Vector Graphics],
  author: [Josef Kuchař],
  advisor: [Mgr. Michal Štefánik, Ph.D.],
  department: [Department of Machine Learning and Data Processing],
  faculty_name: [Faculty of Informatics],
  thesis_type: [Master's Thesis],
  place: "Brno",
  semester: "Spring 2026",
  declaration_body: [
    Hereby I declare that this paper is my original authorial work, which I have
    worked out on my own. All sources, references, and literature used or
    excerpted during elaboration of this work are properly cited and listed in
    complete reference to the due source.
  ],
  thanks_body: [
    These are the acknowledgements for my thesis, which can

    span multiple paragraphs.
  ],
  abstract_body: [
    This is the abstract of my thesis, which can

    span multiple paragraphs.
  ],
  keywords: (
    "scalable vector graphics",
    "raster-to-vector conversion",
    "Bezier curves",
    "flow matching",
    "text-to-image generation",
    "LoRA fine-tuning",
  ),
)

= Thesis scaffold

This thesis studies a two-stage pipeline for text-driven vector graphics
generation. In the first stage, a pretrained text-to-image model is adapted to
generate raster images in a visual domain suitable for subsequent
vectorization. In the second stage, a custom model is trained from scratch to
convert the generated raster image into a structured Bezier-based vector
representation. The main motivation for this decomposition is that high-quality
text-conditioned image synthesis and topology-aware vector generation pose
different modeling challenges. Instead of solving both problems inside a single
model, the proposed approach separates them into two tractable components.

= Introduction

The goal of the work is to generate vector graphics from textual input. Direct
text-to-vector generation is difficult because the model must simultaneously
learn semantic grounding, visual composition, geometric structure, and the
syntactic constraints of vector graphics. A two-stage pipeline offers a more
modular alternative. First, a text prompt is converted into a raster image by a
large pretrained generative model. Second, the raster image is translated into
a vector representation composed of Bezier curves. This separation makes it
possible to exploit the strengths of modern text-to-image models while
developing a specialized vectorizer that operates in a well-defined geometric
output space.

The work also considers several alternative formulations of the problem. In
particular, a direct adaptation of a text-to-raster model into a
text-to-Bezier model would be an elegant solution, because it would remove the
explicit vectorization stage. Preliminary experiments, however, indicate that
this route is not data-efficient in the present setting. The model converged
faster from random initialization than from pretrained image-generation
weights, suggesting that the learned raster-generation representation does not
transfer straightforwardly to the Bezier output space. This direction may still
be feasible at a larger scale, but it likely requires substantially more
paired text-vector data than is available for this thesis.

= Prior work

This thesis is related to several research directions at the intersection of
generative modeling, vector graphics, and multimodal learning. The most
relevant prior work can be grouped into the following categories.

== Text-to-SVG generation

The closest line of work aims to generate SVG content directly from textual
descriptions. These approaches typically formulate the problem either as code
generation, where the model predicts SVG tokens or commands autoregressively,
or as structured graphics generation, where the model predicts vector objects
and their attributes in a more constrained representation. The main advantage
of direct text-to-SVG generation is that it avoids an intermediate raster
representation and can therefore produce editable vector output in a single
stage. However, this formulation is challenging because the model must jointly
learn semantic alignment with text and the geometric and syntactic regularities
of valid SVG documents.

From the perspective of this thesis, direct text-to-SVG methods are important
as a conceptual baseline. They address the same end goal as the proposed
system, but differ in where the complexity is handled. In the direct setting,
semantic generation and vector-structure generation are solved simultaneously.
In the present work, these two difficulties are separated into a raster
generation stage and a dedicated vectorization stage.

Representative recent examples include StarVector @rodriguez2024starvector and
OmniSVG @yang2025omnisvg. StarVector studies SVG generation from both images
and text and frames the problem as generating executable SVG code, which makes
it a useful reference point for code-oriented approaches to vector graphics
synthesis. OmniSVG pursues a unified and scalable formulation of SVG
generation, further highlighting the growing interest in treating SVG creation
as a first-class generative modeling problem rather than only as a downstream
vectorization step. These works are closely aligned with the overall objective
of this thesis, but they differ from the proposed approach in that they seek to
produce vector graphics directly within a single model. By contrast, the method
developed here deliberately decomposes the task into raster generation and
subsequent vectorization, which reduces the burden placed on any one model and
allows the vectorizer to focus specifically on geometric reconstruction from
image evidence.

== Image-to-SVG and vectorization methods

A second related area consists of methods that convert raster images into
vector graphics. Classical vectorization systems rely on contour extraction,
polygon fitting, spline fitting, or optimization-based refinement. More recent
learning-based approaches predict vector primitives directly from raster input,
often using transformers, diffusion models, or autoregressive decoders. These
methods are highly relevant to the second stage of the proposed pipeline,
because they focus on the structured reconstruction problem independently of
text conditioning.

The model proposed in this thesis belongs primarily to this category, but it
differs in the use of flow matching @lipman2023flow and in the specific
Bezier-segment representation employed for training and decoding.

// TODO: Add representative classical and neural vectorization methods.

Existing vectorizers also serve as empirical baselines. Classical systems are
strong engineering tools, but they typically optimize local image fidelity and
often produce dense, fragmented paths when the input contains noise,
compression artifacts, blur, or soft color transitions. Recent neural
text-to-SVG systems, by contrast, often rely on large vision-language models
fine-tuned on SVG data. A planned part of the evaluation is therefore to
measure whether such methods generalize beyond their training distribution,
including on synthetic images whose ground-truth Bezier structure is known.
This experiment has not yet been completed and is kept as an explicit
evaluation item.

== Text-to-image models adapted for vector graphics

Another important line of prior work concerns large text-to-image models that
are adapted to generate images in a style suitable for graphic design,
illustration, icons, or symbol-like imagery. Even when such models do not
produce vector output directly, they can provide strong semantic grounding and
composition capabilities. This idea motivates the first stage of the proposed
pipeline, where a pretrained text-to-image model is adapted through LoRA and
used as a controllable raster generator.

This category is important because it justifies the decomposition adopted in
this thesis. If a text-to-image model can be specialized to generate
vectorization-friendly raster images, then the semantic burden of text
understanding can be largely delegated to that model, while the second stage
can focus on geometric reconstruction.

// TODO: Add references on LoRA adaptation and text-to-image models used for
// stylized or domain-specific generation.

== Position of this work

The proposed method combines ideas from the above areas but occupies a distinct
position. It is not a direct text-to-SVG generator, because it introduces an
intermediate raster representation. It is also not a generic text-to-image
system, because its raster output is explicitly optimized for subsequent
vectorization. Finally, although the second stage is a vectorization model, it
is designed as part of a larger multimodal pipeline rather than as an isolated
image-processing tool. This combination defines the main contribution of the
thesis: a modular text-to-vector pipeline in which semantic generation and
structured geometric generation are addressed by separate but compatible models.

= Proposed pipeline

The proposed system consists of the following two stages:

- Stage 1: text-to-raster generation. A pretrained `z-image` model
  @imageteam2025zimage is adapted with a LoRA module so that it produces
  images with characteristics suitable for vector graphics generation. The
  adapted weights are then applied in the accelerated `Z-Image-Turbo` pipeline
  for efficient inference.
- Stage 2: raster-to-vector generation. A custom conditional flow-matching
  model based on flow matching @lipman2023flow is trained from scratch to
  convert the raster image into a sequence of Bezier-segment descriptors,
  which can then be decoded into SVG paths.

From a methodological perspective, the first stage addresses semantic image
synthesis from text, while the second stage addresses structured geometric
reconstruction. The interface between the two stages is the raster image
itself, which allows the vectorization model to be trained independently of the
text-to-image model once a suitable image distribution has been established.

Training a direct text-to-Bezier model would require large quantities of
paired text descriptions and vector annotations. Such supervision is scarce in
the available data. The vectorizer-based decomposition avoids this bottleneck:
the raster-to-vector model can be pretrained on procedurally generated
synthetic data, for which vector labels are available by construction, and then
adapted to real SVG data. The central experimental question is whether
sufficiently varied synthetic pretraining can transfer to real vector graphics
after fine-tuning.

This is a practical advantage of formulating the second stage as
raster-to-vector generation rather than direct text-to-vector generation. For
the vectorizer, every procedurally generated vector scene can be rendered to a
raster image and used immediately as a paired training example. This makes it
possible to create an effectively unlimited amount of supervised data at low
cost. In contrast, direct text-to-SVG training would require SVGs paired with
high-quality textual descriptions, which are much harder to collect at scale
and are not produced automatically by the vector representation itself.

= Stage 1: Text-to-raster generation

The first stage is based on the pretrained `z-image` family of image-generation
models @imageteam2025zimage. In this work, the goal is not to train such a
model from scratch, but to adapt it to the target visual domain through
low-rank adaptation @hu2022lowrank. A LoRA
module is trained on a dataset of image-text pairs so that the model learns to
produce raster outputs that better match the desired properties of vector-like
illustrations. These properties may include simplified composition, cleaner
silhouettes, reduced texture complexity, and visual styles that are easier to
approximate by Bezier curves.

The LoRA adaptation was trained using the AI-Toolkit framework with the AdamW
optimizer @loshchilov2018decoupled and a learning rate of
$1 times 10^(-4)$. This configuration was used as the default starting point
for the Stage 1 adaptation experiments.

For inference, the base `z-image` model and the accelerated `Z-Image-Turbo`
model were evaluated with different sampling settings. The base model was
sampled with 50 denoising steps and classifier-free guidance
@ho2021classifierfree scale 4. By
contrast, `Z-Image-Turbo` was sampled with 8 denoising steps and without
classifier-free guidance, because the turbo model is guidance-distilled and is
intended to operate without an explicit CFG term at inference time.

After training, the learned LoRA weights are loaded into the `Z-Image-Turbo`
pipeline for fast sampling. This design preserves the knowledge of the original
pretrained model while making inference substantially more efficient than full
base-model fine-tuning. The use of the same SVG-style LoRA on the distilled
turbo variant is motivated by the observation that distilled diffusion models
can preserve the controllability of their teacher models, allowing controls
learned for the base model to remain useful after distillation
@gandikota2025distilling. The first stage of the thesis should therefore
explain the following aspects:

- the choice of the base `z-image` model,
- the motivation for LoRA-based adaptation,
- the training data used for adaptation,
- the prompt design and inference configuration, including the prompt prefix
  `SVG illustration with white background. `,
- the transfer of the learned LoRA weights to `Z-Image-Turbo`.

At this point, this section serves primarily as structural scaffolding. The
detailed experimental and implementation description of the LoRA training
procedure can be filled in later.

A preliminary comparison of several Stage 1 variants is shown qualitatively in
@tab:stage1-raster-examples and quantitatively in @tab:stage1-benchmark. The
compared variants include the base `z-image` model, prompt-prefixing strategies,
the accelerated `Z-Image-Turbo` model, and a provisional LoRA adaptation applied
to the turbo pipeline. Higher CLIP and DINO similarity indicate better alignment
with the reference images, whereas lower vectorization MSE indicates that the
generated raster outputs are easier to convert in the second stage. CLIP-based
and DINO-based similarity measures are also relevant because they have been
reported to correlate well with human preference in vector-graphics evaluation
@rodriguez2024starvector.

The vectorization MSE is a round-trip traceability metric. For each generated
raster image, the benchmark first converts the image to SVG using `vtracer` with
its default command-line settings. The resulting SVG is then rasterized back to
the original image resolution on a white background. The score is the mean
squared error between the original generated RGB image and this rerendered
image,
$ 1 / (3 H W) sum_(c, y, x) (I_(c,y,x) - hat(I)_(c,y,x))^2 $,
where pixel values are measured in the usual 0--255 RGB range. This metric does
not compare the generated image to the reference image directly. Instead, it
measures how much visual information is lost when the image is approximated by
a standard vectorization tool, so lower values indicate images whose shapes and
colors are easier to represent as clean vector graphics.

#figure(
  table(
    columns: (1.6fr, 1fr, 1fr, 1fr, 1fr),
    align: (left, center, center, center, center),
    inset: 4pt,
    stroke: (x, y) => (
      left: if x == 0 { none } else { 0.4pt },
      top: if y == 0 { none } else { 0.4pt },
    ),
    table.header([Variant], [Example 1], [Example 2], [Example 3], [Example 4]),
    [Reference],
    image("assets/raster/reference/0001.png", width: 100%),
    image("assets/raster/reference/0002.png", width: 100%),
    image("assets/raster/reference/0003.png", width: 100%),
    image("assets/raster/reference/0004.png", width: 100%),

    [Base],
    image("assets/raster/base/0001.png", width: 100%),
    image("assets/raster/base/0002.png", width: 100%),
    image("assets/raster/base/0003.png", width: 100%),
    image("assets/raster/base/0004.png", width: 100%),

    [Base prefixed],
    image("assets/raster/base_prefixed/0001.png", width: 100%),
    image("assets/raster/base_prefixed/0002.png", width: 100%),
    image("assets/raster/base_prefixed/0003.png", width: 100%),
    image("assets/raster/base_prefixed/0004.png", width: 100%),

    [OmniSVG 8B],
    image("assets/raster/omnisvg_8b/0001.png", width: 100%),
    image("assets/raster/omnisvg_8b/0002.png", width: 100%),
    image("assets/raster/omnisvg_8b/0003.png", width: 100%),
    image("assets/raster/omnisvg_8b/0004.png", width: 100%),

    [Turbo],
    image("assets/raster/turbo/0001.png", width: 100%),
    image("assets/raster/turbo/0002.png", width: 100%),
    image("assets/raster/turbo/0003.png", width: 100%),
    image("assets/raster/turbo/0004.png", width: 100%),

    [Turbo prefixed],
    image("assets/raster/turbo_prefixed/0001.png", width: 100%),
    image("assets/raster/turbo_prefixed/0002.png", width: 100%),
    image("assets/raster/turbo_prefixed/0003.png", width: 100%),
    image("assets/raster/turbo_prefixed/0004.png", width: 100%),
  ),
  caption: [Qualitative Stage 1 comparison of text-to-raster model variants.],
) <tab:stage1-raster-examples>

#figure(
  table(
    columns: (2.5fr, 1fr, 1fr, 1fr),
    align: (left, center, center, center),
    inset: 6pt,
    stroke: (x, y) => (
      left: if x == 0 { none } else { 0.4pt },
      top: if y == 0 { none } else { 0.4pt },
    ),
    table.header([Variant], [CLIP similarity ↑], [DINO similarity ↑], [Vectorization MSE ↓]),
    [Base], [0.818210], [0.509159], [266.565137],
    [Base prefixed], [0.819865], [0.545802], [230.160058],
    [OmniSVG 8B], [0.833605], [0.425360], [51.145968],
    [Turbo], [0.826786], [0.509892], [227.691742],
    [Turbo prefixed], [0.871237], [0.583856], [142.711678],
    [Turbo prefixed + LoRA (provisional)], [0.879104], [0.600208], [143.174617],
  ),
  caption: [Preliminary Stage 1 benchmark of text-to-raster model variants.],
) <tab:stage1-benchmark>

An additional ablation study was performed for the Stage 1 LoRA adaptation in
order to evaluate the effect of training duration and LoRA rank. Three LoRA
ranks, namely 4, 16, and 64, were evaluated at checkpoints saved every 500
steps. For the files without an explicit checkpoint suffix, the final model is
interpreted as the 5000-step checkpoint. The resulting CLIP similarity, DINO
similarity, and vectorization MSE values are summarized in
@tab:lora-ablation-clip, @tab:lora-ablation-dino, and @tab:lora-ablation-mse.

#figure(
  table(
    columns: (1.2fr, 1fr, 1fr, 1fr),
    align: (left, center, center, center),
    inset: 6pt,
    stroke: (x, y) => (
      left: if x == 0 { none } else { 0.4pt },
      top: if y == 0 { none } else { 0.4pt },
    ),
    table.header([Time steps], [LoRA 4], [LoRA 16], [LoRA 64]),
    [500], [0.880089], [0.871655], [0.884773],
    [1000], [0.886276], [0.887416], [0.884985],
    [1500], [0.885647], [0.886953], [0.884481],
    [2000], [0.885373], [0.885145], [0.886703],
    [2500], [0.887538], [0.888426], [0.881834],
    [3000], [0.886957], [0.886502], [0.888467],
    [3500], [0.885568], [0.887313], [0.887707],
    [5000], [0.886028], [0.885978], [0.882711],
  ),
  caption: [Stage 1 LoRA ablation measured by CLIP similarity (higher is better).],
) <tab:lora-ablation-clip>

#figure(
  table(
    columns: (1.2fr, 1fr, 1fr, 1fr),
    align: (left, center, center, center),
    inset: 6pt,
    stroke: (x, y) => (
      left: if x == 0 { none } else { 0.4pt },
      top: if y == 0 { none } else { 0.4pt },
    ),
    table.header([Time steps], [LoRA 4], [LoRA 16], [LoRA 64]),
    [500], [0.606738], [0.598794], [0.599245],
    [1000], [0.615834], [0.621120], [0.619718],
    [1500], [0.622208], [0.618302], [0.615209],
    [2000], [0.627051], [0.614469], [0.625228],
    [2500], [0.625920], [0.626535], [0.619530],
    [3000], [0.625713], [0.635021], [0.632441],
    [3500], [0.625818], [0.629640], [0.633112],
    [5000], [0.620072], [0.631474], [0.617148],
  ),
  caption: [Stage 1 LoRA ablation measured by DINO similarity (higher is better).],
) <tab:lora-ablation-dino>

#figure(
  table(
    columns: (1.2fr, 1fr, 1fr, 1fr),
    align: (left, center, center, center),
    inset: 6pt,
    stroke: (x, y) => (
      left: if x == 0 { none } else { 0.4pt },
      top: if y == 0 { none } else { 0.4pt },
    ),
    table.header([Time steps], [LoRA 4], [LoRA 16], [LoRA 64]),
    [500], [299.456365], [187.121463], [205.943477],
    [1000], [328.455177], [128.576292], [199.656727],
    [1500], [206.473148], [143.796380], [346.287457],
    [2000], [142.765006], [97.675489], [168.835552],
    [2500], [174.613101], [92.145374], [273.638178],
    [3000], [173.292545], [245.512123], [132.058421],
    [3500], [166.196853], [93.775058], [157.162289],
    [5000], [178.813825], [166.223330], [172.532026],
  ),
  caption: [Stage 1 LoRA ablation measured by vectorization MSE (lower is better).],
) <tab:lora-ablation-mse>

The results suggest that prompt prefixing has a substantial effect, especially
for the turbo model. The best overall semantic similarity is obtained by the
provisional `Turbo prefixed + LoRA` configuration, while the lowest
vectorization error is achieved by `Turbo prefixed`. This indicates that the
adapted LoRA model improves perceptual alignment with the references, but its
advantage with respect to downstream vectorization should be verified on a
larger evaluation.

= Stage 2: Raster-to-vector generation

The second stage is the main methodological contribution of this work. It takes
as input a raster image, either drawn from the real dataset or generated by the
first stage, and predicts a structured vector representation based on Bezier
curves. Unlike the first stage, this model is developed and trained from
scratch specifically for the vectorization task. The following sections
describe the representation, data preparation, synthetic data generation, and
the architecture of the proposed raster-to-vector model.

== Training procedure

The raster-to-vector model is trained in two consecutive phases. The first
phase consists of pretraining on synthetic data generated procedurally in the
Bezier representation. The second phase consists of fine-tuning on the SVG
dataset derived from real vector graphics. This training schedule is motivated
by the observation that synthetic data and real SVG data provide complementary
advantages. Synthetic data offer unlimited quantity and precise control over
geometric variation, while real SVG data provide more realistic structure,
stylistic diversity, and distributional properties closer to the target use
case.

=== Pretraining on synthetic data

In the first phase, the model is exposed to procedurally generated scenes
containing simple primitives, compound shapes, blobs, and shapes with holes.
Because these data are generated directly in the target Bezier representation,
they are guaranteed to be geometrically valid and structurally consistent. This
stage is intended to teach the model the basic grammar of vector graphics:
curve continuity, path organization, contour winding, color consistency within
shapes, and the general relationship between raster appearance and vector
structure.

Synthetic pretraining is expected to be particularly useful in the early stages
of optimization, when the model must first learn how valid Bezier-based shapes
behave before it can model the greater complexity of real-world SVG content.
The effectively unlimited size of the synthetic dataset also reduces the risk
of overfitting and allows controlled experiments with scene complexity, segment
count, and object diversity.

In the current experimental setup, synthetic pretraining was performed on a
single NVIDIA H200 GPU with batch size 256 for approximately 10 days.

// TODO: Add exact pretraining configuration, including number of epochs,
// synthetic scene parameters, optimizer settings, and checkpoint selection.

=== Fine-tuning on the SVG dataset

After pretraining, the model is fine-tuned on a dataset obtained from real SVG
files converted into the internal Bezier representation. This stage adapts the
model from the simplified synthetic distribution to the more heterogeneous and
stylized distribution of real vector graphics. Compared with the synthetic
generator, real SVG data contain richer compositions, more varied contour
structures, and a broader range of design conventions. Fine-tuning therefore
serves to align the model with the final task distribution.

Conceptually, the second phase can be viewed as domain adaptation. The model
enters this phase already equipped with a prior over valid vector geometry and
must then specialize that prior to the statistics of the target dataset. This
two-stage training procedure is expected to be more data-efficient and more
stable than training exclusively on the real SVG dataset from random
initialization.

// TODO: Add fine-tuning details, including dataset split, learning-rate
// schedule, stopping criterion, and comparison against training from scratch.

== Bezier representation

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

This homogeneous Bezier representation is motivated by the learning problem as
well as by the target output format. Flow matching operates naturally in a
continuous vector space, so a fixed-dimensional continuous descriptor for each
segment is more suitable than a heterogeneous command language containing
separate primitives for lines, arcs, rectangles, circles, and paths. Converting
all supported geometry to cubic Bezier segments therefore gives the model a
single native output type while still preserving the ability to reconstruct
standard SVG paths.

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

== SVG Conversion to Bezier Representation

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

== Synthetic data generator

In addition to SVG data collected from external sources, this work uses a
synthetic data generator implemented in `synthetic.py`. Its purpose is to
produce a large number of geometrically valid training examples directly in the
target Bezier representation. This provides precise control over scene
complexity, guarantees compatibility with the representation used by the model,
and makes it possible to generate effectively unlimited training data without
additional annotation or SVG cleaning. This property is central to the proposed
training strategy. Because the generated vector scene is known exactly, the
corresponding raster input can be obtained by rendering, yielding a supervised
raster-to-vector pair without any manual labeling. The synthetic generator
therefore addresses one of the main data bottlenecks in vector-graphics
generation: while captioned SVG datasets are limited and expensive to curate,
uncaptioned vector geometry can be synthesized and rasterized cheaply.

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
with more varied topology and curvature. Examples of generated images are shown
in @fig:synthetic-generator-examples.

#let synthetic-generator-image(path) = box(
  stroke: 0.75pt + gray,
  image(path, width: 100%),
)

#figure(
  grid(
    columns: (1fr, 1fr, 1fr, 1fr),
    gutter: 4pt,
    synthetic-generator-image("assets/syntetic_generator/synthetic_generator_01.png"),
    synthetic-generator-image("assets/syntetic_generator/synthetic_generator_02.png"),
    synthetic-generator-image("assets/syntetic_generator/synthetic_generator_03.png"),
    synthetic-generator-image("assets/syntetic_generator/synthetic_generator_04.png"),

    synthetic-generator-image("assets/syntetic_generator/synthetic_generator_05.png"),
    synthetic-generator-image("assets/syntetic_generator/synthetic_generator_06.png"),
    synthetic-generator-image("assets/syntetic_generator/synthetic_generator_07.png"),
    synthetic-generator-image("assets/syntetic_generator/synthetic_generator_08.png"),

    synthetic-generator-image("assets/syntetic_generator/synthetic_generator_09.png"),
    synthetic-generator-image("assets/syntetic_generator/synthetic_generator_10.png"),
    synthetic-generator-image("assets/syntetic_generator/synthetic_generator_11.png"),
    synthetic-generator-image("assets/syntetic_generator/synthetic_generator_12.png"),

    synthetic-generator-image("assets/syntetic_generator/synthetic_generator_13.png"),
    synthetic-generator-image("assets/syntetic_generator/synthetic_generator_14.png"),
    synthetic-generator-image("assets/syntetic_generator/synthetic_generator_15.png"),
    synthetic-generator-image("assets/syntetic_generator/synthetic_generator_16.png"),
  ),
  caption: [Examples of images produced by the synthetic data generator.],
) <fig:synthetic-generator-examples>

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

== Model architecture

The predictive model is implemented in `model.py` as a conditional flow-matching
transformer. Its input consists of two parts: a sequence of noisy Bezier-segment
descriptors and a raster conditioning image. The output is a sequence of the
same length and dimensionality as the Bezier input, interpreted as a velocity
field in representation space. The architecture therefore operates directly on
the continuous tensor representation introduced above and predicts how a noisy
sample should move toward a valid vector graphic conditioned on the raster
image.

The conditioning branch is based on a pretrained DINOv3 visual encoder
@simeoni2025dinov3,
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

The backbone itself is a stack of transformer blocks of DiT type
@peebles2022dit. Each block
contains three sublayers:

- RoPE self-attention over the Bezier token sequence.
- Cross-attention from Bezier tokens to image-conditioning tokens.
- A position-wise feed-forward network.

Self-attention uses rotary positional embeddings applied to the query and key
vectors @su2024roformer. This gives the model information about the order of segments within
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
training @ho2021classifierfree. With a fixed probability, the image-conditioning sequence is replaced
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

= Experiments

This chapter will evaluate the two stages of the proposed pipeline and the
design choices that connect them. The experiments are organized around three
questions: whether the text-to-raster model can be adapted to a
vectorization-friendly image domain, whether synthetic pretraining improves the
raster-to-vector model, and how the proposed vectorizer compares with existing
classical and neural vectorization systems.

== Alternatives to the proposed decomposition

The first alternative is to adapt a pretrained text-to-raster model directly
into a text-to-Bezier model. This would be conceptually attractive, because it
would collapse the whole pipeline into one model while preserving the semantic
knowledge of the pretrained generator. A preliminary experiment with this
approach was performed, but it did not produce a useful model. In particular,
training from random initialization converged faster than training from the
pretrained raster-generation weights. This result suggests that the
representation learned for image denoising does not directly provide a useful
initialization for predicting Bezier control points. The approach is therefore
not used as the main method, although it remains a possible large-scale
direction if substantially more paired text-vector data become available.

The second alternative is to rely on existing vectorizers. This thesis will
compare the proposed method against both classical raster-to-vector conversion
tools and recent neural systems such as OmniSVG @yang2025omnisvg and
StarVector @rodriguez2024starvector. The comparison should distinguish
in-distribution performance from out-of-distribution behavior. A planned
experiment is to evaluate methods on synthetic raster images generated from
known vector ground truth and then perturb the images with Gaussian noise, JPEG
compression artifacts, and Gaussian blur. This setup makes it possible to test
whether each method reconstructs the original vector structure or overfits to
the visible pixel artifacts.

== Stage 1 fine-tuning

The Stage 1 experiments evaluate fine-tuning of the text-to-raster model on
the SVG Repo dataset. The SVG files are rasterized and used as the visual
target distribution for LoRA adaptation. The goal is not merely to improve
generic image quality, but to make generated images more suitable for
downstream vectorization: flatter color regions, sharper silhouettes, fewer
unnecessary textures, and simpler topology.

The main comparison is between the base text-to-raster model and the
fine-tuned variant. Qualitative examples will be included in the final thesis
to show the visual change before and after fine-tuning. Quantitatively, the
evaluation will combine text-image alignment metrics with traceability metrics.
CLIP similarity measures whether the generated image remains aligned with the
input text. Reconstruction through a standard vectorization tool measures how
well the generated raster image survives a raster-to-vector-to-raster
round-trip, using MSE or SSIM between the original generated image and the
rerendered traced image. Additional useful indicators include the number of
paths or nodes produced by the tracer, PNG compressibility, color entropy or
unique color count, and a targeted FID computed against the rasterized SVG
training distribution. These metrics reflect the fact that a successful Stage
1 model should produce images that are both semantically meaningful and easy to
represent as clean vector graphics.

== Stage 2 vectorizer training

The Stage 2 experiments evaluate the conditional flow-matching vectorizer. The
main training experiment compares fine-tuning from the synthetic pretrained
checkpoint against training from scratch on the SVG Repo data. This comparison
tests the central hypothesis that synthetic Bezier data provide a useful
geometric prior even though they are simpler than real SVG graphics. The
pretraining run used a single NVIDIA H200 GPU for approximately 10 days with
batch size 256 and FlashAttention 2 enabled.

Two architecture ablations will be used to measure the effect of image
conditioning. The first compares the full model with a model trained without an
image encoder. This evaluates whether DINOv3 features provide useful
conditioning beyond what the sequence model can learn from the noisy Bezier
input alone. The second studies encoder scaling by comparing different image
encoder sizes or configurations while keeping the vectorizer backbone as
constant as possible. Together, these experiments should clarify how much of
the performance is due to the flow-matching transformer and how much comes
from the pretrained visual representation.

== Flow-matching inference ablation

Inference requires numerical integration of the learned velocity field. The
number of integration steps directly affects runtime and reconstruction
quality. The inference ablation will therefore evaluate several fixed step
counts and measure the resulting output quality, topology, and rendering error.
This experiment is important because an excessively small number of steps may
produce unstable or incomplete geometry, while too many steps increase runtime
without necessarily improving the final SVG.

== Pipeline evaluation

The final system combines the fine-tuned text-to-raster model with the
raster-to-vector model. Evaluation of the full pipeline should include both
end-to-end qualitative examples and quantitative comparisons with baselines.
The expected baselines include classical vectorization tools, recent neural
SVG-generation systems, and direct raster outputs from the Stage 1 model. The
comparison should emphasize not only pixel-level reconstruction, but also
properties important for editable vector graphics, such as path count, node
count, topological cleanliness, robustness to noisy inputs, and ease of manual
editing.

= Limitations

The current representation is intentionally restricted. All geometry is
expressed as cubic Bezier segments, so higher-level SVG primitives such as
circles, rectangles, and symbolic shape elements are not preserved as separate
objects. Although these primitives can be approximated accurately by Bezier
curves, the resulting SVG is less semantically editable than an SVG that keeps
the original primitive types. This limitation could be addressed by adding a
post-processing stage that converts reconstructed Bezier shapes back into
higher-level primitives where possible. For example, closed contours could be
tested for compatibility with circles, ellipses, rectangles, rounded
rectangles, polygons, or simple compound shapes, and then replaced by the
corresponding SVG elements when the approximation error is sufficiently small.
Such primitive recovery would not change the generative model itself, but it
would improve the semantic editability and compactness of the final SVG output.

The current data pipeline assumes solid fills with opacity and does not parse
gradients, masks, filters, or complex style rules. This excludes a significant
part of the SVG design space and limits the visual complexity of the generated
output. Gradients are a partial exception from a representational perspective:
although the current training data repeat one color over all segments of a
shape, the tensor representation already stores color per segment. This means
that smooth or piecewise-smooth color variation could in principle be
approximated by assigning different colors to neighboring segments. Such an
extension would require changes to data conversion, rendering, and evaluation,
but not necessarily a completely different geometric representation. In
addition, the fixed maximum number of segments imposes a hard capacity limit:
graphics requiring more segments must either be simplified or truncated.

Finally, the current pipeline assumes a white background in both the
text-to-raster stage and the vectorizer training setup. This simplifies
rasterization and evaluation, but it also constrains the kinds of graphics that
can be represented cleanly. A natural extension would be to modify the
text-to-raster stage so that it produces images with transparency instead of
images composited onto a white canvas, and to train the raster-to-vector model
on corresponding transparent raster inputs. Methods for transparent image
generation, such as latent-transparency diffusion @zhang2024latenttransparency,
suggest one possible way to adapt the first stage for this setting. The
synthetic generator could also provide transparent training examples directly,
because each generated vector scene can be rendered as RGBA rather than RGB.
Such a setup would better match the common use of SVG graphics as foreground
assets and would remove the need for the vectorizer to interpret the white
background as a special implicit class.

#bibliography("references.bib")
