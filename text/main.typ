#import "@git/fi-muni-thesis:1.0.0": appendix, fithesis, thesis_bibliography

#show: fithesis.with(
  title: [Generative Neural Models for Scalable Vector Graphics],
  author: [Josef Kuchař],
  advisor: [Mgr. Michal Štefánik, Ph.D.],
  department: [Department of Visual Computing],
  faculty_name: [Faculty of Informatics],
  thesis_type: [Master's Thesis],
  place: "Brno",
  semester: "Spring 2026",
  declaration_body: [
    Hereby I declare that this paper is my original authorial work, which I have
    worked out on my own. All sources, references, and literature used or
    excerpted during elaboration of this work are properly cited and listed in
    complete reference to the due source.

    During the preparation of this thesis, I used the following AI tools:
    - Grammarly for grammar checking,
    - OpenAI Codex for code writing and text writing,
    - Gemini Deep Research for finding relevant literature.

    I declare that I used these tools in accordance with the principles of
    academic integrity. I checked the content and take full responsibility for
    it.
  ],
  thanks_body: [
    I would like to thank my supervisor, Michal Štefánik, for his guidance and
    feedback throughout the work on this thesis. I am also grateful to my
    consultant, Marek Kadlčík, for helpful discussions and practical advice.
    Finally, I would like to thank my family for their support.
  ],
  abstract_body: [
    This thesis studies a two-stage approach to generating scalable vector
    graphics from text. The first stage adapts a pretrained text-to-image model
    to produce raster images that are suitable for vectorization. The second
    stage converts these raster images into a structured representation based
    on cubic Bezier curves using a conditional flow-matching model.

    The work focuses on separating semantic image generation from geometric
    reconstruction. This decomposition makes it possible to use large
    pretrained raster generators for prompt understanding while training the
    vectorizer on supervised raster-vector pairs, including procedurally
    generated synthetic data with known Bezier structure. The thesis describes
    the data conversion pipeline, the Bezier representation, the synthetic
    generator, the flow-matching architecture, and the evaluation methodology
    used to compare the proposed vectorizer with existing SVG-generation and
    raster-to-vector baselines.
  ],
  keywords: (
    "scalable vector graphics",
    "text-to-vector generation",
    "raster-to-vector conversion",
    "vector graphics generation",
    "editable image representation",
    "geometric reconstruction",
    "neural vectorization",
    "neural networks",
  ),
)

#heading(level: 1, numbering: none)[Introduction]

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

In this thesis, _vectorization_ refers to raster-to-vector conversion: the
problem of converting a pixel image into a visually similar vector graphic
described by geometric primitives. Raster images represent visual content as a
discrete grid of colored pixels, whereas vector graphics describe shapes by
parameters such as paths, curves, fills, and strokes. Rendering maps a vector
description to pixels; vectorization attempts the inverse direction by
recovering a compact and editable geometric description from raster evidence
@selinger2003potrace. This inverse problem is inherently ambiguous, because
many different sets of curves and shapes can render to nearly identical pixel
images. A useful vectorizer therefore must balance image fidelity with
structural simplicity, semantic editability, and validity of the resulting SVG
@dziuba2023imagevectorization.

The objective of the thesis is to design, implement, and evaluate a modular
text-to-vector pipeline that connects pretrained raster generation with a
specialized neural vectorizer. The work is organized around four research
questions:

- Can a pretrained text-to-image model be adapted to produce raster images that
  are easier to vectorize?
- Does procedurally generated Bezier data provide a useful pretraining signal
  for raster-to-vector generation?
- How does a conditional flow-matching vectorizer compare with autoregressive,
  classical tracing, and large neural SVG-generation baselines?
- What trade-offs arise between raster fidelity, SVG validity, compactness, and
  practical editability?

The main contributions are the two-stage pipeline design, the fixed-size
Bezier representation, the SVG normalization and conversion procedure, the
synthetic Bezier data generator, the conditional flow-matching vectorizer, and
the evaluation setup used to compare fidelity and structural complexity. The
thesis first reviews related work, then formulates the task and data pipeline,
describes the proposed method and training procedure, evaluates the individual
stages and the full system, and finally summarizes the achieved results and
limitations.

The work also considers several alternative formulations of the problem. In
particular, a direct adaptation of a text-to-raster model into a
text-to-Bezier model would be an elegant solution, because it would remove the
explicit vectorization stage. The experiments, however, indicate that
this route is not data-efficient in the present setting. The model converged
faster from random initialization than from pretrained image-generation
weights, suggesting that the learned raster-generation representation does not
transfer straightforwardly to the Bezier output space. This direction may still
be feasible at a larger scale, but it likely requires substantially more
paired text-vector data than is available for this thesis.

= Background and Related Work

This thesis is related to several research directions at the intersection of
generative modeling, vector graphics, and multimodal learning. The most
relevant prior work can be grouped into the following categories.

== Scalable vector graphics and Bezier curves

The Bezier curves used in this work are a standard way of representing curved
SVG paths. In SVG path data, a cubic Bezier segment is specified by an endpoint
and two control points relative to the current point; sequences of such
segments can describe smooth contours, while fills and strokes determine how
the paths are rendered @w3c2011svgpaths. This makes cubic Bezier curves a
natural low-level representation for learning, because they are expressive
enough to approximate many shapes while still being described by a small fixed
number of continuous parameters per segment.

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

SVG generation is difficult not only because the output is visual, but also
because the output is structured code. A model must produce syntactically valid
path data, choose an ordering of paths and shapes, handle attributes such as
fill color and opacity, and cope with the fact that multiple SVG programs can
render to very similar raster images. Recent SVG-generation systems therefore
either generate full SVG code with a code-oriented language model
@rodriguez2024starvector or simplify SVGs into a smaller command vocabulary
before training @yang2025omnisvg.

From the perspective of this thesis, direct text-to-SVG methods are important
as a conceptual baseline. They address the same end goal as the proposed
system, but differ in where the complexity is handled. In the direct setting,
semantic generation and vector-structure generation are solved simultaneously.
In the present work, these two difficulties are separated into a raster
generation stage and a dedicated vectorization stage.

Representative recent examples include StarVector @rodriguez2024starvector and
OmniSVG @yang2025omnisvg. Both systems treat SVG generation as a sequence
modeling problem, but they differ in how much of the SVG language they expose
to the model and in how they connect visual or textual conditioning to the
generated vector representation.

=== StarVector

StarVector formulates SVG synthesis as multimodal code generation. The model is
conditioned either on a raster image or on a text instruction and then predicts
the SVG document autoregressively as code. For image-to-SVG generation, the
raster input is encoded by a vision transformer, projected through an adapter
into the language-model embedding space, and prepended as visual tokens before
the SVG token sequence. For text-to-SVG generation, the conditioning signal is
provided by the language model's ordinary text tokenizer. In both cases, the
decoder is trained with a next-token objective over SVG code, so inference
amounts to sampling SVG markup until an end-of-SVG token is produced.

A central design choice in StarVector is to operate in the native SVG code
space. This allows the model to use higher-level primitives such as circles, ellipses, polygons, text,
and styling constructs when they are appropriate. The motivation is that a
semantically recognized circle should be emitted as a compact SVG primitive
rather than approximated by many small Bezier path segments. This distinguishes
StarVector from classical vectorizers, which often optimize pixel fidelity but
can produce long, fragmented paths with limited semantic editability.

The training data for StarVector are collected in SVG-Stack, a large dataset of
approximately two million SVG samples paired with raster renderings and
synthetic text descriptions. The dataset is intended to cover a broad range of
web SVG syntax and primitives, which is important because SVG generation is not
only a geometric task but also a code-validity task. StarVector is evaluated
with SVG-Bench on image-to-SVG, text-to-SVG, and diagram-generation tasks. The
paper also argues that purely pixel-based metrics such as MSE can be misleading
for vector graphics, because they do not measure compactness, primitive choice,
or editability. This observation is relevant for this thesis as well: a
round-trip raster error is useful, but it should be interpreted together with
properties of the generated vector structure.

=== OmniSVG

OmniSVG also uses an autoregressive formulation, but it avoids generating raw
XML markup directly. Instead, the input SVGs are simplified into a sequence of
atomic drawing commands and attributes. The representation includes move,
line, cubic Bezier, elliptical arc, close-path, and fill commands, while
coordinates and command types are discretized into tokens. This tokenizer
places vector geometry into the same sequential modeling framework as text and
image tokens, but it removes much of the syntactic variability of full SVG XML.
The model is built on a pretrained vision-language model, Qwen2.5-VL, and uses
text and image inputs as prefix tokens before generating the SVG command
sequence with a next-token prediction objective.

The purpose of this parameterization is to separate the higher-level structure
of the drawing from low-level coordinate prediction. Raw SVG code contains many
equivalent ways to express the same image, for example through transforms,
groups, or different primitive forms. OmniSVG reduces this ambiguity by
normalizing SVGs with tools such as `picosvg` and representing them with a
limited set of atomic commands. At the same time, it remains more expressive
than icon-only systems because it keeps color fills and supports longer
sequences for complex illustrations.

OmniSVG is trained and evaluated with MMSVG-2M, a multimodal dataset containing
about two million SVG assets, including icons, illustrations, and more complex
character graphics. Its benchmark covers text-to-SVG, image-to-SVG, and
character-reference SVG generation. This makes OmniSVG a useful reference point
for scalable conditional SVG generation: it demonstrates that large
vision-language models can be adapted to produce detailed editable vector
outputs when a sufficiently standardized SVG tokenizer and large-scale data are
available.

Both StarVector and OmniSVG are closely aligned with the overall objective of
this thesis, but they solve the problem in a different place in the pipeline.
They aim to learn semantic generation and vector-structure generation jointly
inside a single autoregressive model. The method developed here deliberately
decomposes the task into raster generation followed by raster-to-vector
conversion. This decomposition gives up the possibility of producing semantic
SVG primitives directly from text, but it reduces the data requirement for the
second stage: the vectorizer can be trained on raster-vector pairs generated
from known geometry, without requiring natural-language captions for every
vector sample.

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

Flow matching is a generative modeling objective for learning a continuous
vector field that transports samples from a simple source distribution, such as
Gaussian noise, toward the data distribution. The original flow matching work
frames this as simulation-free training of continuous normalizing flows by
regressing vector fields along prescribed probability paths, with diffusion
paths included as a special case @lipman2023flow. In this thesis, the same idea
is applied in the continuous space of Bezier-segment tensors: the model learns
how a noisy vector representation should move toward a valid raster-conditioned
vector graphic.

Representative approaches include differentiable vector-graphics
rasterization for optimization and learning @li2020diffvg, layer-wise image
vectorization @ma2022live, and learned SVG representations such as DeepSVG
@carlier2020deepsvg.

Existing vectorizers also serve as empirical baselines. Classical systems such
as Potrace @selinger2003potrace and `vtracer` @visioncortexVtracer are strong
engineering tools, but they typically optimize local image fidelity and often
produce dense, fragmented paths when the input contains noise, compression
artifacts, blur, or soft color transitions. The experiments use `vtracer` as
the classical tracing baseline. Recent neural text-to-SVG systems, by contrast,
often rely on large vision-language models fine-tuned on SVG data. The
evaluation in this work therefore distinguishes between performance on the SVG
validation distribution and behavior on synthetic images whose ground-truth
Bezier structure is known. This makes it possible to compare ordinary
reconstruction fidelity with robustness to inputs that differ from the web-SVG
distribution used by large neural baselines.

== Text-to-image models adapted for vector graphics

Another important line of prior work concerns large text-to-image models that
are adapted to generate images in a style suitable for graphic design,
illustration, icons, or symbol-like imagery. Even when such models do not
produce vector output directly, they can provide strong semantic grounding and
composition capabilities. This idea motivates the first stage of the proposed
pipeline, where a pretrained text-to-image model is adapted through LoRA and
used as a controllable raster generator.

Modern text-to-image systems are commonly based on diffusion models. A
diffusion model learns to reverse a gradual noising process: during training,
clean images are corrupted with noise, and the network learns a denoising
transition that can be applied iteratively to synthesize new samples from
noise @ho2020denoising. Latent diffusion models make this process more
efficient by performing the denoising in a compressed latent image space rather
than directly in pixel space, which is one reason they became practical for
high-resolution conditional image synthesis @rombach2022highresolution.

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

= Problem Formulation and Data

This chapter defines the decomposition used by the thesis and describes the
shared data source used by both stages of the pipeline. It separates the
question of what is being solved from the implementation details of the neural
models described in the following chapters.

== Task decomposition


The proposed system consists of the following two stages:

- Stage 1: text-to-raster generation. A pretrained Z-Image model
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

== Source SVG dataset

Both stages use the `mikronai/svg-svgrepo` dataset distributed through Hugging
Face @mikronaiSvgSvgrepo. The dataset is derived from SVG Repo graphics
@svgRepo and is provided as a tabular Parquet dataset. At the time of use, the
default subset
contained approximately 216k examples, split into approximately 214k training
examples, 1010 validation examples, and 1010 test examples. Each row contains
the raw SVG markup in the `item_svg` field, collection and item identifiers,
license metadata, item tags, an item title, and four generated text captions
with associated generation metadata.

This structure makes the dataset useful for both parts of the proposed
pipeline. For Stage 1, the SVG files are rasterized and paired with textual
captions, yielding image-text examples for LoRA adaptation of the
text-to-raster model. For Stage 2, the same SVG files provide vector
supervision: each SVG is converted into the internal Bezier representation and
also rasterized to obtain the conditioning image. The dataset is therefore a
shared source of semantic supervision for raster generation and geometric
supervision for raster-to-vector learning.

The dataset is heterogeneous because it aggregates graphics from many original
collections and licenses. This diversity is useful for evaluating
generalization, but it also requires filtering and normalization before
training. In particular, SVGs containing unsupported constructs such as
gradients, masks, embedded style blocks, or geometry that exceeds the fixed
segment budget are excluded or simplified by the preprocessing pipeline
described below.

= Proposed Method

This chapter describes the proposed two-stage system after the task has been
formulated. The first stage produces raster images from text, while the second
stage predicts a structured Bezier representation from a raster image.

== Text-to-raster adaptation


The first stage is based on the pretrained Z-Image family of image-generation
models @imageteam2025zimage. In this work, the goal is not to train such a
model from scratch, but to adapt it to the target visual domain through
low-rank adaptation @hu2022lowrank. A LoRA
module is trained on a dataset of image-text pairs so that the model learns to
produce raster outputs that better match the desired properties of vector-like
illustrations. These properties may include simplified composition, cleaner
silhouettes, reduced texture complexity, and visual styles that are easier to
approximate by Bezier curves.

LoRA is a parameter-efficient fine-tuning method. Instead of updating all
weights of a large pretrained model, it freezes the base weights and learns
small trainable low-rank matrices whose product approximates the desired weight
update @hu2022lowrank. This is useful in the first stage because the model
should retain the broad semantic and compositional knowledge of the pretrained
text-to-image system while adapting only a comparatively small number of
parameters to the SVG-like raster domain.

For inference, the base Z-Image model and the accelerated Z-Image Turbo
model were evaluated with different sampling settings. The base model was
sampled with 50 denoising steps and classifier-free guidance
@ho2021classifierfree scale 4. By
contrast, Z-Image Turbo was sampled with 8 denoising steps and without
classifier-free guidance, because the turbo model is guidance-distilled and is
intended to operate without an explicit CFG term at inference time.

Classifier-free guidance is a conditioning technique for diffusion models in
which the model is trained with both conditional and unconditional inputs. At
sampling time, the conditional prediction is strengthened by comparing it with
the unconditional prediction, and the guidance scale controls how strongly the
sample is pushed toward the prompt @ho2021classifierfree. This usually improves
prompt adherence but requires additional model evaluations unless the effect
has been distilled into a faster model.

After training, the learned LoRA weights are loaded into the Z-Image Turbo
pipeline for fast sampling. This design preserves the knowledge of the original
pretrained model while making inference substantially more efficient than full
base-model fine-tuning. The use of the same SVG-style LoRA on the distilled
turbo variant is motivated by the observation that distilled diffusion models
can preserve the controllability of their teacher models, allowing controls
learned for the base model to remain useful after distillation
@gandikota2025distilling. In the experiments reported below, prompts are
prefixed with "SVG illustration with white background. " to bias the generator
toward clean foreground graphics on a simple canvas. The resulting samples are
then assessed both as images and as inputs for downstream vectorization.


== Raster-to-vector generation


The second stage is the main methodological contribution of this work. It takes
as input a raster image, either drawn from the real dataset or generated by the
first stage, and predicts a structured vector representation based on Bezier
curves. Unlike the first stage, this model is developed and trained from
scratch specifically for the vectorization task. The following sections
describe the representation, data preparation, synthetic data generation, and
the architecture of the proposed raster-to-vector model.

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
standard SVG paths @w3c2011svgpaths.

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

This convention follows the sequential nature of SVG path data, where each
command starts from the current point left by the previous command and updates
that current point after execution @w3c2011svgpaths. Storing endpoints
implicitly removes one redundant coordinate pair per segment, while still
allowing the full path to be reconstructed when segment order and path
boundaries are known.

All quantities are normalized to the interval $[-1, 1]$. Let the original
raster image have width $W$ and height $H$. Coordinates are normalized with
respect to the image center
$ c_x = W / 2 quad c_y = H / 2 $
and the uniform scale factor
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

== SVG conversion to Bezier representation

The source dataset contains SVG files whose graphical content may be expressed
using a heterogeneous set of primitives, transformations, and grouping
constructs. Before these data can be used for training, each SVG must be
converted into a uniform representation compatible with the tensor encoding
described above. The conversion procedure therefore maps every supported
graphical element to a collection of filled cubic Bezier paths together with a
shared color and opacity.

The conversion begins with structural simplification. SVG files are first
processed externally in Inkscape, a vector graphics editor that supports
command-line batch processing through actions @inkscapeCommandLine. During this
stage, all objects are converted to paths and strokes are expanded into filled
outlines. This step removes many forms of SVG variability and ensures that the
subsequent parser operates on explicit geometric contours rather than on
higher-level drawing commands. Several classes of samples are excluded before
conversion, namely SVGs containing gradient definitions, masks, or embedded
style blocks. These constructs are not supported by the present representation,
which assumes a single solid fill color and a scalar opacity for each shape.

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

The need for this step comes from the SVG fill-rule definition. Under the
non-zero rule, whether a point is inside a shape depends on the signed winding
of contours around that point, while the even-odd rule depends on how many
times a ray from the point crosses the path @w3c2011svgpaths. The same visual
hole can therefore require different contour orientation conventions depending
on the chosen fill rule.

Once all segments have been converted to cubic Bezier curves and, if needed,
their winding order has been normalized, the curve list is partitioned into
Bezier paths. A new path is started whenever the start point of the current
curve does not coincide with the endpoint of the previous one. Each resulting
subpath is stored as one `BezierPath`, and the collection of all subpaths with
their common color and opacity forms one `BezierShape`. The final output of the
parser is therefore a list of shapes in the same hierarchical form that is
subsequently transformed into the fixed-length tensor representation used for
training.

== Model architecture

The predictive model is a conditional flow-matching transformer. Its input
consists of two parts: a sequence of noisy Bezier-segment descriptors and a
raster conditioning image. The output is a sequence of the same length and
dimensionality as the Bezier input, interpreted as a velocity field in
representation space. The architecture therefore operates directly on the
continuous tensor representation introduced above and predicts how a noisy
sample should move toward a valid vector graphic conditioned on the raster
image.

The conditioning branch is based on a pretrained DINOv3 visual encoder
@simeoni2025dinov3,
specifically `facebook/dinov3-vits16-pretrain-lvd1689m`. DINOv3 is a
self-supervised visual foundation model designed to produce transferable visual
features across a broad range of downstream tasks @simeoni2025dinov3. In this
work, the encoder is kept frozen throughout training and is used only to
extract a sequence of visual features from the conditioning raster image.
Concretely, the model takes the last hidden state of DINOv3 and linearly
projects it to the internal hidden dimension of the transformer. This yields a
sequence of conditioning tokens that serve as keys and values in
cross-attention. Freezing the image encoder reduces the number of trainable
parameters and stabilizes optimization, while still providing semantically rich
image descriptors.

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
$ x_t = t x_1 + (1 - t) x_0 $
The target velocity is defined as
$ v^ast = x_1 - x_0 $
Given $x_t$, $t$, and the image-conditioning tokens, the network predicts a
velocity field $v_theta(x_t, t, c)$ and is optimized using the mean squared
error objective
$ L = ||v_theta(x_t, t, c) - v^ast||_2^2 $
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

= Training Procedure

The training procedure follows the same decomposition as the method. The first
stage is adapted from a pretrained text-to-image model using LoRA. The second
stage is trained in two phases: synthetic Bezier pretraining followed by
fine-tuning on converted SVG data.

== Stage 1 LoRA adaptation

The LoRA adaptation was trained using the AI-Toolkit framework#footnote[
  AI-Toolkit project page: https://github.com/ostris/ai-toolkit.
] with the AdamW optimizer @loshchilov2018decoupled and a learning rate of
$1 times 10^(-4)$. This configuration was used as the default starting point
for the Stage 1 adaptation experiments. The rank and checkpoint selection are
evaluated in @sec:stage1-evaluation.

== Synthetic data generator

In addition to SVG data collected from external sources, this work uses a
synthetic data generator. Its purpose is to produce a large number of
geometrically valid training examples directly in the target Bezier
representation. This provides precise control over scene complexity,
guarantees compatibility with the representation used by the model, and makes
it possible to generate effectively unlimited training data without additional
annotation or SVG cleaning. This property is central to the proposed training
strategy. Because the generated vector scene is known exactly, the
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

The value of $kappa$ follows from the standard cubic approximation of one
quadrant of the unit circle @pomaxBezierPrimer. Consider the arc from
$(1, 0)$ to $(0, 1)$ represented by a cubic Bezier curve with control points
$(1, kappa)$ and $(kappa, 1)$, which enforces the correct endpoint tangents. At
$t = 1 / 2$, the cubic Bezier formula gives the midpoint
$
  (1 / 2 + 3 kappa / 8, 1 / 2 + 3 kappa / 8).
$
If this point is constrained to lie on the circular diagonal
$(sqrt(2) / 2, sqrt(2) / 2)$, then
$
  1 / 2 + 3 kappa / 8 = sqrt(2) / 2
$
and therefore
$
  kappa = frac(4, 3) (sqrt(2) - 1) approx 0.5522847498.
$
Scaling the same construction along the horizontal and vertical axes gives the
ellipse approximation used by the generator.

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


== Stage 2 training schedule

The raster-to-vector model is trained in two consecutive phases. The first
phase consists of pretraining on synthetic data generated procedurally in the
Bezier representation. The second phase consists of fine-tuning on the SVG
dataset derived from real vector graphics. This training schedule is motivated
by the observation that synthetic data and real SVG data provide complementary
advantages. Synthetic data offer unlimited quantity and precise control over
geometric variation, while real SVG data provide more realistic structure,
stylistic diversity, and distributional properties closer to the target use
case. This distinction is important because automatic vectorization is
underdetermined from pixels alone: when the vector scene is generated
procedurally, the exact geometric target is known by construction, whereas for
ordinary raster images there may be many plausible vector explanations
@selinger2003potrace@dziuba2023imagevectorization.

=== Synthetic pretraining

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


= Experiments and Evaluation

This chapter evaluates the two stages of the proposed pipeline and the design
choices that connect them. The experiments follow the research questions from
the introduction: adaptation of the text-to-raster model, the usefulness of
synthetic Bezier pretraining, comparison with classical and neural
vectorization systems, and the trade-off between raster fidelity, validity,
compactness, and editability.

== Alternatives to the proposed decomposition

The first alternative is to adapt a pretrained text-to-raster model directly
into a text-to-bezier model. This would be conceptually attractive, because it
would collapse the whole pipeline into one model while preserving the semantic
knowledge of the pretrained generator. An experiment with this
approach was performed by comparing a model initialized from pretrained
text-to-raster weights with a model whose parameters were reset before
training. The resulting optimization curves are shown in
@fig:pretrained-vs-reset-loss. Over the shared training interval, the reset
model learns faster and reaches a lower training loss than the model initialized
from raster-generation weights. This indicates that the pretrained weights do
not provide a useful initialization for Bezier prediction in this setting.
Although the original model has learned a strong representation for raster
image generation, that task requires a substantially different internal
representation from the one needed to predict structured Bezier control points
and attributes. The approach is therefore not used as the main method, although
it remains a possible large-scale direction if substantially more paired
text-vector data become available.

#figure(
  image("assets/wandb/pretrained-vs-reset_train_loss.pdf", width: 90%),
  caption: [Direct adaptation of raster-generation weights to Bezier prediction.],
) <fig:pretrained-vs-reset-loss>

The second alternative is to rely on existing vectorizers. The proposed method
is compared against both classical raster-to-vector conversion tools and recent
neural systems such as OmniSVG @yang2025omnisvg and StarVector
@rodriguez2024starvector. The comparison distinguishes
in-distribution performance from out-of-distribution behavior by evaluating
methods on both SVG validation samples and synthetic raster images generated
from known vector ground truth. This setup makes it possible to test whether
each method reconstructs the original vector structure or overfits to visible
pixel artifacts.

== Stage 1 evaluation <sec:stage1-evaluation>

The comparison of several Stage 1 variants is shown qualitatively in
@tab:stage1-raster-examples and quantitatively in @tab:stage1-benchmark. The
compared variants include the base Z-Image model, prompt-prefixing strategies,
the accelerated `Z-Image-Turbo` model, and a LoRA adaptation applied
to the turbo pipeline. Higher CLIP and DINO similarity indicate better alignment
with the reference images, whereas lower vectorization MSE indicates that the
generated raster outputs are easier to convert in the second stage. CLIP-based
similarity uses a joint image-text representation learned from natural-language
supervision and is therefore useful for measuring semantic alignment
@radford2021learning. DINO-based similarity uses self-supervised visual
features intended to transfer across image distributions and tasks
@oquab2023dinov2. Both types of feature-space similarity are also relevant
because they have been reported to correlate well with human preference in
vector-graphics evaluation @rodriguez2024starvector.

The vectorization MSE is a round-trip traceability metric. For each generated
raster image, the benchmark first converts the image to SVG using `vtracer`, an
open-source raster-to-vector converter that traces color raster images into SVG
paths @visioncortexVtracer, with its default command-line settings. The
resulting SVG is then rasterized back to the original image resolution on a
white background. The score is the mean
squared error between the original generated RGB image and this rerendered
image,
$ 1 / (3 H W) sum_(c, y, x) (I_(c,y,x) - hat(I)_(c,y,x))^2 $,
where pixel values are measured in the usual 0--255 RGB range. This metric does
not compare the generated image to the reference image directly. Instead, it
measures how much visual information is lost when the image is approximated by
a standard vectorization tool, so lower values indicate images whose shapes and
colors are easier to represent as clean vector graphics.

This metric should not be interpreted as a complete measure of vector quality.
As noted in prior vectorization work, a low pixel error can still correspond to
an overly complex SVG with many redundant paths, while a compact and editable
SVG may differ slightly at the pixel level @selinger2003potrace
@rodriguez2024starvector. The metric is therefore used here as a practical
proxy for traceability, not as a replacement for evaluating path count,
primitive structure, or editability.

#figure(
  table(
    columns: (2.5fr, 1fr, 1fr, 1fr, 1fr),
    align: (left + horizon, center, center, center, center),
    inset: 4pt,
    stroke: (x, y) => (
      left: none,
      top: if y == 0 {
        none
      } else {
        0.4pt
      },
    ),
    [Reference],
    image("assets/raster/reference/0001.png", width: 100%),
    image("assets/raster/reference/0002.png", width: 100%),
    image("assets/raster/reference/0003.png", width: 100%),
    image("assets/raster/reference/0004.png", width: 100%),

    [Z-Image Base],
    image("assets/raster/base/0001.png", width: 100%),
    image("assets/raster/base/0002.png", width: 100%),
    image("assets/raster/base/0003.png", width: 100%),
    image("assets/raster/base/0004.png", width: 100%),

    [Z-Image Base\ prefixed],
    image("assets/raster/base_prefixed/0001.png", width: 100%),
    image("assets/raster/base_prefixed/0002.png", width: 100%),
    image("assets/raster/base_prefixed/0003.png", width: 100%),
    image("assets/raster/base_prefixed/0004.png", width: 100%),

    [Z-Image Base\ prefixed + LoRA],
    image("assets/raster/base_prefixed_lora/0001.png", width: 100%),
    image("assets/raster/base_prefixed_lora/0002.png", width: 100%),
    image("assets/raster/base_prefixed_lora/0003.png", width: 100%),
    image("assets/raster/base_prefixed_lora/0004.png", width: 100%),

    [Z-Image Turbo],
    image("assets/raster/turbo/0001.png", width: 100%),
    image("assets/raster/turbo/0002.png", width: 100%),
    image("assets/raster/turbo/0003.png", width: 100%),
    image("assets/raster/turbo/0004.png", width: 100%),

    [Z-Image Turbo\ prefixed],
    image("assets/raster/turbo_prefixed/0001.png", width: 100%),
    image("assets/raster/turbo_prefixed/0002.png", width: 100%),
    image("assets/raster/turbo_prefixed/0003.png", width: 100%),
    image("assets/raster/turbo_prefixed/0004.png", width: 100%),

    [Z-Image Turbo\ prefixed + LoRA],
    image("assets/raster/turbo_prefixed_lora/0001.png", width: 100%),
    image("assets/raster/turbo_prefixed_lora/0002.png", width: 100%),
    image("assets/raster/turbo_prefixed_lora/0003.png", width: 100%),
    image("assets/raster/turbo_prefixed_lora/0004.png", width: 100%),

    table.cell(colspan: 5, inset: 1.5pt)[],

    [OmniSVG1.1 8B],
    image("assets/raster/omnisvg_8b/0001.png", width: 100%),
    image("assets/raster/omnisvg_8b/0002.png", width: 100%),
    image("assets/raster/omnisvg_8b/0003.png", width: 100%),
    image("assets/raster/omnisvg_8b/0004.png", width: 100%),

    [OmniSVG1.1 4B],
    image("assets/raster/omnisvg_4b/0001.png", width: 100%),
    image("assets/raster/omnisvg_4b/0002.png", width: 100%),
    image("assets/raster/omnisvg_4b/0003.png", width: 100%),
    image("assets/raster/omnisvg_4b/0004.png", width: 100%),
  ),
  caption: [Qualitative Stage 1 comparison of text-to-raster model variants.],
) <tab:stage1-raster-examples>

#figure(
  table(
    columns: (3.1fr, 1fr, 1fr, 1fr),
    align: (left + horizon, center, center, center),
    inset: 6pt,
    stroke: (x, y) => (
      left: none,
      top: if y == 0 { none } else { 0.4pt },
    ),
    table.header(
      [Variant],
      [#text(size: 8pt)[CLIP\ similarity] ↑],
      [#text(size: 8pt)[DINO\ similarity] ↑],
      [#text(size: 8pt)[Vectorization MSE] ↓],
    ),
    [Z-Image Base], [0.818210], [0.509159], [266.565137],
    [Z-Image Base prefixed], [0.819865], [0.545802], [230.160058],
    [Z-Image Base prefixed + LoRA], [#strong[0.882769]], [#strong[0.617481]], [145.452631],
    [Z-Image Turbo], [0.826786], [0.509892], [227.691742],
    [Z-Image Turbo prefixed], [0.871237], [0.583856], [142.711678],
    [Z-Image Turbo prefixed + LoRA], [0.879104], [0.600208], [143.174617],
    table.cell(colspan: 4, inset: 1.5pt)[],
    [OmniSVG1.1 8B], [0.833605], [0.425360], [#strong[51.145968]],
    [OmniSVG1.1 4B], [0.828205], [0.391314], [57.620655],
  ),
  caption: [Stage 1 benchmark of text-to-raster model variants.],
) <tab:stage1-benchmark>

An additional ablation study was performed for the Stage 1 LoRA adaptation in
order to evaluate the effect of training duration and LoRA rank. Three LoRA
ranks, namely 4, 16, and 64, were evaluated at checkpoints saved every 500
steps. For the files without an explicit checkpoint suffix, the final model is
interpreted as the 5000-step checkpoint. The resulting CLIP similarity, DINO
similarity, and vectorization MSE values are summarized together in
@tab:lora-ablation. Due to computational constraints, the ablation was
evaluated on a subset of 100 validation samples rather than on the full
validation set of 1010 samples. The explored grid of ranks and checkpoints is
therefore intentionally coarse; a more fine-grained sweep over ranks, training
durations, and sampling seeds would provide a more precise model-selection
criterion, but was outside the available compute budget.

#let lora-table-text(body, weight: "regular") = text(size: 10pt, weight: weight, body)

#pagebreak()
#v(1fr)

#figure(
  table(
    columns: (1fr, 1.8fr, 1fr, 1fr, 1fr),
    align: (left, left, center, center, center),
    inset: 6pt,
    stroke: (x, y) => (
      left: none,
      top: if (y == 1 and x >= 2) or calc.rem(y - 2, 3) == 0 { 0.4pt } else { none },
    ),
    table.header(
      table.cell(rowspan: 2)[Time steps],
      table.cell(rowspan: 2)[Metric],
      table.cell(colspan: 3)[LoRA Rank],
      [4],
      [16],
      [64],
    ),
    table.cell(rowspan: 3, align: left + horizon)[500],
    lora-table-text[CLIP similarity ↑],
    lora-table-text[0.880],
    lora-table-text[0.872],
    lora-table-text[0.885],
    lora-table-text[DINO similarity ↑], lora-table-text[0.607], lora-table-text[0.599], lora-table-text[0.599],
    lora-table-text[Vectorization MSE ↓], lora-table-text[299.456], lora-table-text[187.121], lora-table-text[205.943],

    table.cell(rowspan: 3, align: left + horizon)[1000],
    lora-table-text[CLIP similarity ↑],
    lora-table-text[0.886],
    lora-table-text[0.887],
    lora-table-text[0.885],
    lora-table-text[DINO similarity ↑], lora-table-text[0.616], lora-table-text[0.621], lora-table-text[0.620],
    lora-table-text[Vectorization MSE ↓], lora-table-text[328.455], lora-table-text[128.576], lora-table-text[199.657],

    table.cell(rowspan: 3, align: left + horizon)[1500],
    lora-table-text[CLIP similarity ↑],
    lora-table-text[0.886],
    lora-table-text[0.887],
    lora-table-text[0.884],
    lora-table-text[DINO similarity ↑], lora-table-text[0.622], lora-table-text[0.618], lora-table-text[0.615],
    lora-table-text[Vectorization MSE ↓], lora-table-text[206.473], lora-table-text[143.796], lora-table-text[346.287],

    table.cell(rowspan: 3, align: left + horizon)[2000],
    lora-table-text[CLIP similarity ↑],
    lora-table-text[0.885],
    lora-table-text[0.885],
    lora-table-text[0.887],
    lora-table-text[DINO similarity ↑], lora-table-text[0.627], lora-table-text[0.614], lora-table-text[0.625],
    lora-table-text[Vectorization MSE ↓], lora-table-text[142.765], lora-table-text[97.675], lora-table-text[168.836],

    table.cell(rowspan: 3, align: left + horizon)[2500],
    lora-table-text[CLIP similarity ↑],
    lora-table-text[0.888],
    lora-table-text[0.888],
    lora-table-text[0.882],
    lora-table-text[DINO similarity ↑], lora-table-text[0.626], lora-table-text[0.627], lora-table-text[0.620],
    lora-table-text[Vectorization MSE ↓],
    lora-table-text[174.613],
    lora-table-text[92.145],
    lora-table-text[273.638],

    table.cell(rowspan: 3, align: left + horizon)[3000],
    lora-table-text[CLIP similarity ↑],
    lora-table-text[0.887],
    lora-table-text[0.887],
    lora-table-text(weight: "bold")[0.888],
    lora-table-text[DINO similarity ↑],
    lora-table-text[0.626],
    lora-table-text(weight: "bold")[0.635],
    lora-table-text[0.632],
    lora-table-text[Vectorization MSE ↓],
    lora-table-text[173.293],
    lora-table-text(weight: "bold")[92.112],
    lora-table-text[132.058],

    table.cell(rowspan: 3, align: left + horizon)[3500],
    lora-table-text[CLIP similarity ↑],
    lora-table-text[0.886],
    lora-table-text[0.887],
    lora-table-text[0.888],
    lora-table-text[DINO similarity ↑], lora-table-text[0.626], lora-table-text[0.630], lora-table-text[0.633],
    lora-table-text[Vectorization MSE ↓], lora-table-text[166.197], lora-table-text[93.775], lora-table-text[157.162],

    table.cell(rowspan: 3, align: left + horizon)[5000],
    lora-table-text[CLIP similarity ↑],
    lora-table-text[0.886],
    lora-table-text[0.886],
    lora-table-text[0.883],
    lora-table-text[DINO similarity ↑], lora-table-text[0.620], lora-table-text[0.631], lora-table-text[0.617],
    lora-table-text[Vectorization MSE ↓], lora-table-text[178.814], lora-table-text[166.223], lora-table-text[172.532],
  ),
  caption: [Stage 1 LoRA ablation results.],
) <tab:lora-ablation>

#v(1fr)
#pagebreak()

The results suggest that prompt prefixing has a substantial effect, especially
for the turbo model. The best overall semantic similarity is obtained by the
`Base prefixed + LoRA` configuration, while the lowest
vectorization error is achieved by `Turbo prefixed`. This indicates that the
adapted LoRA model improves perceptual alignment with the references, but its
advantage with respect to downstream vectorization should be verified on a
larger evaluation. The `Base prefixed + LoRA` configuration is therefore used
as the reference text-to-raster setting in the final end-to-end evaluation,
where image quality is prioritized. At the same time, `Turbo prefixed + LoRA`
remains a reasonable practical alternative when inference speed is more
important, because it preserves most of the semantic-similarity gain while
using the accelerated turbo sampler.

Based on the rank and checkpoint ablation, the LoRA model with rank 16 at
3000 training steps was selected for subsequent Stage 1 experiments. This
checkpoint achieves the highest DINO similarity and the lowest vectorization
MSE in the ablation. Although its CLIP similarity is slightly lower than the
best observed value, the difference is small, making this checkpoint a
reasonable choice for subsequent experiments.


== Stage 2 vectorizer evaluation

The Stage 2 experiments evaluate the conditional flow-matching vectorizer.
First, architectural ablations compare the flow-matching formulation with an
autoregressive variant of comparable size, and measure the effect of image
conditioning. The conditioning ablations compare the full model with a model
trained without an image encoder while keeping the vectorizer backbone as
constant as possible.
Together, these experiments clarify how much of the performance is due to the
flow-matching objective, the transformer backbone, and the pretrained visual
representation.

To separate the effect of the generative formulation from the effect of model
capacity, the flow-matching vectorizer was also compared with an autoregressive
variant. The autoregressive model uses the same hidden size, number of layers,
maximum sequence length, and DINOv3 image encoder, but predicts the Bezier
sequence step by step rather than learning a continuous denoising vector field.
The comparison in @fig:flow-matching-vs-autoregressive-mse is capped at the
first 250k training steps, where both runs have logged train and validation
image-space MSE. The autoregressive model reduces the reconstruction error
during training, but remains consistently worse than the flow-matching model
on both splits. This suggests that, for this fixed Bezier representation and
model scale, the flow-matching objective provides a more effective training
signal than next-step autoregressive prediction.

#figure(
  image("assets/wandb/flow-matching-vs-autoregressive_image_mse.pdf", width: 90%),
  caption: [Flow-matching and autoregressive vectorizer comparison.],
) <fig:flow-matching-vs-autoregressive-mse>

The conditioning mechanism was further evaluated by comparing the full
architecture with a variant trained without the image encoder. In the ablated
model, the vectorizer still learns the distribution of valid Bezier sequences,
but it lacks direct visual information about the raster input. This comparison
therefore tests whether the model is merely learning an unconditional vector
graphics prior, or whether the DINOv3 image features provide useful
input-specific guidance.

The results in @fig:image-encoder-ablation-mse and
@fig:image-encoder-ablation-loss are limited to the first 150k training steps.
The model with the image encoder reaches lower train and validation
reconstruction MSE and also maintains a lower training loss over the shared
interval. The difference is especially important on the validation split,
where the image-conditioned model can adapt the generated Bezier curves to the
observed raster image instead of relying only on the learned shape prior. This
supports the use of a pretrained image encoder as a central part of the
conditional vectorizer.

#figure(
  image("assets/wandb/image-encoder-ablation_image_mse.pdf", width: 90%),
  caption: [Image reconstruction error with and without image conditioning.],
) <fig:image-encoder-ablation-mse>

#figure(
  image("assets/wandb/image-encoder-ablation_train_loss.pdf", width: 90%),
  caption: [Training loss with and without image conditioning.],
) <fig:image-encoder-ablation-loss>

The selected architecture is then used for synthetic pretraining before
fine-tuning on the SVG Repo data. This training setup tests the central
hypothesis that synthetic Bezier data provide a useful geometric prior even
though they are simpler than real SVG graphics. The pretraining run used a
single NVIDIA H200 GPU for approximately 10 days with batch size 256 and
FlashAttention 2 enabled.

=== Synthetic pretraining results

The optimization dynamics of this pretraining run are summarized in
@fig:vectorizer-pretraining-loss and @fig:vectorizer-pretraining-mse. The
training objective decreases rapidly during the initial phase and then enters
a slower refinement regime, indicating that the model first learns coarse
Bezier-structure prediction before improving smaller geometric and appearance
errors. The image-space MSE is measured by rendering predicted vectors back to
raster images and comparing them with the corresponding synthetic targets. It
therefore provides a complementary reconstruction-oriented view of pretraining
quality, in addition to the direct flow-matching loss.

#figure(
  image("assets/wandb/classic-serenity-74_train_loss.pdf", width: 90%),
  caption: [Synthetic pretraining loss of the raster-to-vector model.],
) <fig:vectorizer-pretraining-loss>

#figure(
  image("assets/wandb/classic-serenity-74_image_mse.pdf", width: 90%),
  caption: [Image reconstruction error during synthetic pretraining.],
) <fig:vectorizer-pretraining-mse>

Qualitative samples from the final synthetic pretraining checkpoint are shown
in @tab:vectorizer-pretraining-samples. The training examples indicate that
the model has learned to vectorize samples from the synthetic generator: the
predicted Bezier representations preserve the main silhouettes, colors, and
compound-shape structure of the references. The validation examples further
suggest that this learned geometric prior generalizes reasonably well to data
from the SVG dataset, despite the visual and structural differences between
procedurally generated scenes and real vector graphics. In each case, the
generated image is produced by sampling the flow-matching vectorizer
conditioned on the corresponding raster reference and then rendering the
predicted Bezier representation back to an image.

#let pretraining-sample(path) = box(
  width: 100%,
  image(path, width: 100%),
)

#figure(
  table(
    columns: (1.25fr, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr),
    align: (left + horizon, center, center, center, center, center, center),
    inset: 3pt,
    stroke: (x, y) => (
      left: none,
      top: if y == 0 { none } else { 0.4pt },
    ),
    [Train reference],
    pretraining-sample("assets/pretraining/train/ref/0000.png"),
    pretraining-sample("assets/pretraining/train/ref/0001.png"),
    pretraining-sample("assets/pretraining/train/ref/0002.png"),
    pretraining-sample("assets/pretraining/train/ref/0003.png"),
    pretraining-sample("assets/pretraining/train/ref/0004.png"),
    pretraining-sample("assets/pretraining/train/ref/0005.png"),

    [Train generated],
    pretraining-sample("assets/pretraining/train/generated/0000.png"),
    pretraining-sample("assets/pretraining/train/generated/0001.png"),
    pretraining-sample("assets/pretraining/train/generated/0002.png"),
    pretraining-sample("assets/pretraining/train/generated/0003.png"),
    pretraining-sample("assets/pretraining/train/generated/0004.png"),
    pretraining-sample("assets/pretraining/train/generated/0005.png"),

    [SVG validation reference],
    pretraining-sample("assets/pretraining/val/ref/0000.png"),
    pretraining-sample("assets/pretraining/val/ref/0001.png"),
    pretraining-sample("assets/pretraining/val/ref/0002.png"),
    pretraining-sample("assets/pretraining/val/ref/0003.png"),
    pretraining-sample("assets/pretraining/val/ref/0004.png"),
    pretraining-sample("assets/pretraining/val/ref/0005.png"),

    [SVG validation generated],
    pretraining-sample("assets/pretraining/val/generated/0000.png"),
    pretraining-sample("assets/pretraining/val/generated/0001.png"),
    pretraining-sample("assets/pretraining/val/generated/0002.png"),
    pretraining-sample("assets/pretraining/val/generated/0003.png"),
    pretraining-sample("assets/pretraining/val/generated/0004.png"),
    pretraining-sample("assets/pretraining/val/generated/0005.png"),
  ),
  caption: [Qualitative samples from the synthetic pretraining checkpoint.],
) <tab:vectorizer-pretraining-samples>

=== Fine-tuning results

After synthetic pretraining, the same raster-to-vector model is fine-tuned on
the SVG Repo dataset. The fine-tuning dynamics are shown in
@fig:vectorizer-finetuning-loss and @fig:vectorizer-finetuning-mse. The
training loss starts substantially lower than in the synthetic pretraining run,
which is consistent with initialization from the pretrained checkpoint, and
continues to decrease during adaptation to real vector graphics. The
image-space MSE remains noisier than the direct training objective, but it
tracks the rendered reconstruction quality on both training and validation
samples and therefore complements the loss curve.

#figure(
  image("assets/wandb/floral-glade-79_train_loss.pdf", width: 90%),
  caption: [Fine-tuning loss of the raster-to-vector model on SVG Repo data.],
) <fig:vectorizer-finetuning-loss>

#figure(
  image("assets/wandb/floral-glade-79_image_mse.pdf", width: 90%),
  caption: [Image reconstruction error during fine-tuning on SVG Repo data.],
) <fig:vectorizer-finetuning-mse>


=== Flow-matching inference ablation

Inference requires numerical integration of the learned velocity field. The
number of integration steps directly affects runtime and reconstruction
quality. The inference ablation therefore evaluates several fixed step counts
and measures the resulting output quality, topology, and rendering error.
This experiment is important because an excessively small number of steps may
produce unstable or incomplete geometry, while too many steps increase runtime
without necessarily improving the final SVG.

=== Vectorization benchmark on controlled inputs

Before evaluating the full text-to-vector pipeline, the vectorization stage is
evaluated in isolation. This benchmark uses clean raster inputs obtained either
from the SVG validation split or from the synthetic generator. These inputs are
not produced by the text-to-raster model; they are controlled sources with
known reference SVGs or known procedural structure. The purpose of this part is
therefore to measure raster-to-vector quality under reproducible conditions,
rather than to claim end-to-end pipeline performance.

The comparison includes classical vectorization tools and recent neural
SVG-generation systems. It emphasizes not only pixel-level reconstruction, but
also properties important for editable vector graphics, such as path count,
node count, topological cleanliness, robustness to controlled input variation,
and ease of manual editing.

The vectorization comparison is performed with an evaluation script that
renders each reference SVG and each generated SVG at a fixed resolution of
1024 pixels, compares the rendered RGB images, and records both image-space
and structure-related statistics. The comparison contains four methods: the
proposed flow-matching vectorizer,
OmniSVG @yang2025omnisvg, StarVector @rodriguez2024starvector, and `vtracer`
@visioncortexVtracer. All methods are evaluated on the same reference set and
with the same rasterization settings.

The quantitative comparison is paired with two qualitative grids. The first
grid uses samples from the SVG validation split. These examples are useful for
checking performance on the target distribution, but they should be
interpreted as in-distribution examples: the proposed model is trained on the
same dataset family, and large external SVG models may also have been exposed
to visually similar icon data during pretraining. The validation grid therefore
shows how well the methods handle the type of data used in the main benchmark,
rather than proving broad vectorization ability.

#let vectorization-sample(path) = box(
  width: 100%,
  image(path, width: 100%),
)

#figure(
  table(
    columns: (2.2fr, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr),
    align: (left + horizon, center, center, center, center, center, center),
    inset: 3pt,
    stroke: (x, y) => (
      left: none,
      top: if y == 0 { none } else { 0.4pt },
    ),
    [Reference],
    vectorization-sample("assets/vectorization_qualitative/validation/reference/0000.png"),
    vectorization-sample("assets/vectorization_qualitative/validation/reference/0001.png"),
    vectorization-sample("assets/vectorization_qualitative/validation/reference/0002.png"),
    vectorization-sample("assets/vectorization_qualitative/validation/reference/0003.png"),
    vectorization-sample("assets/vectorization_qualitative/validation/reference/0004.png"),
    vectorization-sample("assets/vectorization_qualitative/validation/reference/0005.png"),

    [Ours model],
    vectorization-sample("assets/vectorization_qualitative/validation/proposed/0000.png"),
    vectorization-sample("assets/vectorization_qualitative/validation/proposed/0001.png"),
    vectorization-sample("assets/vectorization_qualitative/validation/proposed/0002.png"),
    vectorization-sample("assets/vectorization_qualitative/validation/proposed/0003.png"),
    vectorization-sample("assets/vectorization_qualitative/validation/proposed/0004.png"),
    vectorization-sample("assets/vectorization_qualitative/validation/proposed/0005.png"),

    [OmniSVG1.1 4B],
    vectorization-sample("assets/vectorization_qualitative/validation/omnisvg_4b/0000.png"),
    vectorization-sample("assets/vectorization_qualitative/validation/omnisvg_4b/0001.png"),
    vectorization-sample("assets/vectorization_qualitative/validation/omnisvg_4b/0002.png"),
    vectorization-sample("assets/vectorization_qualitative/validation/omnisvg_4b/0003.png"),
    vectorization-sample("assets/vectorization_qualitative/validation/omnisvg_4b/0004.png"),
    vectorization-sample("assets/vectorization_qualitative/validation/omnisvg_4b/0005.png"),

    [OmniSVG1.1 8B],
    vectorization-sample("assets/vectorization_qualitative/validation/omnisvg_8b/0000.png"),
    vectorization-sample("assets/vectorization_qualitative/validation/omnisvg_8b/0001.png"),
    vectorization-sample("assets/vectorization_qualitative/validation/omnisvg_8b/0002.png"),
    vectorization-sample("assets/vectorization_qualitative/validation/omnisvg_8b/0003.png"),
    vectorization-sample("assets/vectorization_qualitative/validation/omnisvg_8b/0004.png"),
    vectorization-sample("assets/vectorization_qualitative/validation/omnisvg_8b/0005.png"),

    [StarVector 1B],
    vectorization-sample("assets/vectorization_qualitative/validation/starvector_1b/0000.png"),
    vectorization-sample("assets/vectorization_qualitative/validation/starvector_1b/0001.png"),
    vectorization-sample("assets/vectorization_qualitative/validation/starvector_1b/0002.png"),
    vectorization-sample("assets/vectorization_qualitative/validation/starvector_1b/0003.png"),
    vectorization-sample("assets/vectorization_qualitative/validation/starvector_1b/0004.png"),
    vectorization-sample("assets/vectorization_qualitative/validation/starvector_1b/0005.png"),

    [StarVector 8B],
    vectorization-sample("assets/vectorization_qualitative/validation/starvector_8b/0000.png"),
    vectorization-sample("assets/vectorization_qualitative/validation/starvector_8b/0001.png"),
    vectorization-sample("assets/vectorization_qualitative/validation/starvector_8b/0002.png"),
    vectorization-sample("assets/vectorization_qualitative/validation/starvector_8b/0003.png"),
    vectorization-sample("assets/vectorization_qualitative/validation/starvector_8b/0004.png"),
    vectorization-sample("assets/vectorization_qualitative/validation/starvector_8b/0005.png"),
  ),
  caption: [Qualitative comparison on SVG validation samples.],
) <tab:vectorization-qualitative-validation>

The second qualitative grid uses samples from the synthetic generator. In
this setting, the reference vector structure is produced by a controlled
procedural process rather than collected from the same icon distribution as
the validation set. This makes the comparison a more direct test of general
raster-to-vector capability: the methods must recover clean geometric
structure from rendered images whose underlying shapes, holes, intersections,
and curve configurations are known. The examples are selected by fixed criteria
such as sample index and input source, which avoids choosing only visually
favorable cases.

#figure(
  table(
    columns: (2.2fr, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr),
    align: (left + horizon, center, center, center, center, center, center),
    inset: 3pt,
    stroke: (x, y) => (
      left: none,
      top: if y == 0 { none } else { 0.4pt },
    ),
    [Reference],
    vectorization-sample("assets/vectorization_qualitative/synthetic/reference/0000.png"),
    vectorization-sample("assets/vectorization_qualitative/synthetic/reference/0001.png"),
    vectorization-sample("assets/vectorization_qualitative/synthetic/reference/0002.png"),
    vectorization-sample("assets/vectorization_qualitative/synthetic/reference/0003.png"),
    vectorization-sample("assets/vectorization_qualitative/synthetic/reference/0004.png"),
    vectorization-sample("assets/vectorization_qualitative/synthetic/reference/0005.png"),

    [Ours model],
    vectorization-sample("assets/vectorization_qualitative/synthetic/proposed/0000.png"),
    vectorization-sample("assets/vectorization_qualitative/synthetic/proposed/0001.png"),
    vectorization-sample("assets/vectorization_qualitative/synthetic/proposed/0002.png"),
    vectorization-sample("assets/vectorization_qualitative/synthetic/proposed/0003.png"),
    vectorization-sample("assets/vectorization_qualitative/synthetic/proposed/0004.png"),
    vectorization-sample("assets/vectorization_qualitative/synthetic/proposed/0005.png"),

    [OmniSVG1.1 4B],
    vectorization-sample("assets/vectorization_qualitative/synthetic/omnisvg_4b/0000.png"),
    vectorization-sample("assets/vectorization_qualitative/synthetic/omnisvg_4b/0001.png"),
    vectorization-sample("assets/vectorization_qualitative/synthetic/omnisvg_4b/0002.png"),
    vectorization-sample("assets/vectorization_qualitative/synthetic/omnisvg_4b/0003.png"),
    vectorization-sample("assets/vectorization_qualitative/synthetic/omnisvg_4b/0004.png"),
    vectorization-sample("assets/vectorization_qualitative/synthetic/omnisvg_4b/0005.png"),

    [OmniSVG1.1 8B],
    vectorization-sample("assets/vectorization_qualitative/synthetic/omnisvg_8b/0000.png"),
    vectorization-sample("assets/vectorization_qualitative/synthetic/omnisvg_8b/0001.png"),
    vectorization-sample("assets/vectorization_qualitative/synthetic/omnisvg_8b/0002.png"),
    vectorization-sample("assets/vectorization_qualitative/synthetic/omnisvg_8b/0003.png"),
    vectorization-sample("assets/vectorization_qualitative/synthetic/omnisvg_8b/0004.png"),
    vectorization-sample("assets/vectorization_qualitative/synthetic/omnisvg_8b/0005.png"),

    [StarVector 1B],
    vectorization-sample("assets/vectorization_qualitative/synthetic/starvector_1b/0000.png"),
    vectorization-sample("assets/vectorization_qualitative/synthetic/starvector_1b/0001.png"),
    vectorization-sample("assets/vectorization_qualitative/synthetic/starvector_1b/0002.png"),
    vectorization-sample("assets/vectorization_qualitative/synthetic/starvector_1b/0003.png"),
    vectorization-sample("assets/vectorization_qualitative/synthetic/starvector_1b/0004.png"),
    vectorization-sample("assets/vectorization_qualitative/synthetic/starvector_1b/0005.png"),

    [StarVector 8B],
    vectorization-sample("assets/vectorization_qualitative/synthetic/starvector_8b/0000.png"),
    vectorization-sample("assets/vectorization_qualitative/synthetic/starvector_8b/0001.png"),
    vectorization-sample("assets/vectorization_qualitative/synthetic/starvector_8b/0002.png"),
    vectorization-sample("assets/vectorization_qualitative/synthetic/starvector_8b/0003.png"),
    vectorization-sample("assets/vectorization_qualitative/synthetic/starvector_8b/0004.png"),
    vectorization-sample("assets/vectorization_qualitative/synthetic/starvector_8b/0005.png"),
  ),
  caption: [Qualitative comparison on synthetic-generator samples.],
) <tab:vectorization-qualitative-synthetic>

The quantitative metrics are computed after rendering both SVGs to RGB images
at the same resolution. MSE is the mean squared difference between
corresponding RGB pixel values, reported on the 0--255 intensity scale. It
measures direct raster reconstruction error and is sensitive to both color
differences and small spatial misalignments. SSIM compares the rendered images
using luminance, contrast, and covariance statistics on normalized RGB values;
it is included because two vectorizations can have similar pixel error while
preserving global structure to different degrees @wang2004ssim. In this
implementation SSIM is computed globally per channel and then averaged, so it
should be interpreted as a coarse structural score rather than as a full
windowed perceptual metric.

The remaining fidelity metrics emphasize foreground shape and boundary
alignment. Mask IoU thresholds each rendered image into foreground and
background, treating pixels darker than the foreground threshold as foreground,
and measures the intersection-over-union of the two masks, corresponding to
the Jaccard similarity coefficient @jaccard1901distribution. Boundary F1 first
extracts edge points from the rendered images and then measures precision and
recall under a fixed pixel tolerance, following the common boundary-evaluation
idea of matching contours within a localization tolerance
@martin2004boundaries. The tables report the 2 px tolerance, which rewards
methods whose contours are close to the reference even when the filled regions
are not identical. Chamfer distance averages the nearest-edge distance in both
directions between the reference and generated contours @borgefors1988chamfer,
whereas Hausdorff distance reports the worst nearest-edge discrepancy
@huttenlocher1993hausdorff. Chamfer therefore measures typical contour
alignment, while Hausdorff is more sensitive to outliers such as missing
strokes, distant artifacts, or a single badly placed shape.

SVG validity and complexity are reported separately from visual fidelity. The
valid SVG rate is the fraction of generated files that can be rendered without
error; missing or non-renderable files are replaced by a white image for
fidelity scoring but are counted as failures in the validity table. SVG bytes,
element count, path count, and path-command count are simple proxies for output
complexity. Lower values indicate a more compact and potentially more editable
SVG only when the corresponding fidelity metrics remain competitive, because a
trivially simple file can also be inaccurate.

#figure(
  table(
    columns: (1.4fr, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr),
    align: (left, center, center, center, center, center, center),
    inset: 4pt,
    stroke: (x, y) => (
      left: if x == 0 { none } else { 0.4pt },
      top: if y == 0 { none } else { 0.4pt },
    ),
    table.header(
      [Method], [MSE (0--255) ↓], [SSIM ↑], [Mask IoU ↑], [Boundary F1 at 2 px ↑], [Chamfer px ↓], [Hausdorff px ↓]
    ),
    [Ours model], [7107.52], [0.653], [0.644], [0.324], [16.04], [103.24],
    [OmniSVG1.1 4B], [7696.39], [0.621], [0.631], [0.538], [17.84], [145.46],
    [OmniSVG1.1 8B], [8425.56], [0.589], [0.608], [0.516], [18.89], [149.69],
    [StarVector 1B], [5147.82], [0.652], [0.631], [0.483], [24.26], [143.28],
    [StarVector 8B], [8449.48], [0.461], [0.444], [0.441], [38.75], [229.73],
    [`vtracer`], [92.01], [0.994], [0.984], [0.886], [1.26], [17.20],
  ),
  caption: [Vectorization fidelity on SVG validation samples.],
) <tab:vectorization-fidelity-validation>

#figure(
  table(
    columns: (1.4fr, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr),
    align: (left, center, center, center, center, center, center),
    inset: 4pt,
    stroke: (x, y) => (
      left: if x == 0 { none } else { 0.4pt },
      top: if y == 0 { none } else { 0.4pt },
    ),
    table.header(
      [Method], [MSE (0--255) ↓], [SSIM ↑], [Mask IoU ↑], [Boundary F1 at 2 px ↑], [Chamfer px ↓], [Hausdorff px ↓]
    ),
    [Ours model], [2432.57], [0.725], [0.758], [0.390], [14.50], [108.82],
    [OmniSVG1.1 4B], [9591.49], [0.330], [0.432], [0.461], [32.87], [242.00],
    [OmniSVG1.1 8B], [11024.09], [0.283], [0.407], [0.436], [36.07], [253.81],
    [StarVector 1B], [6339.14], [0.314], [0.297], [0.476], [47.02], [317.25],
    [StarVector 8B], [7536.35], [0.137], [0.104], [0.496], [59.58], [401.38],
    [`vtracer`], [23.95], [0.997], [0.993], [0.920], [1.03], [20.45],
  ),
  caption: [Vectorization fidelity on synthetic-generator samples.],
) <tab:vectorization-fidelity-synthetic>

#figure(
  table(
    columns: (1.4fr, 1fr, 1fr, 1fr, 1fr, 1fr),
    align: (left, center, center, center, center, center),
    inset: 4pt,
    stroke: (x, y) => (
      left: if x == 0 { none } else { 0.4pt },
      top: if y == 0 { none } else { 0.4pt },
    ),
    table.header([Method], [Valid SVG rate ↑], [SVG bytes ↓], [Elements ↓], [Paths ↓], [Path commands ↓]),
    [Ours model], [100.0%], [10329.02], [5.27], [4.27], [102.55],
    [OmniSVG1.1 4B], [99.4%], [5284.03], [5.04], [4.04], [219.53],
    [OmniSVG1.1 8B], [99.3%], [5296.17], [8.62], [7.62], [206.38],
    [StarVector 1B], [79.0%], [1957.71], [9.17], [4.09], [118.60],
    [StarVector 8B], [65.1%], [2220.83], [10.63], [5.36], [213.25],
    [`vtracer`], [100.0%], [14370.11], [10.68], [9.68], [364.55],
  ),
  caption: [SVG validity and complexity on SVG validation samples.],
) <tab:vectorization-complexity-validation>

#figure(
  table(
    columns: (1.4fr, 1fr, 1fr, 1fr, 1fr, 1fr),
    align: (left, center, center, center, center, center),
    inset: 4pt,
    stroke: (x, y) => (
      left: if x == 0 { none } else { 0.4pt },
      top: if y == 0 { none } else { 0.4pt },
    ),
    table.header([Method], [Valid SVG rate ↑], [SVG bytes ↓], [Elements ↓], [Paths ↓], [Path commands ↓]),
    [Ours model], [100.0%], [9090.44], [9.66], [8.66], [90.17],
    [OmniSVG1.1 4B], [99.2%], [9165.78], [13.72], [12.72], [393.75],
    [OmniSVG1.1 8B], [98.6%], [9658.08], [29.95], [28.95], [400.92],
    [StarVector 1B], [43.2%], [4108.50], [30.42], [9.97], [447.18],
    [StarVector 8B], [20.5%], [5354.53], [40.59], [14.22], [962.31],
    [`vtracer`], [100.0%], [24468.73], [14.66], [13.66], [625.33],
  ),
  caption: [SVG validity and complexity on synthetic-generator samples.],
) <tab:vectorization-complexity-synthetic>

The fidelity tables capture visual reconstruction quality, while the
complexity tables capture whether the output is a practical vector graphic.
The proposed model results on the SVG validation split are computed over 1010
pairs. Additional measured values for this run are MAE 31.47, PSNR 11.17 dB,
Boundary F1 0.255 at 1 px and 0.409 at 4 px, mean render time 53.23 ms, and no
rendering errors.
On the synthetic-generator split, the proposed model results are computed over
1000 pairs. Additional measured values for this run are MAE 20.65, PSNR 15.68
dB, Boundary F1 0.336 at 1 px and 0.449 at 4 px, mean render time 53.76 ms, and
no rendering errors.
This separation is important because a method can obtain a low raster error by
creating a very large SVG with many paths or path commands. Conversely, a more
compact SVG may be preferable for editing even when it introduces a small
raster-space error. The interpretation therefore treats the validation results
separately from the synthetic-generator results and reads the corresponding
fidelity and complexity tables together rather than selecting a method from a
single scalar score.

== End-to-end pipeline evaluation

The previous vectorization experiments use either rasterized SVG validation
samples or controlled procedural images. A separate evaluation is needed for
the actual output of the text-to-raster stage, because generated raster images
have a different error profile from both sources. This section is the pipeline
evaluation: unlike the controlled benchmark in the previous section, it
measures vectorization of images rendered by the text-to-raster generator
rather than vectorization of clean dataset or synthetic inputs. The reference
rasters in this evaluation are produced by the `Z-Image Base prefixed + LoRA`
configuration selected in @sec:stage1-evaluation; the turbo-prefixed LoRA
variant discussed there is retained as the faster alternative. The Z-Image
stage may produce anti-aliased contours, slight texture, local color variation,
incomplete boundaries, or other artifacts that are not present in the synthetic
generator, while also differing from clean SVG Repo renderings. This setting
therefore measures the practical interface between the text-to-raster model
and a raster-to-vector converter.

Unlike the controlled vectorization benchmark script, which compares a
generated SVG against a reference SVG after rendering both files, this
experiment uses a separate evaluation script that compares a generated PNG
directly against the SVG obtained from that PNG. Each input raster is rendered
from the Z-Image pipeline, vectorized with `vtracer`, rendered back to a PNG
at the same resolution, and compared with the original raster image. The
reported numbers therefore evaluate raster-vector-raster consistency on
generated images, not semantic agreement with a ground-truth SVG. Qualitative
inspection should focus on whether the vectorization preserves the main shape
layout, fill regions, and contour smoothness, and whether small raster
artifacts are converted into unnecessary paths. Quantitative scores summarize
the same behavior over the full generated set.

#figure(
  table(
    columns: (1.25fr, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr),
    align: (left + horizon, center, center, center, center, center, center),
    inset: 3pt,
    stroke: (x, y) => (
      left: none,
      top: if y == 0 { none } else { 0.4pt },
    ),
    [Reference\ (Z-Image Base\ prefixed + LoRA)],
    vectorization-sample("assets/z_image_vectorization/reference/0000.png"),
    vectorization-sample("assets/z_image_vectorization/reference/0001.png"),
    vectorization-sample("assets/z_image_vectorization/reference/0002.png"),
    vectorization-sample("assets/z_image_vectorization/reference/0003.png"),
    vectorization-sample("assets/z_image_vectorization/reference/0004.png"),
    vectorization-sample("assets/z_image_vectorization/reference/0005.png"),

    [Proposed],
    vectorization-sample("assets/z_image_vectorization/proposed/0000.png"),
    vectorization-sample("assets/z_image_vectorization/proposed/0001.png"),
    vectorization-sample("assets/z_image_vectorization/proposed/0002.png"),
    vectorization-sample("assets/z_image_vectorization/proposed/0003.png"),
    vectorization-sample("assets/z_image_vectorization/proposed/0004.png"),
    vectorization-sample("assets/z_image_vectorization/proposed/0005.png"),

    [OmniSVG1.1 4B],
    vectorization-sample("assets/z_image_vectorization/omnisvg_4b/0000.png"),
    vectorization-sample("assets/z_image_vectorization/omnisvg_4b/0001.png"),
    vectorization-sample("assets/z_image_vectorization/omnisvg_4b/0002.png"),
    vectorization-sample("assets/z_image_vectorization/omnisvg_4b/0003.png"),
    vectorization-sample("assets/z_image_vectorization/omnisvg_4b/0004.png"),
    vectorization-sample("assets/z_image_vectorization/omnisvg_4b/0005.png"),

    [OmniSVG1.1 8B],
    vectorization-sample("assets/z_image_vectorization/omnisvg_8b/0000.png"),
    vectorization-sample("assets/z_image_vectorization/omnisvg_8b/0001.png"),
    vectorization-sample("assets/z_image_vectorization/omnisvg_8b/0002.png"),
    vectorization-sample("assets/z_image_vectorization/omnisvg_8b/0003.png"),
    vectorization-sample("assets/z_image_vectorization/omnisvg_8b/0004.png"),
    vectorization-sample("assets/z_image_vectorization/omnisvg_8b/0005.png"),

    [StarVector 1B],
    vectorization-sample("assets/z_image_vectorization/starvector_1b/0000.png"),
    vectorization-sample("assets/z_image_vectorization/starvector_1b/0001.png"),
    vectorization-sample("assets/z_image_vectorization/starvector_1b/0002.png"),
    vectorization-sample("assets/z_image_vectorization/starvector_1b/0003.png"),
    vectorization-sample("assets/z_image_vectorization/starvector_1b/0004.png"),
    vectorization-sample("assets/z_image_vectorization/starvector_1b/0005.png"),

    [StarVector 8B],
    vectorization-sample("assets/z_image_vectorization/starvector_8b/0000.png"),
    vectorization-sample("assets/z_image_vectorization/starvector_8b/0001.png"),
    vectorization-sample("assets/z_image_vectorization/starvector_8b/0002.png"),
    vectorization-sample("assets/z_image_vectorization/starvector_8b/0003.png"),
    vectorization-sample("assets/z_image_vectorization/starvector_8b/0004.png"),
    vectorization-sample("assets/z_image_vectorization/starvector_8b/0005.png"),

    [`vtracer`],
    vectorization-sample("assets/z_image_vectorization/vtracer/0000.png"),
    vectorization-sample("assets/z_image_vectorization/vtracer/0001.png"),
    vectorization-sample("assets/z_image_vectorization/vtracer/0002.png"),
    vectorization-sample("assets/z_image_vectorization/vtracer/0003.png"),
    vectorization-sample("assets/z_image_vectorization/vtracer/0004.png"),
    vectorization-sample("assets/z_image_vectorization/vtracer/0005.png"),
  ),
  caption: [Qualitative vectorization of generated raster images. The reference
    row contains rasters generated by the `Z-Image Base prefixed + LoRA`
    configuration from @sec:stage1-evaluation.],
) <tab:z-image-raster-vectorization-qualitative>

#figure(
  table(
    columns: (1.4fr, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr),
    align: (left, center, center, center, center, center, center),
    inset: 4pt,
    stroke: (x, y) => (
      left: if x == 0 { none } else { 0.4pt },
      top: if y == 0 { none } else { 0.4pt },
    ),
    table.header(
      [Method], [MSE (0--255) ↓], [MAE (0--255) ↓], [PSNR dB ↑], [SSIM ↑], [Mask IoU ↑], [Boundary F1 at 2 px ↑]
    ),
    [Proposed], [6004.36], [27.31], [12.15], [0.681], [0.651], [0.371],
    [OmniSVG1.1 4B], [8973.13], [38.55], [12.07], [0.550], [0.549], [0.527],
    [OmniSVG1.1 8B], [9923.60], [42.47], [11.58], [0.526], [0.532], [0.503],
    [StarVector 1B], [5057.47], [22.35], [13.25], [0.729], [0.690], [0.417],
    [StarVector 8B], [6592.78], [27.77], [12.08], [0.622], [0.596], [0.427],
    [`vtracer`], [145.45], [2.57], [30.91], [0.984], [0.971], [0.866],
  ),
  caption: [Raster-vector-raster fidelity on generated raster images.],
) <tab:z-image-raster-vectorization-fidelity>

#figure(
  table(
    columns: (1.4fr, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr),
    align: (left, center, center, center, center, center, center),
    inset: 4pt,
    stroke: (x, y) => (
      left: if x == 0 { none } else { 0.4pt },
      top: if y == 0 { none } else { 0.4pt },
    ),
    table.header(
      [Method],
      [Valid SVG rate ↑],
      [Boundary F1 at 1 px ↑],
      [Boundary F1 at 4 px ↑],
      [Chamfer px ↓],
      [Hausdorff px ↓],
      [Render time ms ↓],
    ),
    [Proposed], [100.0%], [0.312], [0.444], [15.62], [102.79], [27.70],
    [OmniSVG1.1 4B], [99.2%], [0.412], [0.640], [22.15], [167.65], [23.96],
    [OmniSVG1.1 8B], [98.6%], [0.398], [0.603], [24.05], [172.41], [24.82],
    [StarVector 1B], [49.0%], [0.336], [0.530], [13.09], [88.63], [30.88],
    [StarVector 8B], [50.5%], [0.361], [0.506], [18.50], [120.81], [25.69],
    [`vtracer`], [100.0%], [0.696], [0.948], [1.93], [27.91], [23.10],
  ),
  caption: [Validity, contour alignment, and rendering cost on generated raster images.],
) <tab:z-image-raster-vectorization-boundary>

#figure(
  table(
    columns: (1.4fr, 1fr, 1fr, 1fr, 1fr),
    align: (left, center, center, center, center),
    inset: 4pt,
    stroke: (x, y) => (
      left: if x == 0 { none } else { 0.4pt },
      top: if y == 0 { none } else { 0.4pt },
    ),
    table.header([Method], [SVG bytes ↓], [Elements ↓], [Paths ↓], [Path commands ↓]),
    [Proposed], [8005.76], [4.67], [3.67], [79.53],
    [OmniSVG1.1 4B], [6696.20], [6.46], [5.46], [244.43],
    [OmniSVG1.1 8B], [6803.54], [10.89], [9.89], [250.28],
    [StarVector 1B], [2102.44], [11.49], [2.87], [40.14],
    [StarVector 8B], [1270.68], [7.10], [2.31], [44.36],
    [`vtracer`], [60192.09], [95.84], [94.84], [1779.18],
  ),
  caption: [SVG complexity on generated raster images.],
) <tab:z-image-raster-vectorization-complexity>

These results show two different failure modes of the final vectorization
stage. Classical tracing is robust on the generated rasters in the narrow sense
of validity and raster reconstruction: all 1010 images produced renderable SVG
files, the SSIM is high, and the foreground mask overlap remains close to the
input. At the same time, the resulting SVGs are large, with more than 1600 path
commands on average. The OmniSVG evaluations produce much more compact SVG
files: OmniSVG1.1 4B averages 6696 bytes and 244 path commands, while OmniSVG1.1 8B
averages 6804 bytes and 250 path commands. StarVector is more compact still:
the 1B variant averages 2102 bytes and 40 path commands, while the 8B variant
averages 1271 bytes and 44 path commands. The StarVector models obtain better
pixel-level fidelity than OmniSVG on the subset that renders successfully, but
only about half of their outputs are valid SVGs in this generated-raster
setting. The neural vectorizers therefore expose a trade-off between
compactness, fidelity on successful outputs, and reliability. On raster images
produced by the Z-Image stage, direct tracing remains substantially more
reliable at preserving the visible raster content.

#heading(level: 1, numbering: none)[Conclusion]

This thesis addressed text-conditioned generation of scalable vector graphics
through a two-stage pipeline. The first stage adapts a pretrained
text-to-image model to produce raster images in a style that is more suitable
for vectorization. The second stage studies raster-to-vector conversion as a
separate supervised problem, using a compact representation based on cubic
Bezier segments and a conditional flow-matching model. The central objective
was therefore not only to generate visually plausible images from text, but to
connect semantic raster generation with a structured vector representation that
can be rendered as SVG.

== Assessment of the achieved results

The work shows that this decomposition is a practical way to organize the
problem. It makes it possible to use the semantic knowledge of large
text-to-image models without requiring the vectorizer itself to learn language
understanding. At the same time, it gives the second stage access to training
data with known geometric structure, including procedurally generated Bezier
examples. The experiments also show why the decomposition is useful: a direct
adaptation of raster-generation weights to Bezier prediction did not provide a
better initialization in the comparison, whereas a dedicated
raster-to-vector model can be trained and evaluated with explicit geometric
targets.

== Author's contribution

The author's main contribution is the design and implementation of this
pipeline for SVG generation. This includes the SVG parsing and normalization
procedure, the fixed-size Bezier representation with geometric and appearance
attributes, the synthetic data generator, the conditional flow-matching
vectorizer, and the evaluation scripts used to compare raster fidelity,
contour alignment, SVG validity, and structural complexity. The evaluation
also contributes a clearer view of the trade-offs among vectorization methods.
Classical tracing remains a strong baseline for raster reconstruction and
validity, especially on images generated by the raster stage, but it often
produces large SVG files with many path commands. Neural SVG-generation
systems can be more compact, but their outputs are less reliable under the
tested raster-to-vector setting. This confirms that vector graphics generation
should not be assessed only by rendered pixel similarity: compactness,
validity, editability, and failure rate are equally important for practical
SVG output.

== Broader perspective

The achieved results should also be interpreted in a broader perspective. The
pipeline demonstrates that generative image models and structured vector
models can be combined without forcing all aspects of the task into a single
autoregressive SVG generator. This is useful when paired text-SVG data are
limited, because the language-conditioned part and the geometry-conditioned
part can be trained with different sources of supervision. However, the same
decomposition also introduces an interface problem. Errors introduced by the
raster stage, such as weak boundaries, antialiasing artifacts, or unnecessary
texture, become inputs to the vectorizer and may be converted into unwanted
geometry. The final quality of the generated SVG is therefore constrained by
both stages and by how well their data distributions match.

== Limitations

Several limitations remain. The current vector representation expresses all
geometry as cubic Bezier segments, so higher-level SVG primitives such as
circles, rectangles, and symbolic shape elements are not preserved as separate
objects. Although these primitives can be approximated accurately by Bezier
curves, the resulting SVG is less semantically editable than an SVG that keeps
the original primitive types. The data pipeline also assumes solid fills with
opacity and does not parse gradients, masks, filters, or complex style rules.
This excludes a significant part of the SVG design space. In addition, the
fixed maximum number of segments imposes a hard capacity limit: graphics
requiring more segments must either be simplified or truncated.

== Future work

Future work should focus on improving both the expressiveness of the
representation and the reliability of the full pipeline. One direction is
primitive recovery: closed contours could be tested for compatibility with
circles, ellipses, rectangles, rounded rectangles, polygons, or simple compound
shapes, and then replaced by the corresponding SVG elements when the
approximation error is sufficiently small. Another direction is to extend the
data conversion and renderer to gradients, transparency, masks, and richer
style attributes. The current pipeline assumes a white background in both the
text-to-raster stage and the vectorizer training setup. Producing and training
on transparent raster inputs would better match the common use of SVG graphics
as foreground assets; methods for transparent image generation, such as
latent-transparency diffusion @zhang2024latenttransparency, suggest one
possible way to adapt the first stage. Finally, larger paired vector datasets
and stronger conditioning could make it possible to revisit direct
text-to-vector generation, but the results of this thesis indicate that the
two-stage formulation remains a useful and data-efficient baseline for further
research. The public implementation and trained model artifacts are listed in
@app:implementation-artifacts.

#thesis_bibliography(read("references.bib", encoding: none))

#appendix(label: <app:implementation-artifacts>)[Implementation artifacts][
  The source code for the implementation developed in this thesis is available
  in the GitHub repository:
  #link("https://github.com/JosefKuchar/svg-generator")[
    https://github.com/JosefKuchar/svg-generator
  ].

  Trained model checkpoints and accompanying model artifacts are available on
  Hugging Face:
  #link("https://huggingface.co/JosefKuchar/svg-generator")[
    https://huggingface.co/JosefKuchar/svg-generator
  ].

  Both the implementation and the trained model artifacts are released under
  the Apache License 2.0.
]
