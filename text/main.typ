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
    "raster-to-vector conversion",
    "Bezier curves",
    "flow matching",
    "text-to-image generation",
    "LoRA fine-tuning",
  ),
)

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

In this thesis, _vectorization_ refers to raster-to-vector conversion: the
problem of converting a pixel image into a visually similar vector graphic
described by geometric primitives. Raster images represent visual content as a
discrete grid of colored samples, whereas vector graphics describe shapes by
parameters such as paths, curves, fills, and strokes. Rendering maps a vector
description to pixels; vectorization attempts the inverse direction by
recovering a compact and editable geometric description from raster evidence
@selinger2003potrace. This inverse problem is inherently ambiguous, because
many different sets of curves and shapes can render to nearly identical pixel
images. A useful vectorizer therefore must balance image fidelity with
structural simplicity, semantic editability, and validity of the resulting SVG
@dziuba2023imagevectorization.

The Bezier curves used in this work are a standard way of representing curved
SVG paths. In SVG path data, a cubic Bezier segment is specified by an endpoint
and two control points relative to the current point; sequences of such
segments can describe smooth contours, while fills and strokes determine how
the paths are rendered @w3c2011svgpaths. This makes cubic Bezier curves a
natural low-level representation for learning, because they are expressive
enough to approximate many shapes while still being described by a small fixed
number of continuous parameters per segment.

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
space rather than in a restricted curve-only representation. This allows the
model to use higher-level primitives such as circles, ellipses, polygons, text,
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

// TODO: Add representative classical and neural vectorization methods.

Existing vectorizers also serve as empirical baselines. Classical systems are
strong engineering tools, but they typically optimize local image fidelity and
often produce dense, fragmented paths when the input contains noise,
compression artifacts, blur, or soft color transitions. Recent neural
text-to-SVG systems, by contrast, often rely on large vision-language models
fine-tuned on SVG data. The evaluation in this work therefore distinguishes
between performance on the SVG validation distribution and behavior on
synthetic images whose ground-truth Bezier structure is known. This makes it
possible to compare ordinary reconstruction fidelity with robustness to inputs
that differ from the web-SVG distribution used by large neural baselines.

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

== Source SVG dataset

Both stages use the `mikronai/svg-svgrepo` dataset distributed through Hugging
Face @mikronaiSvgSvgrepo. The dataset is derived from SVG Repo graphics and is
provided as a tabular Parquet dataset. At the time of use, the default subset
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

LoRA is a parameter-efficient fine-tuning method. Instead of updating all
weights of a large pretrained model, it freezes the base weights and learns
small trainable low-rank matrices whose product approximates the desired weight
update @hu2022lowrank. This is useful in the first stage because the model
should retain the broad semantic and compositional knowledge of the pretrained
text-to-image system while adapting only a comparatively small number of
parameters to the SVG-like raster domain.

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

Classifier-free guidance is a conditioning technique for diffusion models in
which the model is trained with both conditional and unconditional inputs. At
sampling time, the conditional prediction is strengthened by comparing it with
the unconditional prediction, and the guidance scale controls how strongly the
sample is pushed toward the prompt @ho2021classifierfree. This usually improves
prompt adherence but requires additional model evaluations unless the effect
has been distilled into a faster model.

After training, the learned LoRA weights are loaded into the `Z-Image-Turbo`
pipeline for fast sampling. This design preserves the knowledge of the original
pretrained model while making inference substantially more efficient than full
base-model fine-tuning. The use of the same SVG-style LoRA on the distilled
turbo variant is motivated by the observation that distilled diffusion models
can preserve the controllability of their teacher models, allowing controls
learned for the base model to remain useful after distillation
@gandikota2025distilling. In the experiments reported below, prompts are
prefixed with `SVG illustration with white background. ` to bias the generator
toward clean foreground graphics on a simple canvas. The resulting samples are
then assessed both as images and as inputs for downstream vectorization.

A preliminary comparison of several Stage 1 variants is shown qualitatively in
@tab:stage1-raster-examples and quantitatively in @tab:stage1-benchmark. The
compared variants include the base `z-image` model, prompt-prefixing strategies,
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

    [OmniSVG 4B],
    image("assets/raster/omnisvg_4b/0001.png", width: 100%),
    image("assets/raster/omnisvg_4b/0002.png", width: 100%),
    image("assets/raster/omnisvg_4b/0003.png", width: 100%),
    image("assets/raster/omnisvg_4b/0004.png", width: 100%),

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
    [OmniSVG 4B], [0.828205], [0.391314], [57.620655],
    [Turbo], [0.826786], [0.509892], [227.691742],
    [Turbo prefixed], [0.871237], [0.583856], [142.711678],
    [Turbo prefixed + LoRA], [0.879104], [0.600208], [143.174617],
  ),
  caption: [Preliminary Stage 1 benchmark of text-to-raster model variants.],
) <tab:stage1-benchmark>

#figure(
  table(
    columns: (1.4fr, 1fr, 1fr, 1fr, 1fr),
    align: (left, center, center, center, center),
    inset: 4pt,
    stroke: (x, y) => if x == 0 or y == 0 { 0.8pt } else { 0.4pt },
    table.header(
      [Variant],
      [Sample 1],
      [Sample 2],
      [Sample 3],
      [Sample 4],
    ),
    [Reference],
    [#image("assets/raster/reference/0001.png", width: 100%)],
    [#image("assets/raster/reference/0002.png", width: 100%)],
    [#image("assets/raster/reference/0003.png", width: 100%)],
    [#image("assets/raster/reference/0004.png", width: 100%)],
    [Base],
    [#image("assets/raster/base/0001.png", width: 100%)],
    [#image("assets/raster/base/0002.png", width: 100%)],
    [#image("assets/raster/base/0003.png", width: 100%)],
    [#image("assets/raster/base/0004.png", width: 100%)],
    [Base prefixed],
    [#image("assets/raster/base_prefixed/0001.png", width: 100%)],
    [#image("assets/raster/base_prefixed/0002.png", width: 100%)],
    [#image("assets/raster/base_prefixed/0003.png", width: 100%)],
    [#image("assets/raster/base_prefixed/0004.png", width: 100%)],
    [Turbo],
    [#image("assets/raster/turbo/0001.png", width: 100%)],
    [#image("assets/raster/turbo/0002.png", width: 100%)],
    [#image("assets/raster/turbo/0003.png", width: 100%)],
    [#image("assets/raster/turbo/0004.png", width: 100%)],
    [Turbo prefixed],
    [#image("assets/raster/turbo_prefixed/0001.png", width: 100%)],
    [#image("assets/raster/turbo_prefixed/0002.png", width: 100%)],
    [#image("assets/raster/turbo_prefixed/0003.png", width: 100%)],
    [#image("assets/raster/turbo_prefixed/0004.png", width: 100%)],
  ),
  caption: [Qualitative Stage 1 comparison of generated raster outputs.],
)

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

#figure(
  table(
    columns: (1fr, 1.8fr, 1fr, 1fr, 1fr),
    align: (left, left, center, center, center),
    inset: 6pt,
    stroke: (x, y) => (
      left: none,
      top: if y == 0 or calc.rem(y - 1, 3) == 0 { 0.4pt } else { none },
    ),
    table.header(
      table.cell(rowspan: 2)[Time steps],
      table.cell(rowspan: 2)[Metric],
      table.cell(colspan: 3)[Rank],
      [4],
      [16],
      [64],
    ),
    table.cell(rowspan: 3)[500], text(size: 8pt)[CLIP similarity ↑], text(size: 8pt)[0.880], text(size: 8pt)[0.872], text(size: 8pt)[0.885],
    text(size: 8pt)[DINO similarity ↑], text(size: 8pt)[0.607], text(size: 8pt)[0.599], text(size: 8pt)[0.599],
    text(size: 8pt)[Vectorization MSE ↓], text(size: 8pt)[299.456], text(size: 8pt)[187.121], text(size: 8pt)[205.943],

    table.cell(rowspan: 3)[1000], text(size: 8pt)[CLIP similarity ↑], text(size: 8pt)[0.886], text(size: 8pt)[0.887], text(size: 8pt)[0.885],
    text(size: 8pt)[DINO similarity ↑], text(size: 8pt)[0.616], text(size: 8pt)[0.621], text(size: 8pt)[0.620],
    text(size: 8pt)[Vectorization MSE ↓], text(size: 8pt)[328.455], text(size: 8pt)[128.576], text(size: 8pt)[199.657],

    table.cell(rowspan: 3)[1500], text(size: 8pt)[CLIP similarity ↑], text(size: 8pt)[0.886], text(size: 8pt)[0.887], text(size: 8pt)[0.884],
    text(size: 8pt)[DINO similarity ↑], text(size: 8pt)[0.622], text(size: 8pt)[0.618], text(size: 8pt)[0.615],
    text(size: 8pt)[Vectorization MSE ↓], text(size: 8pt)[206.473], text(size: 8pt)[143.796], text(size: 8pt)[346.287],

    table.cell(rowspan: 3)[2000], text(size: 8pt)[CLIP similarity ↑], text(size: 8pt)[0.885], text(size: 8pt)[0.885], text(size: 8pt)[0.887],
    text(size: 8pt)[DINO similarity ↑], text(size: 8pt)[0.627], text(size: 8pt)[0.614], text(size: 8pt)[0.625],
    text(size: 8pt)[Vectorization MSE ↓], text(size: 8pt)[142.765], text(size: 8pt)[97.675], text(size: 8pt)[168.836],

    table.cell(rowspan: 3)[2500], text(size: 8pt)[CLIP similarity ↑], text(size: 8pt)[0.888], text(size: 8pt)[0.888], text(size: 8pt)[0.882],
    text(size: 8pt)[DINO similarity ↑], text(size: 8pt)[0.626], text(size: 8pt)[0.627], text(size: 8pt)[0.620],
    text(size: 8pt)[Vectorization MSE ↓], text(size: 8pt)[174.613], text(size: 8pt, weight: "bold")[92.145], text(size: 8pt)[273.638],

    table.cell(rowspan: 3)[3000], text(size: 8pt)[CLIP similarity ↑], text(size: 8pt)[0.887], text(size: 8pt)[0.887], text(size: 8pt, weight: "bold")[0.888],
    text(size: 8pt)[DINO similarity ↑], text(size: 8pt)[0.626], text(size: 8pt, weight: "bold")[0.635], text(size: 8pt)[0.632],
    text(size: 8pt)[Vectorization MSE ↓], text(size: 8pt)[173.293], text(size: 8pt)[245.512], text(size: 8pt)[132.058],

    table.cell(rowspan: 3)[3500], text(size: 8pt)[CLIP similarity ↑], text(size: 8pt)[0.886], text(size: 8pt)[0.887], text(size: 8pt)[0.888],
    text(size: 8pt)[DINO similarity ↑], text(size: 8pt)[0.626], text(size: 8pt)[0.630], text(size: 8pt)[0.633],
    text(size: 8pt)[Vectorization MSE ↓], text(size: 8pt)[166.197], text(size: 8pt)[93.775], text(size: 8pt)[157.162],

    table.cell(rowspan: 3)[5000], text(size: 8pt)[CLIP similarity ↑], text(size: 8pt)[0.886], text(size: 8pt)[0.886], text(size: 8pt)[0.883],
    text(size: 8pt)[DINO similarity ↑], text(size: 8pt)[0.620], text(size: 8pt)[0.631], text(size: 8pt)[0.617],
    text(size: 8pt)[Vectorization MSE ↓], text(size: 8pt)[178.814], text(size: 8pt)[166.223], text(size: 8pt)[172.532],
  ),
  caption: [Stage 1 LoRA ablation across training duration, LoRA rank, and evaluation metric. Higher is better for CLIP and DINO similarity; lower is better for vectorization MSE.],
) <tab:lora-ablation>

The results suggest that prompt prefixing has a substantial effect, especially
for the turbo model. The best overall semantic similarity is obtained by the
`Turbo prefixed + LoRA` configuration, while the lowest
vectorization error is achieved by `Turbo prefixed`. This indicates that the
adapted LoRA model improves perceptual alignment with the references, but its
advantage with respect to downstream vectorization should be verified on a
larger evaluation.

Based on the rank and checkpoint ablation, the LoRA model with rank 16 at
3000 training steps was selected for subsequent Stage 1 experiments. This
checkpoint achieves the highest DINO similarity in the ablation and provides a
reasonable compromise between semantic alignment and traceability, even though
the lowest vectorization MSE is observed for the rank-16 checkpoint at 2500
steps.

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
case. This distinction is important because automatic vectorization is
underdetermined from pixels alone: when the vector scene is generated
procedurally, the exact geometric target is known by construction, whereas for
ordinary raster images there may be many plausible vector explanations
@selinger2003potrace @dziuba2023imagevectorization.

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
  caption: [Training loss during the synthetic pretraining phase of the raster-to-vector model. The vertical axis uses a logarithmic scale, and the curve is smoothed for readability.],
) <fig:vectorizer-pretraining-loss>

#figure(
  image("assets/wandb/classic-serenity-74_image_mse.pdf", width: 90%),
  caption: [Train and validation image reconstruction MSE during the synthetic pretraining phase of the raster-to-vector model. The metric is computed after rasterizing the predicted vector representation. The vertical axis uses a logarithmic scale, and the curves are smoothed for readability.],
) <fig:vectorizer-pretraining-mse>

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
  caption: [Train and validation image reconstruction MSE for the flow-matching vectorizer and an autoregressive vectorizer with comparable capacity and the same image encoder. The graph is limited to the first 250k training steps. The vertical axis uses a logarithmic scale, and the curves are smoothed for readability.],
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
  caption: [Train and validation image reconstruction MSE for the flow-matching vectorizer with and without the image encoder. The graph is limited to the first 150k training steps. The vertical axis uses a logarithmic scale, and the curves are smoothed for readability.],
) <fig:image-encoder-ablation-mse>

#figure(
  image("assets/wandb/image-encoder-ablation_train_loss.pdf", width: 90%),
  caption: [Training loss for the flow-matching vectorizer with and without the image encoder. The graph is limited to the first 150k training steps. The vertical axis uses a logarithmic scale, and the curves are smoothed for readability.],
) <fig:image-encoder-ablation-loss>

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
    align: (left, center, center, center, center, center, center),
    inset: 3pt,
    stroke: (x, y) => (
      left: if x == 0 { none } else { 0.4pt },
      top: if y == 0 { none } else { 0.4pt },
    ),
    table.header([Split and image], [Example 1], [Example 2], [Example 3], [Example 4], [Example 5], [Example 6]),
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
  caption: [Qualitative samples from the final synthetic pretraining checkpoint. Each generated image is rendered from the predicted Bezier representation and paired with the corresponding raster reference.],
) <tab:vectorizer-pretraining-samples>

The qualitative comparison is complemented by quantitative evaluation of the
same checkpoint. The metrics combine image-space reconstruction scores with
vector-structure diagnostics, as outlined in
@tab:vectorizer-pretraining-quantitative-todo.

#figure(
  table(
    columns: (2.2fr, 1fr, 1fr, 2.2fr),
    align: (left, center, center, left),
    inset: 6pt,
    stroke: (x, y) => (
      left: if x == 0 { none } else { 0.4pt },
      top: if y == 0 { none } else { 0.4pt },
    ),
    table.header([Metric], [Synthetic train], [SVG validation], [Purpose]),
    [Rendered image MSE ↓], [TODO], [TODO], [Pixel-level reconstruction error after rendering the predicted Bezier representation.],
    [Rendered image SSIM ↑], [TODO], [TODO], [Perceptual structural similarity between the reference raster and rendered prediction.],
    [DINO similarity ↑], [TODO], [TODO], [Feature-space similarity that is less sensitive to small rasterization differences.],
    [Valid SVG rate ↑], [TODO], [TODO], [Fraction of samples that can be decoded and rendered without geometry or parsing failures.],
    [Segment precision / recall ↑], [TODO], [TODO], [Agreement between predicted and reference Bezier structure when a segment-level matching procedure is available.],
  ),
  caption: [Quantitative evaluation template for the final synthetic pretraining checkpoint. The TODO cells mark measurements that are still to be inserted after running the evaluation script on the saved samples or full evaluation split.],
) <tab:vectorizer-pretraining-quantitative-todo>

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

This chapter evaluates the two stages of the proposed pipeline and the design
choices that connect them. The experiments are organized around three
questions: whether the text-to-raster model can be adapted to a
vectorization-friendly image domain, whether synthetic pretraining improves the
raster-to-vector model, and how the proposed vectorizer compares with existing
classical and neural vectorization systems.

== Alternatives to the proposed decomposition

The first alternative is to adapt a pretrained text-to-raster model directly
into a text-to-Bezier model. This would be conceptually attractive, because it
would collapse the whole pipeline into one model while preserving the semantic
knowledge of the pretrained generator. A preliminary experiment with this
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
  caption: [Training loss for direct adaptation of a pretrained text-to-raster model to Bezier prediction compared with training the same architecture after resetting the weights. The vertical axis uses a logarithmic scale, and the curves are smoothed for readability.],
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

== Stage 1 fine-tuning

The Stage 1 experiments evaluate fine-tuning of the text-to-raster model on
the SVG Repo dataset. The SVG files are rasterized and used as the visual
target distribution for LoRA adaptation. The goal is not merely to improve
generic image quality, but to make generated images more suitable for
downstream vectorization: flatter color regions, sharper silhouettes, fewer
unnecessary textures, and simpler topology.

The main comparison is between the base text-to-raster model and the
fine-tuned variant. Qualitative examples show the visual change before and
after fine-tuning. Quantitatively, the evaluation combines text-image alignment
metrics with traceability metrics.
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

Additional ablations compare the flow-matching formulation with an
autoregressive variant of comparable size, and measure the effect of image
conditioning. The conditioning ablations compare the full model with a model
trained without an image encoder and study encoder scaling by changing the
image encoder while keeping the vectorizer backbone as constant as possible.
Together, these experiments clarify how much of the performance is due to the
flow-matching objective, the transformer backbone, and the pretrained visual
representation.

== Flow-matching inference ablation

Inference requires numerical integration of the learned velocity field. The
number of integration steps directly affects runtime and reconstruction
quality. The inference ablation therefore evaluates several fixed step counts
and measures the resulting output quality, topology, and rendering error.
This experiment is important because an excessively small number of steps may
produce unstable or incomplete geometry, while too many steps increase runtime
without necessarily improving the final SVG.

== Pipeline evaluation

The final system combines the fine-tuned text-to-raster model with the
raster-to-vector model. Evaluation of the full pipeline includes both
end-to-end qualitative examples and quantitative comparisons with baselines.
The baselines include classical vectorization tools, recent neural
SVG-generation systems, and direct raster outputs from the Stage 1 model. The
comparison emphasizes not only pixel-level reconstruction, but also properties
important for editable vector graphics, such as path count, node count,
topological cleanliness, robustness to noisy inputs, and ease of manual
editing.

The vectorization comparison is performed with the `evaluate_vectorization.py`
script. The script renders each reference SVG and each generated SVG at a fixed
resolution of 1024 pixels, compares the rendered RGB images, and records both
image-space and structure-related statistics. The
comparison contains four methods: the proposed flow-matching vectorizer,
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
    columns: (1.25fr, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr),
    align: (left, center, center, center, center, center, center),
    inset: 3pt,
    stroke: (x, y) => (
      left: if x == 0 { none } else { 0.4pt },
      top: if y == 0 { none } else { 0.4pt },
    ),
    table.header([Method], [0000], [0001], [0002], [0003], [0004], [0005]),
    [Reference],
    vectorization-sample("assets/vectorization_qualitative/validation/reference/0000.png"),
    vectorization-sample("assets/vectorization_qualitative/validation/reference/0001.png"),
    vectorization-sample("assets/vectorization_qualitative/validation/reference/0002.png"),
    vectorization-sample("assets/vectorization_qualitative/validation/reference/0003.png"),
    vectorization-sample("assets/vectorization_qualitative/validation/reference/0004.png"),
    vectorization-sample("assets/vectorization_qualitative/validation/reference/0005.png"),
    [OmniSVG 4B],
    vectorization-sample("assets/vectorization_qualitative/validation/omnisvg_4b/0000.png"),
    vectorization-sample("assets/vectorization_qualitative/validation/omnisvg_4b/0001.png"),
    vectorization-sample("assets/vectorization_qualitative/validation/omnisvg_4b/0002.png"),
    vectorization-sample("assets/vectorization_qualitative/validation/omnisvg_4b/0003.png"),
    vectorization-sample("assets/vectorization_qualitative/validation/omnisvg_4b/0004.png"),
    vectorization-sample("assets/vectorization_qualitative/validation/omnisvg_4b/0005.png"),
    [OmniSVG 8B],
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
  caption: [Qualitative comparison on SVG validation samples. Each generated SVG is rendered with the same rasterizer used for quantitative evaluation; missing or non-renderable SVG files are shown as white images. These examples test behavior on the target validation distribution, but not necessarily out-of-distribution generalization.],
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
    columns: (1.25fr, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr),
    align: (left, center, center, center, center, center, center),
    inset: 3pt,
    stroke: (x, y) => (
      left: if x == 0 { none } else { 0.4pt },
      top: if y == 0 { none } else { 0.4pt },
    ),
    table.header([Method], [0000], [0001], [0002], [0003], [0004], [0005]),
    [Reference],
    vectorization-sample("assets/vectorization_qualitative/synthetic/reference/0000.png"),
    vectorization-sample("assets/vectorization_qualitative/synthetic/reference/0001.png"),
    vectorization-sample("assets/vectorization_qualitative/synthetic/reference/0002.png"),
    vectorization-sample("assets/vectorization_qualitative/synthetic/reference/0003.png"),
    vectorization-sample("assets/vectorization_qualitative/synthetic/reference/0004.png"),
    vectorization-sample("assets/vectorization_qualitative/synthetic/reference/0005.png"),
    [OmniSVG 4B],
    vectorization-sample("assets/vectorization_qualitative/synthetic/omnisvg_4b/0000.png"),
    vectorization-sample("assets/vectorization_qualitative/synthetic/omnisvg_4b/0001.png"),
    vectorization-sample("assets/vectorization_qualitative/synthetic/omnisvg_4b/0002.png"),
    vectorization-sample("assets/vectorization_qualitative/synthetic/omnisvg_4b/0003.png"),
    vectorization-sample("assets/vectorization_qualitative/synthetic/omnisvg_4b/0004.png"),
    vectorization-sample("assets/vectorization_qualitative/synthetic/omnisvg_4b/0005.png"),
    [OmniSVG 8B],
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
  caption: [Qualitative comparison on synthetic-generator samples. These examples test general vectorization behavior on controlled geometric inputs, complementing the in-distribution SVG validation comparison. Missing or non-renderable SVG files are shown as white images.],
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
      [Method],
      [MSE (0--255) ↓],
      [SSIM ↑],
      [Mask IoU ↑],
      [Boundary F1 at 2 px ↑],
      [Chamfer px ↓],
      [Hausdorff px ↓],
    ),
    [Proposed model], [TODO], [TODO], [TODO], [TODO], [TODO], [TODO],
    [OmniSVG 4B], [7696.39], [0.621], [0.631], [0.538], [17.84], [145.46],
    [OmniSVG 8B], [8425.56], [0.589], [0.608], [0.516], [18.89], [149.69],
    [StarVector 1B], [5147.82], [0.652], [0.631], [0.483], [24.26], [143.28],
    [StarVector 8B], [8449.48], [0.461], [0.444], [0.441], [38.75], [229.73],
    [`vtracer`], [92.01], [0.994], [0.984], [0.886], [1.26], [17.20],
  ),
  caption: [Vectorization-fidelity comparison on SVG validation samples. All metrics are computed after rendering the generated SVG and the reference SVG at 1024 px resolution with `evaluate_vectorization.py`. Lower MSE, Chamfer distance, and Hausdorff distance are better; higher SSIM, mask IoU, and boundary F1 are better. The OmniSVG 4B, OmniSVG 8B, StarVector 1B, and StarVector 8B rows are each computed over 1010 pairs.],
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
      [Method],
      [MSE (0--255) ↓],
      [SSIM ↑],
      [Mask IoU ↑],
      [Boundary F1 at 2 px ↑],
      [Chamfer px ↓],
      [Hausdorff px ↓],
    ),
    [Proposed model], [TODO], [TODO], [TODO], [TODO], [TODO], [TODO],
    [OmniSVG 4B], [9591.49], [0.330], [0.432], [0.461], [32.87], [242.00],
    [OmniSVG 8B], [11024.09], [0.283], [0.407], [0.436], [36.07], [253.81],
    [StarVector 1B], [6339.14], [0.314], [0.297], [0.476], [47.02], [317.25],
    [StarVector 8B], [7536.35], [0.137], [0.104], [0.496], [59.58], [401.38],
    [`vtracer`], [23.95], [0.997], [0.993], [0.920], [1.03], [20.45],
  ),
  caption: [Vectorization-fidelity comparison on synthetic-generator samples. This table uses the same metrics as @tab:vectorization-fidelity-validation, but evaluates controlled procedural inputs to test general vectorization behavior outside the SVG validation distribution. MSE is converted from the normalized `evaluate_vectorization.py` output to the 0--255 RGB scale for consistency with the Stage 1 vectorization MSE. The OmniSVG 4B, OmniSVG 8B, StarVector 1B, and StarVector 8B rows are each computed over 1000 pairs.],
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
    table.header(
      [Method],
      [Valid SVG rate ↑],
      [SVG bytes ↓],
      [Elements ↓],
      [Paths ↓],
      [Path commands ↓],
    ),
    [Proposed model], [TODO], [TODO], [TODO], [TODO], [TODO],
    [OmniSVG 4B], [99.4%], [5284.03], [5.04], [4.04], [219.53],
    [OmniSVG 8B], [99.3%], [5296.17], [8.62], [7.62], [206.38],
    [StarVector 1B], [79.0%], [1957.71], [9.17], [4.09], [118.60],
    [StarVector 8B], [65.1%], [2220.83], [10.63], [5.36], [213.25],
    [`vtracer`], [100.0%], [14370.11], [10.68], [9.68], [364.55],
  ),
  caption: [SVG validity and complexity comparison on SVG validation samples. The valid SVG rate is derived from render failures reported by `evaluate_vectorization.py`; the remaining columns report mean generated-SVG statistics over successfully produced files. Lower complexity values are preferable only when visual fidelity remains comparable. The valid SVG rates are computed from 6 render errors for OmniSVG 4B, 7 for OmniSVG 8B, 212 for StarVector 1B, and 352 for StarVector 8B, each among 1010 pairs.],
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
    table.header(
      [Method],
      [Valid SVG rate ↑],
      [SVG bytes ↓],
      [Elements ↓],
      [Paths ↓],
      [Path commands ↓],
    ),
    [Proposed model], [TODO], [TODO], [TODO], [TODO], [TODO],
    [OmniSVG 4B], [99.2%], [9165.78], [13.72], [12.72], [393.75],
    [OmniSVG 8B], [98.6%], [9658.08], [29.95], [28.95], [400.92],
    [StarVector 1B], [43.2%], [4108.50], [30.42], [9.97], [447.18],
    [StarVector 8B], [20.5%], [5354.53], [40.59], [14.22], [962.31],
    [`vtracer`], [100.0%], [24468.73], [14.66], [13.66], [625.33],
  ),
  caption: [SVG validity and complexity comparison on synthetic-generator samples. This table reports the same SVG validity and structure statistics as @tab:vectorization-complexity-validation, but on controlled procedural inputs. The OmniSVG 4B valid SVG rate is computed from 8 render errors among 1000 pairs, the OmniSVG 8B valid SVG rate from 14 render errors among 1000 pairs, the StarVector 1B valid SVG rate from 568 render errors among 1000 pairs, and the StarVector 8B valid SVG rate from 795 render errors among 1000 pairs.],
) <tab:vectorization-complexity-synthetic>

The fidelity tables capture visual reconstruction quality, while the
complexity tables capture whether the output is a practical vector graphic.
This separation is important because a method can obtain a low raster error by
creating a very large SVG with many paths or path commands. Conversely, a more
compact SVG may be preferable for editing even when it introduces a small
raster-space error. The interpretation therefore treats the validation results
separately from the synthetic-generator results and reads the corresponding
fidelity and complexity tables together rather than selecting a method from a
single scalar score.

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
