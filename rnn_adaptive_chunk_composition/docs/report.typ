#let overlay(img, color) = layout(bounds => {
  let size = measure(img, ..bounds)
  img
  place(top + left, block(..size, fill: color))
})

#let oist_header = image("assets/oist_logo_full_header.svg", fit: "contain", width: 100%)
#set page(
  paper: "a4",
  columns: 1,
  margin: (top: 2.25cm, bottom: 1.5cm, left: 1.5cm, right: 1.5cm),
  header: context {
    if counter(page).get().at(0) == 1 [
      #align(center)[
        #overlay(oist_header, white.transparentize(40%))
      ]
    ]
  },
  footer: context[
    #v(1em)
    #h(1fr) #counter(page).display("1") #h(1fr)
  ],
  header-ascent: 0em,
  footer-descent: 0.5em,
)
#set par(justify: true)
#set math.mat(delim: "[")

#place(
  top + center,
  scope: "parent",
  float: true,
  [
    #text(1.4em, fill: rgb("#8b0000"), weight: "bold")[
      A313 Cognitive Neurorobotics: Project Report
    ]\
    #text(1.2em, weight: "bold")[
      Investigating Adaptive Chunk-based Composition in RNNs
    ]
  ],
)

*Student Name:* Pranshu Malik\
*Student ID:* 2401050\
*Date of Submission:* #datetime.today().display("[day] [month repr:long] [year]")\

== Abstract
We investigated whether gated-recurrent unit (GRU) based recurrent neural networks (RNNs) can learn compositional structures, such as shared subsequences or "chunks", shared across different training examples. Specifically, we trained these RNNs to reproduce a small set of words written in cursive handwriting with shared subsequences at various positions (progressive similarities or differences) and then analyzed their learnt outputs and hidden activities to identify any chunk-based composition strategies qualitatively and quantitatively. The code and trained models for this project are available on #text(font: "IBM Plex Mono", size: 9pt, fill: rgb("#0074d9"))[#link("https://github.com/pranshumalik14/cognitive-neurorobotics-labs/blob/main/rnn_adaptive_chunk_composition/proj.ipynb")[Github]] and the results are also compiled in these #text(font: "IBM Plex Mono", size: 9pt, fill: rgb("#0074d9"))[#link("https://docs.google.com/presentation/d/1AVO0R8dZ1R9ykwVeqA99lbB2LoenpcHXG6WBghcKy_s/edit?usp=sharing")[slides]].

= Introduction and Methodology
We trained GRU-based RNNs on four cursively handwritten words (i.e. human action sequences) of four letters each: "well", "hell", "help", and "weld" (@fig1). Each word contains shared subsequences (or "chunks") at different positions, such as "wel" or "hel" at the beginning, "el" in the middle, and "ell" at the end (@fig2). Our objective was to determine if these shared chunks are encoded within the network's hidden states during sequence reproduction. Importantly, each trial began from an identical initial context (of zeroes) and the same starting position (also zero), with the one-hot encoded target-specifying vector provided as input at every timestep.
#figure(
  grid(
    columns: 2,
    gutter: 0em,
    inset: 0em,
    align: horizon+left,
    image("assets/model.png", fit: "contain", width: 90%),
    image("assets/data.png", fit: "contain", width: 100%),
  ),
  caption: "Model architecture and training data"
) <fig1>
Each sequence comprised $T = 74$ steps, excluding the initial position. Training datapoints were uniformly sampled along SVG paths representing the handwritten words. Alongside the one-hot encoded target vector ($bold(s)_"tgt"$) representing executive intention, the network received the endpoint position ($bold(p)_(t-1)$) as input and produced a change in position ($bold(u)_t$) as output at each timestep, mimicking a sensorimotor task. This setup allowed easy “mixing” of targets as instructions to test for common elements and allowed silencing hidden units as well as principal components without any bias, due to the initial context being zero.
#figure(
  grid(
    columns: 2,
    gutter: 5em,
    inset: 0em,
    grid.cell(align: right, image("assets/adaptivechunks1.png", fit: "contain", width: 50%)),
    grid.cell(align: left, image("assets/adaptivechunks2.png", fit: "contain", width: 50%)),
  ),
  caption: "Some possible chunk-based compositions and progression of their similarities and differences"
) <fig2>
Training loss at each step was the squared distance between true sequence position and the network's output action applied to its position input, which was either the true previous global position (in teacher-forcing) or the cumulative sum of previous actions (in fine-tuning). Given the long prediction horizon, training was structured incrementally; we progressively increased the sequence length from 30 steps through 40, 50, 60, and finally to 74 steps. Concurrently, we adjusted the fine-tuning ratio from pure teacher-forcing (0) towards pure fine-tuning (1) in increments of 0.25. The training epochs were similarly scaled, beginning at 1000 epochs and increasing to 6000 epochs in the final stage. Note that for intermediate fine-tuning ratios (between 0 and 1), the choice of input (teacher-forcing or fine-tuning) at each timestep was probabilistically sampled from a Bernoulli distribution based on that ratio and this was consistently accounted for during backpropagation through time. We formalize this below.
$ 
bold(u)_t &:= mat(Delta p_x, Delta p_y) = "Network"(bold(x)_(t-1) | Omega_(t-1)), quad t >=1\
bold(x)_t &:= mat(bold(s)_"tgt", bold(p)_t)\
bold(p)_t &= cases("zero " &t=0, hat(bold(y))_t &"if fine-tune and" t >=1, bold(y)_t &"otherwise for" t >=1)\
hat(bold(y))_t &= cases(bold(u)_t + bold(p)_(t-1) " " &"in general", display(sum_(1<=tau<=t)bold(u)_tau) &"reduced form in pure fine-tuning")\
E_t &= ||bold(y)_t - (bold(u)_t + bold(p)_(t-1))||^2, quad t >=1
$
The accuracy of the network at any instant during training was estimated by calculating the mean loss (i.e. mean of deviation errors at each timestep) over the entire prediction length given a fixed set of parameters,
$ E_"mean" = frac(1, T)sum_(1<=t<=T)E_t (Omega). $

= Results and Discussion
We first see the effect of model size on training performance when the network receives the cumulative sum of its outputs as feedback while generating the target sequences (@fig3). In closed-loop generation, the model needed around a minimum of 16 hidden units to reach good task performance over 15000 training epochs. Qualitative similarities indicative of shared elements or patterns (i.e. chunks) are highlighted.
#figure(
  image("assets/closedloop_perf.png", fit: "contain", width: 100%),
  caption: "Feedback training performance"
) <fig3>
To understand the computation behind generation of these sequences, we examined the target-mixing effects and open-loop performance for each model to estimate the presence of any chunking (@fig4). The idea behind target mixing was to see if mild corruption (15%-85%) or full competition (50%-50%) between two target patterns during closed-loop generation still preserves any dominant or shared pattern, which could be indicative of chunking. The figure shows that initial common elements between target patterns are preserved despite competition, and that the higher-dimensional model is robust to corruption by a more dissimilar target. Moreover, open-loop performance (with feedback held constant) shows that the models express baseline bias present in target patterns (i.e. 'w' words being lower than 'h' words due to starting from the same position) and primarily function as feedback controllers.
#figure(
  grid(
    columns: (2fr, 1fr),
    gutter: 0em,
    inset: 0em,
    align: horizon+left,
    image("assets/closedloop_mixing.png", fit: "contain", width: 100%),
    image("assets/closedloop_comparison.png", fit: "contain", width: 100%),
  ),
  caption: "Target mixing and open-loop performance"
) <fig4>

== Training in both Feedback and Feedforward Settings
In order to push the networks to learn more systematic elements intrinsic to the target sequences, rather than only learning "quick and dirty" feedback control tricks, we trained the models over an equal mixture of closed-loop (feedback) and open-loop (feedforward) trials using the same loss function and doubled the total number of training epochs. This approach allowed us to more explicitly test whether the models can learn to reuse dynamic components within the recurrent network when generating similar sequences, which also provides a better opportunity to study the underlying computational mechanisms. Particularly, by reducing the network's reliance on immediate feedback, we aimed to make the models transition from predominantly being feedback controllers towards being more balanced feedforward-feedback controllers. We expected this shift to prioritize learning shared structural elements or "chunks", such as 'hel', 'wel', and 'ell', as being adaptively composable learned actions.
#figure(
  image("assets/openloop_perf.png", fit: "contain", width: 100%),
  caption: "Feedback and feedforward training performance"
) <fig5>
The same kind of qualitative similarities (as @fig3) suggestive of chunking are also apparent in feedback-feedforward training performance (@fig5). However, achieving good performance in this more challenging setting necessitated higher-dimensional models, with $gt.tilde 32$ hidden units, to effectively embed the feature complexity in the four-word dataset during both closed- and open-loop generation. Similar chunking patterns are also visible and robustly preserved after mixing targets for the high-dimensional model (@fig6). Finally, open-loop generation also indicated chunking, notably with 'weld' written similarly to 'well' but exhibiting drift reminiscent of a de-afferentiated patient writing without visual feedback @smyth1987 @teasdale1993.
#figure(
  grid(
    columns: (1.5fr, 1.92fr),
    gutter: 0em,
    inset: 0em,
    align: horizon+center,
    image("assets/openloop_mixing.png", fit: "contain", width: 75%),
    image("assets/openloop_comparison.png", fit: "contain", width: 75%),
  ),
  caption: "Target mixing and open-loop performance"
) <fig6>

== Analyzing Hidden Activity of the Network
To understand the neural mechanisms underlying sequence generation in these models, we analyzed the hidden unit activity of the trained networks (e.g. selected units in @fig7). Qualitatively, under feedback-only training, larger networks showed more systematic chunking-dependent features in raw hidden-unit activity, while smaller networks seemed to rely more on target- and state-dependent (feedback) strategies without any such apparent chunk-specific features. With feedback-feedforward training, some units in higher-dimensional networks exhibited temporal target-dependent similarities akin to adaptive chunking, whereas units in smaller networks displayed some chunk-dependent partitioning but with a weaker presence of temporal changes between those partitions, somewhat resembling the nature of patterns seen for larger networks trained in a feedback setting only.
#figure(
  grid(
    columns: 2,
    gutter: 0em,
    inset: 0em,
    align: horizon+center,
    image("assets/closedloop_hidden_activity.png", fit: "contain", width: 75%),
    image("assets/openloop_hidden_activity.png", fit: "contain", width: 75%),
  ),
  caption: "Hidden unit activities from feedback training (left) and feedback-feedforward training (right)"
) <fig7>
For a more objective analysis of hidden unit activity across time, target, and their interaction, we used demixed Principal Component Analysis (dPCA) as developed by Kobak et al. (#cite(<dpca>, form: "year")). Distinct temporal phases (early, middle, late) in sequence generation were revealed by dPCA for models trained under feedback only, a feature less evident as such in the feedback-feedforward trained high-dimensional models (Figures 8-10, first row in dPCA plots). Target-dependent dPCs (middle row in dPCA plots) showed more systematic (chunk-dependent) similarities for higher-dimensional models across both training regimes, supporting the view that smaller networks act more as simpler feedback controllers. Time-target interaction dPCs (last row in dPCA plots) show no clear adaptively chunked components for feedback-only trained models, but there are stronger hints for it in high-dimensional models under feedback-feedforward training. To understand the functional roles of these dPCs, we also examined the effects of silencing them. Smaller networks were completely disrupted by silencing even the least dominant target-dPC (@fig8). In contrast, silencing time-target-dPCs in higher-dimensional models had similar impacts on 'w' and 'h' words (@fig9), or affected the generation of "ell" in “well” and “hell” for both target and time-target dPCs (@fig10), providing clear evidence for chunk-dependent sequence generation.
#pagebreak()
#figure(
  image("assets/closedloop_16units_dpca.png", fit: "contain", width: 100%),
  caption: "dPCA analysis for model with 16 hidden units under feedback training"
) <fig8>
#v(1em)
#figure(
  image("assets/closedloop_32units_dpca.png", fit: "contain", width: 100%),
  caption: "dPCA analysis for model with 32 hidden units under feedback training"
) <fig9>
#v(1em)
#figure(
  image("assets/openloop_32units_dpca.png", fit: "contain", width: 100%),
  caption: "dPCA analysis for model with 32 hidden units under feedback-feedforward training"
) <fig10>
#pagebreak()
Further explorations to probe and understand neural computations in these RNNs include:
- Looking at individual hidden unit contributions by silencing them directly.
- Finding fixed points in the network's hidden activity (for example using this #text(font: "IBM Plex Mono", size: 9pt, fill: rgb("#0074d9"))[#link("https://github.com/mattgolub/fixed-point-finder")[toolbox]]) to further understand how the computations evolve in time and if they are repurposed for different targets (see Sussillo and Barak #cite(<Sussillo2013-nc>, form: "year")).
- Plotting the evolution of output phase space subject to the corresponding hidden context, to qualitatively see and contrast how position and target feedback shape the output dynamics. If this is done over hidden activity with respect to input position feedback, we can also see how the network partitions and/or reuses attractors in its latent dimensions (see Radhakrishnan et al. #cite(<Radhakrishnan2020-pnas>, form: "year")).
- Silencing dPCs without introducing a compounding effect over time by instead seeing its effect independently at each timestep by allowing the ideal hidden states to evolve and only silencing the projection on that particular dPC while producing the output at each step.

It is also possible for better generalization (i.e. reuse) to occur if target words are on the same global baseline position. Currently, the words starting with 'w' are shifted down relative to words starting with 'h' due to the initial stroke having different heights. This will require shifting the input data accordingly (during pre-processing), and the networks can be trained in a re-run of the code notebook without much further changes. However, this is not a feature that is generally true for sequences in the real world, and such situations can be better handled with minor additions to the network architecture, and so we skipped it in this project.

== Training with Delayed Feedback
To encourage the formation of internal representations that persist longer in time, are reused across multiple targets, and are less dependent on immediate feedback, we introduced a delay in the feedback provided to the network. This forced the models to simulate similar internal feedback for shared segments across target words, akin to open-loop trials but now within closed-loop operation as well. We hypothesized that this would promote adaptive chunking in the true sense (i.e. time-evolving), rather than simply using feedback to steer target-specific temporal bases to give rise to "chunks".
#figure(
  image("assets/delayfb_perf.png", fit: "contain", width: 40%),
  caption: "Delayed feedback training performance"
) <fig11>
As evident in @fig11, even high-dimensional models (with 128 units) struggled to compensate for a delay of 10 time steps and produce legible output. This suggests they were unable to generate effective internal feedback within their capacity. It is plausible that networks with even higher dimensionality are needed to adequately embed the computations necessary for learning adaptive chunks across all target words.

== Conclusion
Our investigation suggests the presence of compositionality in the trained RNNs, although not entirely of a "natural" kind. We observed indications of quasi-adaptive chunking with feedback-dependent steering of target-specific bases in feedback-only trained models. However, direct evidence for invariant and chunk-dependent shared representations evolving in time was limited and only hinted at in the feedback-feedforward trained models.

== Limitations and Future Directions
It is possible that the relatively small training set of four words allowed the networks to primarily learn target-specific operations rather than necessitating robust chunk-based composition. Expanding the training set with words like "welp" and "held" could better balance pattern and target complexity, potentially encouraging more sophisticated learning strategies. Furthermore, the use of handwritten target sequences introduces the possibility of systematic biases inherent in the dataset (e.g. target-specific spread or contraction of strokes). We could explore automated or more controlled dataset generation. It is also plausible that architectural constraints, such as a hierarchical structure, might be necessary to elicit more apparent adaptive chunking. Additionally, introducing control constraints by having the network directly control a simulated arm-like appendage for "writing" (e.g. as in the #text(font: "IBM Plex Mono", size: 9pt, fill: rgb("#0074d9"))[#link("https://motornet.org/")[motornet]] toolbox by Codol et al. #cite(<motornet>, form: "year")) could promote more naturalistic representations. Finally, we note a potential issue with the training data: "weld" and "help" had slightly longer path lengths compared to "well" and "hell". Our uniform sampling of 75 points likely resulted in longer substrokes for these words, which could have hindered the network's ability to learn chunk-based composition effectively. This limitation could be addressed in future work by using synthetic generation or uniform length-based sampling of the target words.

#bibliography("refs.bib", title: "References", style: "chicago-author-date")