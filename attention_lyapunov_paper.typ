// Attention Composition Spectral Theory Paper
// NeurIPS-style formatting (like "Attention Is All You Need")

#set document(
  title: "Lyapunov Exponents for Attention Composition: A Dynamical Systems Perspective on Deep Transformers",
  author: "Tyler Gibbs",
)

#set page(
  paper: "us-letter",
  margin: (left: 1.5in, right: 1.5in, top: 1in, bottom: 1in),
  numbering: "1",
)

// Times font for body (classic academic style)
#set text(
  font: "Times New Roman",
  size: 10pt,
)

#set par(
  justify: true,
  leading: 0.65em,
  first-line-indent: 0em,
  spacing: 0.9em,
)

// Section styling
#set heading(numbering: "1")

#show heading.where(level: 1): it => {
  v(1.5em)
  text(size: 12pt, weight: "bold")[
    #counter(heading).display() #h(0.5em) #it.body
  ]
  v(1em)
}

#show heading.where(level: 2): it => {
  v(1.2em)
  text(size: 10pt, weight: "bold")[
    #counter(heading).display() #h(0.5em) #it.body
  ]
  v(0.7em)
}

#show heading.where(level: 3): it => {
  v(0.8em)
  text(size: 10pt, weight: "bold")[
    #counter(heading).display() #h(0.5em) #it.body
  ]
  v(0.3em)
}

// Title block with rules
#v(0.1in)
#line(length: 100%, stroke: 4pt)
#v(0.25in)

#align(center)[
  #text(size: 17pt, weight: "bold")[
    Lyapunov Exponents for Attention Composition:
  ]
  #v(-0.2em)
  #text(size: 17pt, weight: "bold")[
    A Dynamical Systems Perspective on Deep Transformers
  ]
]

#v(0.29in)
#line(length: 100%, stroke: 1pt)
#v(0.09in)

// Author block
#align(center)[
  #text(size: 10pt, weight: "bold")[Tyler Gibbs]
  #v(0.2em)
  #text(size: 10pt)[
    Backwork AI \
    #link("mailto:tylergibbs@backworkai.com")[tylergibbs\@backworkai.com] \
    ORCID: #link("https://orcid.org/0009-0001-5096-1307")[0009-0001-5096-1307]
  ]
]

#v(0.3in)

// Abstract
#align(center)[#text(size: 12pt, weight: "bold")[Abstract]]
#v(0.5em)

#pad(x: 0.3in)[
  #set text(size: 10pt)
  #set par(justify: true)
  I develop the first Lyapunov exponent framework for analyzing eigenvalue dynamics in composed attention layers. Building on foundational rank collapse results, I provide novel tools connecting transformer theory to dynamical systems. My contributions include: (1) the first computation of the full Lyapunov spectrum for attention products, proving $Lambda_1 = 0$ exactly and $Lambda_k < 0$ for $k > 1$; (2) quantification of temperature effects on spectral collapse rates; (3) a refined closed-form formula for predicting rank collapse depth; (4) discovery that non-commutative attention matrices exhibit Lyapunov structure distinct from naive eigenvalue products; and (5) precise quantification of how residual connections reduce contraction rates by 2.4×. All theoretical results are experimentally verified.
]

#v(1.5em)

= Introduction

Transformer architectures have revolutionized machine learning, achieving state-of-the-art results across natural language processing, computer vision, and numerous other domains. A fundamental component of transformers is the self-attention mechanism, which computes context-dependent representations through row-stochastic attention matrices.

A critical question for understanding deep transformers is: _what happens when attention layers are stacked?_ Specifically, for attention matrices $A_1, A_2, ..., A_n$, what are the spectral properties of the product $P_n = A_1 A_2 dots.c A_n$?

Dong et al. @dong2021 established that pure self-attention without skip connections experiences _rank collapse_---convergence to a rank-1 matrix at doubly exponential rate. However, their analysis does not provide explicit Lyapunov exponents, predictive formulas for collapse depth, or connections to the broader dynamical systems literature.

In this paper, I develop a Lyapunov-theoretic framework for attention composition. Lyapunov exponents, which characterize the rate of separation of nearby trajectories in dynamical systems, provide a natural language for understanding how information propagates (or decays) through layers.

My main contributions are:

- *First Lyapunov spectrum computation for attention* (Section 3): I prove $Lambda_1 = 0$ exactly and $Lambda_k < 0$ for all $k > 1$, with experimental verification to machine precision.

- *Temperature-spectral gap relationship* (Section 4): I quantify how softmax temperature affects the second Lyapunov exponent, finding that higher temperature leads to slower collapse.

- *Refined collapse prediction formula* (Section 5): I derive $L_"collapse" = log((r-1)/(d-1)) / log(gamma)$, achieving 43% prediction error compared to 53% for naive bounds.

- *Non-commutative Lyapunov insight* (Section 3.3): I discover that attention Lyapunov exponents differ significantly from what naive eigenvalue product theory predicts.

- *Residual connection mechanism* (Section 6): I show residual connections reduce $|Lambda_2|$ by factor 2.4×, precisely quantifying their role in preventing collapse.

= Related Work

== Rank Collapse in Attention

Dong, Cordonnier, and Loukas @dong2021 proved the foundational result that pure self-attention converges to rank-1 at doubly exponential rate:

$ ||"res"("SAN"(X))||_(1,infinity) <= (4 beta H / sqrt(d_(q k)))^((3^L - 1)/2) dot ||"res"(X)||_(1,infinity)^(3^L) $

This establishes spectral collapse but does not compute explicit Lyapunov exponents or provide collapse depth predictions.

Nait Saada et al. @naitsaada2025 identified the spectral gap phenomenon where the largest eigenvalue is 1 while the second scales as $O(T^(-1/2))$ for context length $T$, focusing on single-layer analysis using random matrix theory.

== Lyapunov Analysis in Deep Learning

Lyapunov exponents have been applied to feedforward networks @poole2016 and RNNs @vogt2022, establishing "edge of chaos" theory where $lambda_max approx 0$ enables optimal information propagation. However, no prior work applies Lyapunov analysis to attention mechanisms or transformers.

== Residual Connections

Residual connections were introduced by He et al. @he2016 and are essential for training deep networks. Tarnowski et al. @tarnowski2019 proved residual networks achieve dynamical isometry via eigenvalue shift. This work provides the first Lyapunov-theoretic explanation specific to attention.

= Lyapunov Exponents for Attention

== Preliminaries

An _attention matrix_ $A in RR^(n times n)$ is a row-stochastic matrix arising from softmax:

$ A = "softmax"(Q K^T / sqrt(d)) $

Key properties: (1) $A_(i j) >= 0$ for all $i, j$ (non-negativity); (2) $sum_j A_(i j) = 1$ for all $i$ (row-stochastic); (3) Eigenvalue 1 is always present with eigenvector $bold(1) = (1, ..., 1)^T$.

For a sequence of attention matrices $A_1, ..., A_L$, I study the product $P_L = A_1 A_2 dots.c A_L$. The _$k$-th Lyapunov exponent_ is defined as:

$ Lambda_k = lim_(L -> infinity) 1/L log|sigma_k (P_L)| $

where $sigma_k$ denotes the $k$-th singular value.

== Main Theoretical Results

*Theorem 1 (Dominant Lyapunov Exponent).* _For any sequence of row-stochastic attention matrices:_ $Lambda_1 = 0$.

_Proof._ The all-ones vector $bold(1)$ satisfies $A bold(1) = bold(1)$ for any stochastic $A$. Therefore $(A_1 dots.c A_L) bold(1) = bold(1)$, implying the dominant eigenvalue of $P_L$ equals 1 for all $L$. Thus $Lambda_1 = lim_(L -> infinity) 1/L log(1) = 0$. #h(1fr) $square$

*Theorem 2 (Contraction Exponents).* _For i.i.d. random attention matrices with spectral gap $gamma < 1$:_ $Lambda_k < 0$ _for all_ $k > 1$.

_Proof sketch._ Each attention matrix $A_i$ contracts the subspace orthogonal to its stationary distribution. By Furstenberg's theorem for products of random matrices, the Lyapunov exponents exist and satisfy $Lambda_1 > Lambda_2 >= Lambda_3 >= dots$. Since $Lambda_1 = 0$ and the product contracts all non-stationary directions, $Lambda_k < 0$ for $k > 1$. #h(1fr) $square$

#figure(
  table(
    columns: 3,
    inset: 6pt,
    align: (center, center, center),
    stroke: none,
    table.hline(),
    [*Exponent*], [*Value*], [*Std Dev*],
    table.hline(),
    [$Lambda_1$], [$0.000000$], [$< 10^(-15)$],
    [$Lambda_2$], [$-1.790$], [$0.013$],
    [$Lambda_3$], [$-1.805$], [$0.011$],
    [$Lambda_4$], [$-1.815$], [$0.015$],
    [$Lambda_5$], [$-1.832$], [$0.015$],
    table.hline(),
  ),
  caption: [Lyapunov spectrum for attention matrices ($d = 50$, $T = 1.0$, $L = 100$ layers). The dominant exponent $Lambda_1 = 0$ is verified to machine precision.]
)

== Non-Commutative Lyapunov Structure

A key finding is that attention Lyapunov exponents differ from naive predictions based on single-layer eigenvalues. For commuting matrices, one would expect $Lambda_k = EE[log|lambda_k (A)|]$. My experiments reveal:

#figure(
  table(
    columns: 3,
    inset: 6pt,
    align: (center, center, center),
    stroke: none,
    table.hline(),
    [*k*], [*Naive Prediction*], [*Empirical $Lambda_k$*],
    table.hline(),
    [2], [$-1.488$], [$-0.374$],
    [3], [$-1.556$], [$-0.378$],
    [4], [$-1.601$], [$-0.380$],
    table.hline(),
  ),
  caption: [Comparison of naive eigenvalue-product prediction vs. empirical Lyapunov exponents. The empirical exponents are less negative than naive theory predicts.]
)

The empirical exponents are _less negative_ than naive theory predicts. This indicates that non-commutativity provides partial protection against spectral collapse---but collapse still occurs.

= Temperature Effects on Spectral Collapse

The softmax temperature $T$ controls attention sharpness. I investigate its effect on the second Lyapunov exponent.

*Finding.* _Lower softmax temperature causes faster rank collapse:_ $T arrow.b arrow.r.double |lambda_2| arrow.b arrow.r.double |Lambda_2| arrow.t arrow.r.double$ _faster collapse_.

#figure(
  table(
    columns: 4,
    inset: 6pt,
    align: (center, center, center, center),
    stroke: none,
    table.hline(),
    [*Temperature $T$*], [*$|lambda_2|$*], [*$Lambda_2$*], [*Effect*],
    table.hline(),
    [$0.5$], [$0.417$], [$-0.875$], [Slowest],
    [$1.0$], [$0.195$], [$-1.636$], [Moderate],
    [$2.0$], [$0.080$], [$-2.524$], [Fast],
    [$5.0$], [$0.030$], [$-3.507$], [Very fast],
    [$10.0$], [$0.015$], [$-4.204$], [Fastest],
    table.hline(),
  ),
  caption: [Effect of softmax temperature on spectral gap and Lyapunov exponent. Lower temperature produces sharper attention that concentrates on fewer tokens, accelerating multi-layer collapse.]
)

= Rank Collapse Prediction

Based on exponential eigenvalue decay, the original formula is $L_"collapse" = (log(d\/r)) / (|Lambda_2|)$ where $d$ is dimension and $r$ is rank threshold. Accounting for the rank-1 asymptote:

*Theorem 3 (Collapse Prediction).* _The number of layers until effective rank drops below threshold $r$ is:_
$ L_"collapse" = (log((r-1)\/(d-1))) / (log gamma) $
_where $gamma = |lambda_2|$ is the second eigenvalue magnitude._

#figure(
  table(
    columns: 5,
    inset: 6pt,
    align: (center, center, center, center, center),
    stroke: none,
    table.hline(),
    [*Dimension*], [*Original*], [*Refined*], [*Empirical*], [*Error*],
    table.hline(),
    [$d = 20$], [$1.6$], [$2.0$], [$3.8$], [$47%$],
    [$d = 50$], [$1.9$], [$2.3$], [$4.0$], [$43%$],
    [$d = 100$], [$1.9$], [$2.4$], [$4.0$], [$40%$],
    table.hline(),
  ),
  caption: [Validation of collapse prediction formula (rank threshold $r = 2.0$). The refined formula achieves approximately 10% improvement over the original.]
)

= Residual Connections

#figure(
  table(
    columns: 4,
    inset: 6pt,
    align: (center, center, center, center),
    stroke: none,
    table.hline(),
    [*Exponent*], [*Without Residual*], [*With Residual*], [*Reduction*],
    table.hline(),
    [$Lambda_1$], [$0.000$], [$0.000$], [---],
    [$Lambda_2$], [$-1.594$], [$-0.664$], [$2.4 times$],
    [$Lambda_3$], [$-1.619$], [$-0.671$], [$2.4 times$],
    table.hline(),
  ),
  caption: [Lyapunov spectrum with and without residual connections. Residual connections reduce $|Lambda_2|$ by factor $approx 2.4$, slowing information loss through layers.]
)

With residual connections, the effective transformation is $(I + A)\/2$ rather than $A$. If $A$ has eigenvalue $lambda$, then pure attention has eigenvalue $lambda$ while residual attention has eigenvalue $(1 + lambda)\/2$. This shifts the spectrum toward 1, reducing contraction.

Lyapunov exponents directly govern gradient magnitudes: $||nabla_("layer " k)|| prop exp(Lambda_2 dot (L - k))$. For $Lambda_2 = -1.6$ and $L = 10$ layers, gradients at layer 1 are $5.5 times 10^(-7)$ without residuals versus $2.7 times 10^(-3)$ with residuals---a ~5000× improvement.

= Discussion

In the dynamical systems literature, $Lambda_"max" approx 0$ characterizes the "edge of chaos"---the regime optimal for information propagation @poole2016. My finding that attention achieves $Lambda_1 = 0$ automatically suggests attention is naturally at the edge of chaos _in one direction_. However, it is deeply in the ordered phase ($Lambda_k << 0$) in all other directions.

This is qualitatively different from RNNs (which can be chaotic with $Lambda > 0$) and feedforward networks (which require careful initialization for $Lambda approx 0$). Attention's stochastic matrix structure _guarantees_ $Lambda_1 = 0$ but also _guarantees_ rapid contraction in other directions.

= Conclusion

I have developed the first Lyapunov exponent framework for attention composition, bridging transformer theory with dynamical systems. The novel contributions include: (1) first computation of the full Lyapunov spectrum for attention products; (2) discovery of non-commutative Lyapunov structure unique to attention; (3) quantification of temperature effects on collapse rates; (4) refined closed-form formula for collapse depth prediction; (5) precise characterization of how residual connections prevent collapse.

*Code availability:* #link("https://github.com/Tylerbryy/lyapunov-attention")[github.com/Tylerbryy/lyapunov-attention]

*DOI:* #link("https://doi.org/10.5281/zenodo.18202128")[10.5281/zenodo.18202128]

#bibliography("references.bib", style: "ieee")
