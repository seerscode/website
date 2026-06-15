---
title: "Where Did It Come From? A Bayesian Walk Through a Genetic Question"
date: "2026-06-12"
excerpt: "When a proband carries an extra copy of a small stretch of DNA, the obvious question — where did it come from? — is a question about probability, not biology. This is a rigorous Bayesian treatment of parent-of-origin inference: priors from the de novo rate, the likelihood each hypothesis assigns to each test result, the posterior after a negative parental test and a clear sibling, and why honest answers arrive as ranges, not point estimates."
category: "genetics"
---

## The setup

Consider a proband built from a slightly different set of genetic instructions than most — one who carries an extra copy of a small stretch of DNA on one chromosome, a copy-number gain detected on chromosomal microarray. The coordinates do not matter for this argument. What matters is the question asked within seconds of any such report:

*Where did it come from?*

This is not, at root, a biology question. It is a probability question — a parent-of-origin inference — and working it through cleanly demonstrates something general about how evidence actually moves belief. So let us work it through.

Every term will carry its translation alongside it. If a sentence needs a glossary, the sentence has failed.

---

## The three hypotheses

For an extra (or missing) segment of DNA in a proband, the possibilities reduce to three:

1. **A parent carries it** (call this parent A), and transmitted it.
2. **The other parent carries it** (parent B), and transmitted it.
3. **Neither parent carries it** — it arose anew, by chance, in the single gamete (egg or sperm) that formed the proband. This is a *de novo* event, Latin for "afresh."

That is the entire hypothesis space. "Where did it come from?" is just: *which hypothesis?*

The first honest admission: at the outset, before anyone is tested, the answer is unknown. One can only assign **priors** — starting probabilities based on what is generally true for findings of this class. For many recurrent copy-number variants, the published literature shows they are *more often inherited* than de novo; the de novo fraction varies widely by locus but is frequently a minority of cases. So the priors lean toward a carrier parent, split roughly evenly between the two parents (absent any reason to favor one), with a smaller slice for *de novo*:

> **parent A ≈ 47.5%  ·  parent B ≈ 47.5%  ·  de novo ≈ 5%**

These figures are not arbitrary, but they are not specific to this family either. They encode "what tends to be true for findings like this," before any data about *this particular* family are in hand.

---

## Evidence, one result at a time

Then the test results arrive, and each is a piece of evidence that should move the odds. The formal tool for updating probability with evidence is **Bayes' theorem**, but the equation is not required to feel how it works. The mechanism is what matters: each hypothesis assigns a *likelihood* — a probability — to the observed result, and the hypothesis that predicted the result better gains posterior probability at the expense of the one that predicted it worse.

**Parent B is tested first. Negative** — no detectable carriage.

Hypothesis 2 does not close completely (more on that shortly), but it nearly does. Here is the part that surprises people: when one hypothesis is eliminated, its probability is *not* shared out evenly to the survivors. It flows mostly to whichever surviving hypothesis already had the higher prior. Because "inherited" carried the high prior, almost all of parent B's probability slides to **parent A**, not to *de novo*. After parent B's negative test, the posterior is approximately:

> **parent A ≈ 90%  ·  de novo ≈ 9%  ·  parent B anyway ≈ 1%**

That residual 1% is the honest acknowledgment that a blood test can miss a parent who carries the variant only in a fraction of cells — *germline or somatic mosaicism*. The likelihood of a true-negative blood test given a low-level mosaic carrier is not 100%, so the hypothesis never collapses fully to zero.

So far, intuitive. Parent A is the front-runner, and has not yet been tested directly. Then a second, indirect piece of evidence arrives.

---

## The unaffected sibling

The proband has a younger sibling, and that sibling's targeted test comes back **clear** — no carriage of the extra segment.

Does that say anything about where the *proband's* copy came from? Intuitively it feels irrelevant — different child, independent meiosis. But it is informative, and seeing *why* is the entire point.

Consider what each hypothesis predicts for the sibling — the likelihood each assigns to "sibling is clear":

- **If parent A is a carrier of an autosomal variant**, each child has a 50% chance of inheriting it — an independent coin flip per conception. So under this hypothesis, the probability the sibling would be clear was ~50%.
- **If the variant was *de novo* in the proband** — a one-off event in the single gamete that formed the proband — then the sibling was essentially never at risk. Under this hypothesis, "sibling is clear" was very nearly guaranteed, ~100%. (Not exactly 100%: rare parental germline mosaicism can produce a low recurrence risk even for an apparently de novo event, which is precisely why post-test counseling never quotes zero.)

There is the asymmetry. The sibling's clear result is *exactly what the de novo hypothesis predicts*, but only a *coin-flip's worth* of what the parent-A-carrier hypothesis predicts. Evidence that fits one explanation better than another shifts belief toward the explanation it fits.

The magnitude of the shift is the ratio of those likelihoods — the **Bayes factor** — roughly 100% versus 50%, or **2 to 1** in favor of *de novo* relative to parent A. Not decisive. A genuine nudge.

Applying that factor to the previous posterior yields:

> **parent A ≈ 82%  ·  de novo ≈ 16%  ·  parent B anyway ≈ 1%**

Parent A is still the front-runner. But the probability this was a de novo event in the proband has **roughly doubled** — from ~9% to ~16% — purely because an unaffected sibling came back clear. One healthy sibling, and the picture moves measurably.

---

## Why the honest answer is a range, not a point

Reporting "82%" would slightly misrepresent the situation — not the arithmetic, but the false confidence the single number implies.

The honest output is a **range**, because the dominant input — the de novo rate for this class of variant, which sets the prior — is not known to a decimal. It is known to a band. Propagating the same logic across the plausible span of that input gives a *sensitivity analysis*:

| If the de novo rate is really… | …then parent A is about | and de novo is about |
|---|---|---|
| low (2%) | 91% | 8% |
| middle (5%) | 82% | 16% |
| high (10%) | 70% | 29% |

So the truthful statement is not "82%." It is: **parent A most likely carries it — somewhere around 70 to 91% — and the chance it was a de novo event in the proband is somewhere around 8 to 29%.** A front-runner and a live alternative, both quantified, neither inflated into false precision.

Reporting the range *is* the rigor. A single number would feel more authoritative and be less true.

---

## The one observation that settles it

The point that keeps all of the above honest: every number here is a placeholder awaiting deletion.

The moment parent A undergoes the one targeted test — assaying specifically for the proband's exact variant — the entire model collapses to a fact. If parent A carries it, the answer is not "82%," it is "yes," and the origin is parent A. If parent A does not carry it, the *de novo* hypothesis swings open and becomes the overwhelming favorite (with the residual reserved for the mosaicism that no blood test fully excludes). One test replaces the whole probability model with an observation.

So why bother with the model if a test will overwrite it? Two reasons. First, the model identifies *which* test is worth doing, and in what order — it is a map for allocating the next unit of diagnostic effort, ensuring the most informative, least invasive assay is run first. Second, in the interval before results — and there is always an interval — the posterior is how uncertainty is held calibrated, without either overreacting or dismissing.

---

## The general lesson underneath

Strip out the genetics and this is a compact, general statement about how evidence operates:

- Begin with honest, humble priors — here, the population de novo rate.
- Each new fact updates them — and a fact moves belief in proportion to *how much better it fits one hypothesis than another* (the Bayes factor). A clear result in a sibling is not "nothing"; it is a 2-to-1 nudge, and it counts.
- When one hypothesis is eliminated, its probability flows to whichever survivor already had the higher prior, not equally to all.
- The intellectually honest output is usually a range, not a point — because the inputs themselves are ranges.

This arithmetic is worth doing because the alternative — treating an unknown as either a catastrophe or a nothing — is worse. The numbers do not deliver certainty. They deliver *calibration*, which in the long interval of not-yet-knowing is a far more defensible place to stand.

Eventually parent A is tested. Then the odds are deleted and the answer is written down. Until then, this is how a question of origin is held: with arithmetic, a range, and a refusal to pretend in either direction.
