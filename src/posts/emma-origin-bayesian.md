---
title: "Where Did It Come From? A Father Does the Bayesian Math"
date: "2026-06-12"
excerpt: "My daughter Emma carries an extra copy of a small stretch of DNA. The obvious question — where did it come from? — turns out to be a question about probability, not biology. This is a plain-language walk through how a single new piece of evidence (a clear test in her baby sister) quietly moves the odds, and why honest numbers come as ranges, not certainties."
category: "genetics"
---

## A note before we start

This is a short one, and it's personal. My eldest daughter, Emma, was built from a slightly different set of genetic instructions than most children — she carries an extra copy of a small stretch of DNA on one of her chromosomes. I'll spare you the coordinates; what matters for this essay is a question any parent in this situation asks within about thirty seconds of getting the report:

*Where did it come from?*

It turns out that's not really a biology question. It's a probability question. And working through it taught me something clean about how evidence actually moves belief — the kind of thing I wish someone had shown me at a kitchen table years ago instead of in a textbook. So let me show you.

No jargon without a translation sitting right next to it. If a sentence needs a dictionary, I failed.

---

## The three doors

When a child has an extra (or missing) piece of DNA, there are only a few places it can have come from. For Emma, the realistic options are three doors:

1. **Mom carries it too**, and passed it on.
2. **Dad carries it too**, and passed it on.
3. **Nobody carries it** — it appeared brand new, by chance, in the single egg or sperm that became Emma. Geneticists call this *de novo*, Latin for "anew."

That's the whole board. The question "where did it come from?" is just: *which door?*

Here's the first honest admission. At the very start, before any of us were tested, you don't get to know. You can only assign **priors** — a fancy word for "starting odds based on what's generally true for this kind of finding." For the specific stretch Emma carries, the published data say these things are *usually inherited* rather than brand-new. So the starting odds lean toward a parent carrying it, split roughly evenly between Mom and Dad, with a smaller slice for *de novo*:

> **Mom ≈ 47.5%   ·   Dad ≈ 47.5%   ·   brand-new (de novo) ≈ 5%**

Those numbers aren't magic. They're just "what tends to be true for findings like this one," before we know anything about *our* family specifically.

---

## Evidence, one piece at a time

Then the tests start coming back, and each result is a piece of evidence that should move the odds. The tool for moving odds with evidence has a name — **Bayes' theorem** — but you don't need the equation to feel how it works. The intuition is everything.

**Dad got tested first. Negative** — he doesn't carry it.

Door 2 doesn't slam completely shut (more on that in a second), but it nearly does. And here's the part that surprised me: when one door closes, the probability behind it *doesn't* get shared out evenly to the others. It flows mostly to whichever remaining door was already most likely. Because "inherited" had the high prior, almost all of Dad's lost probability slid over to **Mom**, not to *de novo*. After Dad's negative test, the odds looked roughly like:

> **Mom ≈ 90%   ·   brand-new ≈ 9%   ·   Dad anyway ≈ 1%**

(That last 1% is the honest acknowledgment that a blood test can miss a parent who carries the change only in some of their cells. Never quite zero.)

So far, so intuitive. Mom is the front-runner. We hadn't tested her yet — but then a different piece of evidence arrived from an unexpected direction.

---

## The little sister

Emma has a baby sister. And her sister's genetic test came back **clear** — she does *not* carry the extra piece.

Now: does that tell us anything about where *Emma's* came from? It feels like it shouldn't. Different child, different dice. But it does, and seeing *why* is the whole point of this essay.

Think about what each door predicts for the little sister.

- **If Mom is a carrier**, then each child has a 50% chance of inheriting the change — a coin flip per pregnancy. So under this door, there was a 50% chance the little sister would be clear.
- **If it was *de novo* in Emma** — a one-off accident in the egg or sperm that made Emma — then the little sister was essentially *never* at risk. Under this door, "little sister is clear" was basically guaranteed, near 100%.

Do you see the asymmetry? The little sister's clear result is *exactly what the de novo story predicts*, but only a *coin-flip's worth* of what the Mom-carrier story predicts. When a piece of evidence fits one explanation better than another, it nudges belief toward the explanation it fits.

The size of the nudge is just the ratio of those two predictions: 100% versus 50%, or **2 to 1** in favor of *de novo* relative to Mom. Not a knockout. A nudge.

Running that nudge through the math turns the previous odds into:

> **Mom ≈ 82%   ·   brand-new ≈ 16%   ·   Dad anyway ≈ 1%**

Mom is still the front-runner. But the chance this was a brand-new event in Emma just **roughly doubled** — from about 9% to about 16% — purely because her little sister came back clear. One healthy sibling, and the picture shifts.

---

## Why I won't give you a single number

If you asked me "so what are the odds?" and I said "82%," I'd be lying slightly — not about the math, but about the false confidence.

The honest answer is a **range**, because the single biggest input — how often this kind of change happens brand-new versus inherited — isn't known to a decimal point. It's known to a band. So I ran the same logic across the reasonable span of that input. The answer moves like this:

| If "brand-new" is really… | …then Mom is about | and de novo is about |
|---|---|---|
| rare (2%) | 91% | 8% |
| middle (5%) | 82% | 16% |
| common (10%) | 70% | 29% |

So the truthful statement isn't "82%." It's: **Mom most likely carries it — somewhere around 70 to 91% — and the chance it was a brand-new event in Emma is somewhere around 8 to 29%.** A front-runner and a live underdog, both quantified, neither pretended into false precision.

I've come to think giving the range *is* the rigor. A single number would feel more authoritative and be less true.

---

## The one thing that settles it

Here's the punchline that keeps me humble about all of the above: every number in this essay is a placeholder waiting to be deleted.

The moment Mom gets the one targeted test — checking specifically for Emma's exact change — all of this collapses to an answer. If she carries it, the odds aren't "82%," they're "yes," and the door is Mom. If she doesn't, the *de novo* door swings wide open and becomes the overwhelming favorite. One test replaces the entire probability model with a fact.

So why bother with the math at all, if a test will overwrite it? Two reasons. First, because the math tells you *which* test is worth doing, and in what order — it's a map for spending the next dollar of attention. And second, because in the waiting — and there is always waiting — the odds are how you hold the uncertainty without either panicking or pretending. They let you say something honest to yourself in the gap between the question and the answer.

---

## The thing I actually want you to take away

Strip out the genetics and this is a tiny, general lesson about how evidence works, and I find it weirdly comforting:

- You start with honest, humble starting odds.
- Each new fact moves them — and a fact moves your belief in proportion to *how much better it fits one explanation than another*. A clear result in a sibling isn't "nothing"; it's a 2-to-1 nudge, and it counts.
- When one explanation dies, its probability flows to whichever survivor was already strongest, not equally to all.
- And the intellectually honest output is usually a range, not a point — because the inputs themselves are ranges.

I do this arithmetic about my daughters at my kitchen table because the alternative — treating the unknown as either a catastrophe or a nothing — is worse. The numbers don't make me certain. They make me *calibrated*, which in the long middle of not-yet-knowing is a much kinder place to stand.

We'll test Mom. Then I'll delete the odds and write down the answer. Until then, this is how a father holds a question: with arithmetic, a range, and a refusal to pretend either way.
