---
title: "How Do You Know When You've Looked Everywhere?"
date: "2026-06-14"
excerpt: "\"The genetic test came back\" and \"the genome has been examined\" are very different statements. This is a field guide to genetic-test completeness: what each assay physically resolves, what it is constitutively blind to, and why a single result — even a negative one — almost never closes the diagnostic question. A structured map of the workup, assay by assay, for reading a report and knowing what remains unexamined."
category: "genetics"
---

## The statement that sounds like an answer but isn't

"The genetic test came back."

It is tempting to read that as a door closing — either it found something or it didn't, and either way the matter is now settled. But once the architecture of these assays is understood, the statement turns out to mean something closer to "one drawer was checked." There are many drawers. Each test opens exactly one of them and is largely blind to the rest. This is rarely spelled out, because it is inconvenient and a little unsettling, so it is worth setting down plainly — for anyone handed a result who is unsure whether it actually resolves anything.

When a structural change appears — say, an extra or missing copy of a stretch of DNA — the natural next question is whether the finding is the *whole* story, part of the story, or an incidental bystander caught on camera. Answering that does not mean running "a genetic test." It means running a series of mechanistically distinct assays, each interrogating a different *class* of variation. The discipline lies in knowing which classes have been excluded and which have not yet been glanced at.

## Different assays detect different classes of variation

There is no single "is the DNA intact" test. There are several, and their detection windows overlap far less than one might hope. It helps to think of DNA changes as different categories of edit in a very long document:

- **A whole paragraph was duplicated or deleted.** Large, structural, detectable if one is measuring *dosage* — copy number. The instrument is the **chromosomal microarray** (also reported as **array-CGH** or SNP array). It is typically run first, and it resolves gains and losses down to roughly tens of kilobases in targeted regions. But it measures copy number only. It cannot read sequence, it cannot detect balanced rearrangements (a translocation or inversion that moves material without changing the total dose), and it is blind to everything below.
- **A single letter differs** — a point change that alters one codon and flips the meaning. Dosage measurement cannot see this; the bases must actually be read. That is **whole-exome sequencing (WES)**, which sequences the ~1–2% of the genome that codes for protein. Run as a *trio* — proband plus both parents — it gains substantial power, because it distinguishes inherited variants from those arising *de novo*, and de novo status in a constrained gene is a strong pathogenicity signal.
- **A short motif was over-repeated** — the same few bases tandemly expanded far beyond the normal range, like a key held down too long. This requires a dedicated **repeat-expansion assay** (the canonical example is **Fragile X / FMR1** CGG-repeat testing, historically by repeat-primed PCR and Southern blot). Short-read sequencing routinely *misses* large expansions, because the reads are shorter than the repeat tract and cannot be uniquely mapped across it, unless a specialized expansion caller is explicitly applied.
- **The bases are correct, but the wrong genes are switched on or off.** The text is intact; the annotation is wrong. This is the regulatory chemistry layered *on top* of the sequence — DNA methylation and the associated chromatin marks that set each gene's expression level — and it is read with **methylation testing**, commonly **MS-MLPA**. The same assay also flags **uniparental disomy**, in which both homologs of a chromosome derive from a single parent, disrupting imprinted loci. A pure sequence read sees nothing wrong here, because nothing is wrong with the sequence itself.
- **The change is mosaic — present in some cells, absent in others.** If a variant arose after the first cell division, a blood sample can be entirely clean while the variant resides in an unsampled tissue. Detecting this means **sampling more than one tissue** (a buccal swab, or banked fibroblasts cultured from a small skin biopsy) and, increasingly, sequencing to high depth so low-frequency alleles are not lost in the noise.

Five classes of variation. Five different instruments. And the operative fact is that each instrument is *dark* to the others. The dosage assay cannot resolve a point mutation. The sequence read cannot resolve an epigenetic misconfiguration. Running one, obtaining a clean result, and concluding "clean means all clear" is a category error. It means *that one drawer* was empty.

## The ladder

The useful reframe is to stop treating this as a single test and treat it as a ladder. Each rung is a different class of variation, and the only honest question is: how many rungs have actually been stood on?

The bottom rung — large structural change, copy number — is the **chromosomal microarray**, which is usually where a copy-number finding surfaces in the first place. (A conventional **karyotype** sits alongside it for visualizing whole-chromosome aneuploidy and balanced rearrangements that an array, measuring only dose, will miss.)

The next rung is the one most often omitted: the regulatory layer — the switched-on-or-off chemistry — read with **methylation testing (MS-MLPA)**. A microarray physically *cannot* resolve it. Not "probably didn't" — *cannot*; it is the wrong measurement. It is comparatively inexpensive and specific, and it is the rung most frequently skipped. Above it sits the **repeat-expansion / Fragile X (FMR1)** assay, then the slow, comprehensive **read-every-coding-letter test — whole-exome sequencing (WES)**, then **whole-genome sequencing (WGS)** to reach the noncoding regulatory regions, deep-intronic splice variants, and structural breakpoints the exome skips. Reserved for the residual cases are the assays for rare structural mechanisms: **long-read sequencing** (which spans repeats and resolves complex rearrangements short reads cannot phase), **optical genome mapping** (which images very large structural and balanced events at megabase scale), and **mitochondrial DNA sequencing** (a separate genome, with heteroplasmy that nuclear assays do not assess).

The point worth holding onto: a workup can be **much further along than "we did a genetic test" implies, and much less finished than "the test was negative" implies.** Both statements are traps. The reality is a checklist — and a checklist is something that can be worked through one rung at a time.

## Why a negative result is not the closure it appears to be

Suppose the comprehensive sequence read comes back clean — **trio whole-exome sequencing (WES)** on the proband and both parents. The instinct is relief, followed almost immediately by a sharper unease: if it is clean, *what is the explanation?*

This is the intuition worth retraining. A negative result on one rung does not lower the prior probability that something is there. It redistributes the remaining probability onto the rungs that have *not* been climbed. A clean exome does not assert "there is nothing." It asserts "it is not a coding point variant in a known gene — so the remaining probability now sits on methylation, repeat expansions, noncoding and structural variation that a whole-genome read would reach, and mosaicism a single tissue would miss." A negative does not close the search. It *aims* it.

There is also a humbler possibility worth room for. Many structural findings are *incompletely penetrant* and *variably expressive* — in most individuals who carry them, they produce little or no phenotype. The same copy-number variant can be pathogenic in one carrier and silent in another, depending on the variant on the second allele, modifier loci, and stochastic developmental factors. A finding can therefore be one contributor among many small ones rather than a sole cause, with no single villain to indict. That is not a satisfying resolution. It can still be the correct one. Doing this rigorously means being willing to hold an explanation that does not collapse to a single clean cause.

## The ceiling is below one hundred percent

It is worth being explicit about the ceiling, because the pull toward *certainty* is strong and the data do not support it.

One can climb a long way — a handful of assays, two of them inexpensive, covers most of the ladder. But the final stretch never fully closes, for three honest reasons:

> **Not every disease gene is known yet  ·  the hardest regions of the genome still cannot be read cleanly  ·  some phenotypes are distributed across hundreds of small contributions, not one**

The first reason keeps a peculiar door open. A variant of uncertain significance today — a result no one can yet interpret — can become a diagnosis in a few years as gene–disease relationships are established. Stored raw data are re-interpretable later. So one of the highest-yield actions is not a new assay at all: it is retaining the raw sequence files and requesting periodic *reanalysis* — typically every one to two years — against updated databases as the field accumulates knowledge.

There is something genuinely useful in that. Not "everything will be known," but "today's data can keep working years from now without being regenerated."

## What to do with all of this

The shift that helps is to stop asking "is the test back yet" as though one envelope ends the story, and instead maintain a single page — a ladder with checkboxes. Which rung is done, which is next, where it is performed, and precisely what it can and cannot resolve. When a result lands, the question is not "good or bad." It is "which rung did this clear, and which rung is now the most important thing no one has examined."

That substitution converts an open-ended unknown into a finite task list. The unknown has no edges. A task list does.

One may never learn which rung holds the answer, or whether there is a single answer to hold. But one can know exactly which drawers have been opened and which have not — and that no one gets to quietly declare the matter finished while drawers remain shut. Some days that is the most the science can offer. Most days, it is enough.
