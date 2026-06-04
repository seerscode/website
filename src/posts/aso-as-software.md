---
title: "ASO Design as Software Engineering"
date: "2026-05-25"
excerpt: "Antisense oligonucleotides are the closest thing rare disease has to a programmable therapy. The mental model that helped me move fastest into the field is treating ASO development as a software stack — with the same kinds of trade-offs, the same kinds of debugging, and the same kinds of failure modes."
category: "therapeutics"
---

Coming from software, the first thing that helped me understand antisense oligonucleotides was treating the whole stack as engineering rather than biology. The metaphor is not perfect, but it is close enough that the failure modes line up.

An ASO is, roughly, a regex against a messenger RNA. You write a short pattern — typically 18 to 20 nucleotides of chemically modified DNA — and the cell's RNase H finds the matching transcript and cuts it. You have written a query that runs against the cell's RNA pool, and the side effect of the match is deletion.

That mental model gets you surprisingly far. Selecting a good ASO sequence is a search problem: you screen all the candidate windows along a transcript, score them for predicted efficacy and off-target risk, and rank. Off-target risk is BLAST against the transcriptome. Toxicity is a runtime issue: the same sequence that looks great in silico can trigger immune responses or pull in proteins it should not. The development pipeline behaves like CI — you have a series of gates (in vitro potency, in vivo PK/PD, tox in two species, GLP tox, IND) that each filter out candidates, and the cost per gate goes up by an order of magnitude.

The build-versus-buy decision in ASO development looks like the same decision in software infrastructure. You can build the design pipeline yourself — open-source thermodynamic prediction, public off-target databases, in-house BLAST — for almost no money. You can also outsource the entire stack to a CDMO that hands you a finished candidate. The middle ground, which is where most small programs end up, is a hybrid: design in-house, synthesis at a vendor like Axolabs or LGC, in vitro screening at a CRO, in vivo at a different CRO, GLP tox at a third. Each handoff is an API boundary, and the data formats are usually not compatible.

The chemistry — MOE gapmers, LNA wings, phosphorothioate backbones — sits at roughly the same conceptual level as choosing a language runtime. The decision matters enormously for the final product, but the design logic above it does not change much. You pick a chemistry based on target tissue, half-life requirements, and tolerability profile, the same way you pick between Python and Rust for a service.

Where the analogy breaks is biology. Software is deterministic; ASOs are not. Two animals dosed with the same compound at the same level produce different responses, and the variance does not always shrink with N. You also cannot unit test against a real human nervous system. The closest you have are induced pluripotent stem cell-derived neurons, organoids, and animal models that approximate but never reproduce the target condition. Every gate in the pipeline is a noisy estimator, and the noise compounds.

The other place it breaks is reversibility. A bad deploy in software is a rollback away. A bad ASO dose is permanent until the molecule clears, which depending on chemistry can be weeks to months. CSF dosing for CNS programs adds an additional irreversibility: an intrathecal injection cannot be retrieved. This forces a much more conservative posture on first-in-human dose escalation than software engineers are used to.

The development timeline also bends differently. A software feature that takes a senior engineer two weeks is roughly two weeks regardless of company size. A first-in-human ASO program takes 4 to 6 years independent of how well-funded you are, because most of the time is sitting in the calendar of regulators and CROs rather than in engineering effort. The forcing function is not throughput, it is the slowest serial gate. Optimization looks like parallelization wherever the regulatory path allows it.

I find the analogy useful mostly because it makes the field approachable for people coming from other disciplines. Antisense is one of the smallest stacks in modern therapeutics: a target gene, a sequence, a delivery vehicle, a dose, a route. There are at most a few hundred parameters across the whole development plan. That is a smaller surface area than a typical mid-size web application. The complexity is in the runtime — the living patient — not in the design.

If more software engineers were willing to look at therapeutics this way, more rare disease programs would exist. The bottleneck right now is not money or chemistry. It is the number of people willing to ship.
