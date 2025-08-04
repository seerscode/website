
---
title: "The Challenges of Superposition in LLMs"
date: "2025-07-28"
excerpt: "How neural networks represent more features than they have neurons, and why this complicates interpretability efforts."
---

# What is Superposition?

In the context of neural networks, superposition refers to the phenomenon where a model represents more features than it has dimensions (neurons) in a given layer. It achieves this by packing multiple, often unrelated, features into the activity of a single neuron—a state known as polysemanticity.

## Efficiency vs. Interpretability

Superposition is likely a strategy the network develops during training to maximize efficiency, especially when features in the input data are sparse. However, it poses a significant challenge for researchers trying to understand the model.

If a single neuron activates for "cat," "car," and "Cauchy distribution," simply looking at that neuron's activations won't tell us much.

## Dictionary Learning

Current research focuses on techniques like Dictionary Learning (e.g., Sparse Autoencoders) to disentangle these superposed features into an overcomplete basis, making them individually interpretable.
