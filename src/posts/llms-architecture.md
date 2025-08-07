
---
title: "A Comprehensive Architectural Review of Large Language Models"
date: "2025-08-04"
excerpt: "This report provides a comprehensive, expert-level deconstruction of this architectural evolution."
---

### Introduction

In recent years, Large Language Models (LLMs) have emerged as a transformative force in technology and society, demonstrating remarkable capabilities in understanding, generating, and manipulating human language [1]. These models, which are a specialized category of deep learning, are trained on immense volumes of text data, enabling them to perform a vast array of natural language processing (NLP) tasks. These tasks range from generating coherent articles and software code to translating languages, summarizing complex documents, and engaging in conversational dialogue [1]. The applications of this technology are already reshaping entire industries, including customer service, content creation, finance, healthcare, and scientific research [1].

The central thesis of this report is that the sophisticated abilities of modern LLMs are not a product of a single breakthrough but are the culmination of a profound and rapid architectural evolution. This journey has been characterized by a series of conceptual shifts, each designed to overcome the fundamental limitations of its predecessor. The architectural lineage of LLMs begins with early attempts to model the sequential nature of language using Recurrent Neural Networks (RNNs), which, despite their ingenuity, were hobbled by critical flaws in memory and computation. The field then witnessed a paradigm shift with the introduction of the Transformer architecture, which abandoned recurrence in favor of a parallelizable "attention" mechanism, unlocking unprecedented scalability and contextual understanding.

This report provides a comprehensive, expert-level deconstruction of this architectural evolution. It begins by establishing the foundational principles of sequence modeling, examining the design of RNNs and their more advanced gated variants like Long Short-Term Memory (LSTM) networks, and analyzing the inherent weaknesses that necessitated a new approach. The core of the report is a meticulous, component-by-component breakdown of the seminal Transformer architecture, detailing its mechanisms for parallel processing, self-attention, and deep contextualization. Following this, the analysis will explore how this foundational architecture was specialized into the dominant families of modern LLMs: encoder-only models like BERT, designed for deep understanding; decoder-only models like GPT, designed for fluent generation; and encoder-decoder models like T5, designed for sequence-to-sequence transformations.

Finally, the report will venture to the cutting edge of research, examining frontier architectures such as the computationally efficient Mixture of Experts (MoE) and the linear-time State Space Models (Mamba), which seek to address the remaining bottlenecks of the Transformer. It will also cover the critical training and alignment paradigms—including scaling laws, instruction tuning, and Reinforcement Learning from Human Feedback (RLHF)—that are essential for creating the powerful, helpful, and safe LLMs in use today. Through this exhaustive review, we will illuminate the intricate relationship between architectural design and emergent capability, providing a definitive guide to the inner workings of the models that are defining the current era of artificial intelligence.

***

### Section 1: The Foundations of Sequence Modeling: Recurrence and Its Limitations

To fully appreciate the revolutionary impact of the Transformer, it is essential to first understand the architectural paradigm it replaced. For many years, the dominant approach to processing sequential data like language was rooted in the concept of recurrence. This section establishes the historical and technical context for the Transformer's design by exploring the strengths and, more critically, the inherent failures of recurrent models. The limitations of this paradigm were not merely incremental issues but fundamental bottlenecks that ultimately paved the way for a complete architectural rethinking.

#### 1.1 The Challenge of Language as a Sequence

Language is fundamentally a sequence, where the order of words is not arbitrary but is crucial for conveying meaning [7]. The meaning of a word is heavily dependent on the context provided by the words that precede and often follow it [9]. For example, in the phrase "turn down the music," the meaning of "down" is determined by its relationship with "turn." A traditional feedforward neural network, which processes all its inputs simultaneously and independently, is structurally incapable of capturing such temporal dependencies [7]. These networks assume that inputs are independent of one another, an assumption that is explicitly violated by sequential data. The central challenge for early NLP researchers was therefore to design a neural network architecture that possessed a form of "memory," allowing it to retain and utilize information from previous parts of a sequence when processing the current part.

#### 1.2 Recurrent Neural Networks (RNNs): The Concept of a Hidden State

Recurrent Neural Networks (RNNs) were the first major architecture designed to solve this problem [7]. Unlike a feedforward network, the core of an RNN is a recurrent unit featuring a loop. This recurrent connection feeds the output of a neuron at one time step back as an input to that same neuron at the subsequent time step [7]. This simple yet powerful mechanism allows the network to maintain a "hidden state" (**h_t**), a vector that serves as a compressed summary or memory of all previous inputs in the sequence up to that point [7].

The operation of a basic RNN cell at each time step **t** can be described by the following equation:

**h_t = σ(W_hh * h_t-1 + W_xh * x_t + b_h)**

where **x_t** is the input vector at the current time step, **h_t-1** is the hidden state from the previous time step, **W_hh** and **W_xh** are weight matrices for the recurrent and input connections respectively, **b_h** is a bias term, and **σ** is a non-linear activation function (typically tanh or ReLU) [10]. The output at time step **t**, **y_t**, is then typically calculated from the hidden state:

**y_t = W_hy * h_t + b_y**

For the purposes of training and analysis, this compact, looped representation is often "unfolded" or "unrolled" through time [7]. Unfolding transforms the RNN into a deep feedforward-like network, where each "layer" corresponds to a single time step in the sequence. A crucial feature of this unrolled network is that the weight matrices (**W_hh**, **W_xh**, **W_hy**) are shared across all time steps, which drastically reduces the number of parameters to be learned and allows the model to generalize its learned patterns across different positions in the sequence [9].

RNNs are trained using an algorithm called Backpropagation Through Time (BPTT). BPTT is an adaptation of the standard backpropagation algorithm applied to the unrolled network structure. It calculates the error at the final output and propagates this error signal backward through time, from the last time step to the first, to compute the gradients needed to update the shared weights [8].

#### 1.3 The Problem of Long-Term Dependencies: Vanishing and Exploding Gradients

While RNNs theoretically have the capacity to connect information across arbitrary time lags, in practice they suffer from a critical flaw that severely limits their memory: the vanishing and exploding gradient problem [9]. This issue arises directly from the nature of BPTT and the repeated application of weights through the unrolled network.

During BPTT, the gradient at an early time step is calculated using the chain rule, which involves a long product of derivatives of the activation function and the recurrent weight matrix **W_hh** for each intervening time step [14].

* **Vanishing Gradients:** If the values in this repeated multiplication (specifically, the norms of the weight matrix and the derivatives of the activation function) are consistently less than 1, the gradient signal shrinks exponentially as it is propagated backward through time. By the time the gradient reaches the earliest layers of the unrolled network, it can become so small as to be effectively zero [12]. This "vanishing" of the gradient means that the weights connecting to early inputs receive no meaningful update signal, making it impossible for the model to learn dependencies between events that are far apart in the sequence. The practical result is that a standard RNN has a very short-term memory, often struggling to remember context beyond a few time steps [12].
* **Exploding Gradients:** Conversely, if the values in the chain rule's product are consistently greater than 1, the gradient can grow exponentially, becoming astronomically large [9]. These "exploding" gradients lead to massive, unstable updates to the network's weights, causing the training process to become erratic and diverge. While this issue is often easier to detect and can be partially mitigated by techniques like gradient clipping (capping the gradient at a certain threshold), it does not solve the more insidious underlying problem of long-term dependency learning [12].

#### 1.4 Gated Architectures: A More Sophisticated Memory Cell

The failure of simple RNNs to handle long-term dependencies spurred the development of more complex recurrent units. These new architectures, most notably the Long Short-Term Memory (LSTM) and the Gated Recurrent Unit (GRU), did not alter the high-level recurrent structure but fundamentally redesigned the internal mechanics of the cell to combat the vanishing gradient problem [10].

The evolution from a simple RNN to a gated architecture like LSTM represents a critical conceptual leap. The passive, constantly overwritten memory of an RNN is replaced with an active, managed memory system. The introduction of gates is one of the earliest and most important examples of conditional computation in sequence modeling. The network learns to execute different operations—to write, to read, or to erase parts of its memory—based on the context of the input data. This principle of intelligent, input-dependent information routing is a conceptual ancestor to the highly advanced mechanisms seen in modern architectures like Mixture of Experts, which apply the same idea of conditional computation at a much larger scale.

##### Long Short-Term Memory (LSTM)

Introduced by Hochreiter and Schmidhuber in 1997, the LSTM network was explicitly designed to overcome the long-term dependency problem [17].

* **Core Idea:** The central innovation of the LSTM is the introduction of a dedicated "cell state" (**C_t**). This cell state acts as an information conveyor belt, running straight down the entire sequence with only minor linear interactions [18]. This design allows information to flow through the network unchanged, preserving it over long time periods. This cell state is what provides the LSTM with its "long-term memory."
* **The Gates:** The key to the LSTM's power is a set of three "gates." These are neural network layers (typically a sigmoid activation function followed by an element-wise multiplication) that regulate the flow of information into and out of the cell state [17]. They learn which information is important to keep or discard.
    * **Forget Gate (f_t):** This gate decides what information to throw away from the previous cell state, **C_t-1**. It looks at the current input **x_t** and the previous hidden state **h_t-1** and outputs a vector of numbers between 0 and 1 for each number in the cell state. A 1 represents "completely keep this," while a 0 represents "completely get rid of this" [17].
    * **Input Gate (i_t):** This gate decides what new information to store in the cell state. It consists of two parts: a sigmoid layer that decides which values to update, and a tanh layer that creates a vector of new candidate values, that could be added to the state [14]. These two are combined to update the cell state.
    * **Output Gate (o_t):** This gate decides what information to output from the cell state. The output will be based on the cell state, but will be a filtered version. First, a sigmoid layer decides which parts of the cell state to output. Then, the cell state is passed through a tanh function (to push the values to be between -1 and 1) and multiplied by the output of the sigmoid gate, so that only the desired parts are output [19].

The update mechanism of the LSTM cell state involves an additive interaction, which is crucial. Instead of multiplying the old state by a new value (as in a simple RNN), it adds the new information, controlled by the input gate, after selectively forgetting old information. This additive nature, protected by the gates, allows gradients to flow back through time without vanishing or exploding, enabling the model to learn dependencies over hundreds or even thousands of time steps [13].

##### Gated Recurrent Unit (GRU)

The Gated Recurrent Unit is a more recent and slightly simpler variant of the LSTM [7]. It combines the forget and input gates into a single "update gate" and also merges the cell state and hidden state. GRUs have fewer parameters than LSTMs and are often computationally more efficient, while demonstrating comparable performance on many NLP tasks [9].

The inherent limitations of these recurrent architectures were not merely computational but also conceptual. Their strictly sequential, one-token-at-a-time processing imposes a linear and constrained view of language. When an RNN processes a sentence like, "The cat, which was chased by the dog, sat on the mat," the information about the subject "cat" becomes progressively diluted as it passes through the intermediate states associated with "which," "was," "chased," etc., before reaching the verb "sat" [8]. Furthermore, a standard unidirectional RNN processing "sat" has no access to the future context of "mat," which is often crucial for disambiguation. While Bidirectional RNNs (Bi-RNNs) were developed to address this by processing the sequence in both forward and backward directions, this approach doubles the computational cost and still relies on an indirect, sequential propagation of context [7]. This fundamentally inefficient model of context handling was a primary motivator for the field to seek a non-sequential solution, leading directly to the development of the Transformer.

***

### Section 2: The Transformer: A New Paradigm in Parallel Processing

The introduction of the Transformer architecture in the 2017 paper "Attention Is All You Need" by Vaswani et al. marked a watershed moment in the history of deep learning and NLP [22]. It proposed a novel architecture that dispensed with recurrence and convolutions entirely, instead relying solely on a mechanism called "self-attention" [26]. This design choice solved the sequential computation bottleneck of RNNs, enabling massive parallelization and paving the way for training models of unprecedented scale. This section provides a meticulous deconstruction of the Transformer, examining each of its components from the high-level concept down to the intricate mathematical and engineering details that make it work.

#### 2.1 Escaping the Sequential Trap: Input Representation

The core innovation of the Transformer is its ability to process all tokens in an input sequence simultaneously, a stark contrast to the one-by-one processing of RNNs [5]. This parallelization is key to its efficiency and scalability. However, by abandoning recurrence, the model loses the inherent sense of word order that RNNs possess. This creates a new challenge: how can a model that sees all words at once understand their sequence? [29]. The Transformer solves this through a two-step input representation process.

##### Tokenization and Input Embeddings

First, the raw input text is segmented into a sequence of smaller units called tokens. These tokens can be words, sub-words, or even characters, and the process is managed by a tokenizer algorithm [28]. Each unique token in the model's vocabulary is then mapped to a high-dimensional vector via an "embedding" layer. This embedding is a learned representation that captures the semantic meaning of the token [29]. For instance, tokens with similar meanings like "king" and "queen" will have embedding vectors that are close to each other in the vector space. The dimensionality of these embedding vectors is a critical hyperparameter of the model, denoted as **d_model**, which was set to 512 in the original Transformer paper [26].

##### Positional Encoding

To reintroduce the notion of order into the model, a "positional encoding" vector is generated and added to each token's embedding vector [27]. This vector provides the model with explicit information about the absolute or relative position of each token within the sequence. While this could be a learned embedding, the original paper proposed a clever and efficient fixed method using sine and cosine functions of varying frequencies [30].

For a token at position **pos** in the sequence and for each dimension **i** of the embedding vector, the positional encoding **PE** is calculated as follows:

**PE(pos, 2i) = sin(pos / 10000^(2i / d_model))**
**PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))**

Here, **2i** refers to the even-indexed dimensions and **2i+1** to the odd-indexed dimensions of the embedding vector. This formulation has several elegant properties. Each position is assigned a unique encoding vector. More importantly, because the encoding for position **pos+k** can be represented as a linear function of the encoding for **pos**, the model can easily learn to attend to relative positions. This method also has the advantage of being able to generalize to sequence lengths longer than those encountered during training [30]. The final input to the first layer of the Transformer is the sum of the token embedding and its corresponding positional encoding.

#### 2.2 The Core Mechanism: Scaled Dot-Product Attention (Self-Attention)

With the input prepared, the central mechanism of the Transformer comes into play: self-attention. Self-attention, sometimes called intra-attention, is an attention mechanism that relates different positions of a single sequence in order to compute a contextualized representation of that sequence [26]. It allows the model to look at other words in the input sequence for clues and to weigh their importance when processing a specific word [31].

##### Query, Key, and Value (QKV) Vectors

The self-attention mechanism operates on three vectors derived from each input embedding (which already includes positional information): the **Query**, the **Key**, and the **Value**. These vectors are generated by multiplying the input embedding by three separate weight matrices (**W_Q**, **W_K**, **W_V**), which are learned during the training process [31].

An intuitive analogy is to think of this process as a retrieval system, like searching for a video online [38]:

* **Query (Q):** This is your search query. In the context of a sentence, it represents the current token's perspective, effectively asking, "Given my role, what other tokens in this sentence are relevant to me?" [38].
* **Key (K):** These are like the keywords or tags associated with each video. Each token in the sentence has a Key vector that represents its own attributes or what it "offers." The Query is matched against all the Keys to find the most relevant ones [38].
* **Value (V):** This is the actual content of the video. Once the Query-Key match determines the relevance of other tokens, their Value vectors are used to construct the output. The Value vector contains the information that is actually passed on [38].

##### The Attention Formula

The output of the attention mechanism is calculated using a specific formula that encapsulates the QKV logic [22]:

**Attention(Q, K, V) = softmax((Q * K^T) / sqrt(d_k)) * V**

This calculation can be broken down into four steps [37]:

1.  **Calculate Scores:** The first step is to measure the compatibility or similarity between the Query of the current token and the Key of every other token in the sequence. This is done by computing the dot product of the Query vector **Q** with the transpose of the Key matrix **K^T**. This results in a matrix of attention scores, where each score indicates how much token *i* should attend to token *j* [37].
2.  **Scale:** The scores are then divided by the square root of the dimension of the key vectors, **sqrt(d_k)**. This is a critical normalization step. For large values of **d_k**, the dot products can grow very large in magnitude, pushing the softmax function into regions where it has extremely small gradients. This scaling factor counteracts that effect, ensuring more stable gradients and facilitating effective learning [22].
3.  **Softmax:** A softmax function is applied to the scaled scores along each row. This converts the scores into a set of positive weights that sum to 1. These weights represent a probability distribution, indicating the amount of "attention" each token should receive relative to the current token [31].
4.  **Weighted Sum of Values:** Finally, the attention weights are used to create a weighted sum of the Value vectors. The resulting vector is the output for the current token—a new, contextualized representation that is a blend of information from all other tokens in the sequence, weighted by their calculated relevance [26].

#### 2.3 A Broader Perspective: Multi-Head Attention

A single self-attention mechanism might learn to focus on a particular type of relationship, for instance, how verbs relate to their subjects. However, language is complex, with many layers of relationships (syntactic, semantic, co-reference, etc.). To allow the model to jointly attend to information from different representation subspaces at different positions, the Transformer employs "Multi-Head Attention" [22].

The intuition is that instead of performing one large attention calculation, it is more beneficial to have multiple "attention heads" that can each learn to focus on different aspects of the language in parallel [41].

The mechanism works as follows [26]:

1.  Instead of having a single set of Q, K, and V weight matrices, Multi-Head Attention has **h** separate sets (the original paper used **h=8** heads) [26].
2.  The input embedding vector is projected into **h** different, typically smaller-dimensional, sets of Q, K, and V vectors using these **h** distinct learned weight matrices. Each of these projected sets is an "attention head" [40].
3.  The Scaled Dot-Product Attention function is then performed in parallel for each of these **h** heads. This results in **h** separate output vectors, each capturing a different type of contextual relationship [26].
4.  These **h** output vectors are then concatenated back together to form a single, full-dimensional vector [40].
5.  This concatenated vector is passed through one final linear projection layer, governed by a learned weight matrix **W_O**, to produce the final output of the Multi-Head Attention block [42].

This multi-headed approach allows the model to build a much richer and more nuanced representation of the input. For example, one head might learn to track subject-verb agreement, while another tracks pronoun antecedents, and a third tracks semantic similarity, all at the same time [40].

#### 2.4 Building Blocks: The Full Encoder-Decoder Architecture

The original Transformer was designed for machine translation, which is a sequence-to-sequence task. As such, its architecture consists of two main components: an Encoder stack and a Decoder stack [5].

##### The Encoder Stack

The encoder's role is to process the entire input sequence and build a rich, contextualized representation of it [34]. It is composed of a stack of **N** identical layers (e.g., **N=6** in the original paper) [26]. Each encoder layer contains two primary sub-layers:

* A Multi-Head Self-Attention mechanism, as described above. In this sub-layer, each token in the input sequence calculates attention scores with respect to all other tokens in the same input sequence.
* A simple, position-wise Feed-Forward Network (FFN), which will be detailed later.

##### The Decoder Stack

The decoder's role is to generate the output sequence, one token at a time, using the representation created by the encoder. It is also composed of a stack of **N=6** identical layers [26]. However, each decoder layer has three sub-layers:

* A **Masked** Multi-Head Self-Attention mechanism. This sub-layer is nearly identical to the one in the encoder, but with a crucial modification: it applies a "causal mask." During the calculation of attention scores, this mask sets the scores for all future tokens to negative infinity. After the softmax operation, this effectively makes their attention weights zero. This masking ensures that the prediction for a token at position *i* can only depend on the known outputs at positions less than *i*, preserving the autoregressive property required for sequential generation [45].
* A Multi-Head **Cross-Attention** mechanism. This is the vital layer that connects the decoder to the encoder. It functions similarly to self-attention, but its inputs come from two different sources. The Query (Q) vectors are generated from the output of the previous decoder layer (the masked self-attention sub-layer). The Key (K) and Value (V) vectors, however, are generated from the final output of the encoder stack. This allows every token in the decoder to attend to every token in the input sequence, enabling it to use the full context of the source sentence to inform its generation process [26].
* A position-wise Feed-Forward Network (FFN).

After the final decoder layer, the resulting vector is passed through a final linear layer and a softmax function. The linear layer projects the vector into a size equal to the vocabulary, and the softmax function converts these "logits" into a probability distribution over all possible next tokens [28].

This architectural separation between understanding (Encoder) and generating (Decoder) is a fundamental design choice with profound implications. The encoder's sole purpose is to build the richest possible representation of the input, using bidirectional attention where every token can see every other token. It is an "auto-encoding" model [45]. The decoder's job is to generate text autoregressively, using masked attention to only see the past. The cross-attention layer serves as the bridge, allowing the generative decoder to query the comprehensive understanding provided by the encoder. This conceptual split directly foreshadowed the development of specialized models. Researchers soon realized that for tasks requiring only understanding (e.g., sentiment classification), the encoder alone would suffice (leading to BERT). For tasks requiring only generation from a prompt (e.g., chatbots), the decoder alone was sufficient (leading to GPT). The full encoder-decoder architecture remains ideal for tasks that explicitly transform one sequence into another, like translation or summarization (leading to T5 and BART). The architecture itself thus defined the future taxonomy of foundation models.

#### 2.5 Stabilizing Deep Networks: Residual Connections and Layer Normalization

Training very deep neural networks is notoriously difficult due to issues like vanishing or exploding gradients. The Transformer architecture, with its stack of N layers, is no exception. It employs two critical techniques, borrowed from computer vision and adapted for NLP, to ensure stable training and effective gradient flow [27].

* **Residual Connections:** Each of the sub-layers (self-attention, cross-attention, and FFN) in both the encoder and decoder has a residual connection, also known as a "skip connection," around it. The input to the sub-layer (**x**) is added directly to the output of that sub-layer (**Sublayer(x)**). The result passed to the next stage is **x + Sublayer(x)** [16]. This simple addition creates a direct path for the gradient to flow through the network, bypassing the transformations within the sub-layer. This is vital for preventing the gradient signal from vanishing as it propagates backward through many layers, enabling the training of much deeper models [16].
* **Layer Normalization:** Immediately following each residual connection, a layer normalization step is applied: **LayerNorm(x + Sublayer(x))** [26]. Unlike batch normalization, which normalizes across the batch dimension, layer normalization normalizes the features for each individual training example. It computes the mean and variance used for normalization from all of the summed inputs to the neurons in a layer for a single training case. This helps to stabilize the training dynamics, reduce the model's sensitivity to weight initialization, and speed up convergence.

The combination of these two techniques is a masterstroke of engineering. The Transformer is not just a single invention but a synthesis of multiple ideas, each addressing a weakness introduced by another. The removal of recurrence necessitated positional encodings. The need to relate tokens in parallel led to self-attention. The simplicity of a single attention view led to multi-head attention. The depth of the stacked layers required residual connections and layer normalization for stability. It is a prime example of systems-level thinking in neural architecture design.

#### 2.6 The Final Transformation: The Position-wise Feed-Forward Network (FFN)

The final key component within each encoder and decoder layer is the position-wise Feed-Forward Network (FFN) [47].

* **Structure:** This is a relatively simple component, consisting of two linear transformations with a non-linear activation function in between. The most common activation is the Rectified Linear Unit (ReLU), though modern variants often use the Gaussian Error Linear Unit (GELU) [28]. The formula is:

    **FFN(x) = max(0, x * W1 + b1) * W2 + b2**

    The dimensionality of the input and output is **d_model**, while the inner layer is typically larger, for example, **d_ff = 2048** in the original paper.
* **Purpose:** This FFN is applied independently and identically to each position (each token's representation) in the sequence [27]. While the attention layers are responsible for mixing information and capturing dependencies between different tokens, the FFN's role is to perform a rich, non-linear transformation on each token's representation individually. It can be thought of as processing the contextually-aware information gathered by the attention mechanism and projecting it into a more complex and suitable representation for the next layer. The FFN significantly increases the model's capacity and is a major contributor to its total parameter count [23].

***

### Section 3: Architectural Specialization and the Rise of Foundational Models

The original Transformer, with its elegant encoder-decoder structure, was a versatile tool for sequence-to-sequence tasks. However, the true power of the architecture was fully unleashed when researchers began to deconstruct it, realizing that its constituent parts—the encoder and the decoder—were themselves powerful models in their own right. This led to the emergence of three distinct architectural families, each specialized for a different class of NLP problems. This specialization gave rise to the concept of "foundation models": large, pre-trained models that serve as a base for a wide range of downstream tasks.

#### 3.1 Encoder-Only Architectures (e.g., BERT): Models for Deep Understanding

The first major specialization involved isolating the encoder stack of the Transformer. This gave rise to models like Google's BERT (Bidirectional Encoder Representations from Transformers), which are designed for tasks that require a deep understanding of language rather than generation [28].

* **Architecture:** Encoder-only models consist solely of the Transformer's encoder stack [46]. Their defining architectural feature is the use of a fully **bidirectional** self-attention mechanism. In every layer, each token's representation is computed by attending to all other tokens in the sequence, both those that come before it and those that come after it [46]. This allows the model to build a rich, holistic, and deeply contextualized understanding of every word in the input text.
* **Pre-training Objective:** Since these models can see the entire sentence at once, they cannot be trained with a simple next-token prediction objective. Instead, BERT introduced two novel unsupervised pre-training tasks:
    * **Masked Language Modeling (MLM):** This is the core pre-training task for BERT. During training, a certain percentage (e.g., 15%) of the input tokens are randomly replaced with a special `<MASK>` token. The model's objective is then to predict the original identity of these masked tokens by leveraging the bidirectional context of the surrounding unmasked words [48]. This "fill-in-the-blanks" task forces the model to learn powerful representations of language that capture both local and global context.
    * **Next Sentence Prediction (NSP):** In this task, the model is presented with two sentences, A and B, and must predict whether sentence B is the actual sentence that follows A in the original corpus or if it is just a random sentence. The goal was to teach the model to understand relationships between sentences, which is important for tasks like question answering [51]. While integral to the original BERT, later models like RoBERTa demonstrated that this task was less impactful than MLM and could be removed without harming performance [49].
* **Use Cases:** Because they excel at producing high-quality contextual embeddings for input text, encoder-only models are the ideal choice for Natural Language Understanding (NLU) or analysis tasks. These are tasks where the model needs to comprehend the full text to make a prediction or extract information. They are not inherently generative. Common applications include:
    * **Text Classification:** Assigning a label to a piece of text (e.g., sentiment analysis, spam detection) [5].
    * **Named Entity Recognition (NER):** Identifying and classifying entities like people, organizations, and locations within text.
    * **Extractive Question Answering:** Given a question and a passage of text, extracting the span of text that contains the answer [48].

#### 3.2 Decoder-Only Architectures (e.g., GPT): Models for Fluent Generation

The second specialization involves using only the decoder stack of the Transformer. This family of models, pioneered by OpenAI's GPT (Generative Pre-trained Transformer) series, are masters of text generation and form the architectural basis for most of today's most famous LLMs [33].

* **Architecture:** Decoder-only models consist of a stack of Transformer decoder layers [51]. Their defining feature is the **causal** (or masked) self-attention mechanism. When processing a token at position *i*, the attention mechanism is masked to prevent it from attending to any tokens at positions *j > i* [45]. This enforces a strict left-to-right, unidirectional processing flow. This **autoregressive** property is what makes these models natural text generators: they generate output one token at a time, with each new token conditioned on the sequence of tokens generated before it.
* **Pre-training Objective:** The pre-training objective for decoder-only models is straightforward and classic: next-token prediction, also known as standard causal language modeling. Given a sequence of tokens from the training data, the model is trained to predict the very next token in that sequence [1]. This simple objective, when applied at a massive scale of data and parameters, proves to be incredibly effective at teaching the model grammar, facts, reasoning abilities, and style.
* **Use Cases:** These models are designed for Natural Language Generation (NLG). They are the go-to architecture for any task that requires creating new, coherent, and contextually relevant text. Most of the well-known large-scale generative models, including the GPT series (GPT-3, GPT-4), Llama, and Claude, are decoder-only architectures [5]. Their applications include:
    * **Chatbots and Dialogue Systems:** Engaging in open-ended conversations.
    * **Content Creation:** Writing articles, poems, emails, and marketing copy.
    * **Code Generation:** Writing functional code in various programming languages based on natural language descriptions.
    * **Summarization and Translation:** While also seq2seq tasks, large decoder-only models can perform these by being prompted with instructions like "Summarize the following text:..." [48].

The industry's overwhelming convergence on decoder-only architectures for large-scale, general-purpose assistants like ChatGPT is not solely due to their generative prowess. It is also heavily influenced by pragmatic engineering and computational efficiency, particularly in a conversational setting. An encoder-decoder model, when used for multi-turn chat, would be highly inefficient. For each new user turn, the entire conversation history would need to be re-processed by the encoder, as its bidirectional attention means every token's representation is dependent on every other token [53]. This is computationally expensive. In contrast, a decoder-only model is far more efficient. Due to its causal mask, the attention states (the Key-Value cache) for all previous tokens are fixed and do not depend on future tokens. When a new message is added, the model only needs to compute attention for the **new** tokens, while reusing the cached states from the past [53]. This makes inference for interactive applications significantly faster and cheaper. While encoder-based models have a theoretical advantage in deep understanding, empirical findings from scaling laws have shown that massive decoder-only models can develop surprisingly powerful contextual understanding capabilities on their own [54]. Thus, the dominance of this architecture is a story of economics and engineering as much as it is about pure architectural theory.

#### 3.3 Encoder-Decoder Architectures (e.g., T5, BART): Models for Transformation

The third family of models retains the full, original Transformer architecture, using both an encoder and a decoder stack. These models, such as Google's T5 (Text-to-Text Transfer Transformer) and Meta's BART, are explicitly designed for sequence-to-sequence (seq2seq) tasks that involve transforming an input sequence into a distinct output sequence [5].

* **Architecture:** These models leverage the strengths of both components. The bidirectional encoder processes the entire input sequence to create a rich, contextualized representation. The autoregressive decoder then uses this representation, accessed via the cross-attention mechanism, to conditionally generate the target output sequence [46].
* **Pre-training Objective (T5 Example):** T5 introduced a unified and elegant "text-to-text" framework, where every NLP task is reframed as a text-to-text problem. For example, for a classification task, the input is the text, and the output is the text of the class label. T5's pre-training objective is a form of span corruption or denoising. During training, random contiguous spans of tokens in the input text are corrupted (replaced by a single sentinel token, e.g., `<X>`). The model is then trained to reconstruct the original, uncorrupted text spans in the output, with different sentinel tokens (`<Y>`, `<Z>`) marking the different corrupted regions [51]. This objective encourages the model to be flexible and robust.
* **Use Cases:** Encoder-decoder models are the natural choice for tasks that have a clearly defined input sequence and a corresponding output sequence that needs to be generated. They excel at:
    * **Machine Translation:** Translating a sentence from a source language to a target language [51].
    * **Text Summarization:** Taking a long document as input and generating a concise summary as output.
    * **Generative Question Answering:** Where the input is a context and a question, and the output is a generated, natural language answer.

#### Architectural and Functional Comparison of Foundational Models

To crystallize the distinctions between these three architectural families, the following table provides an at-a-glance summary of their core features and ideal applications. This comparison is invaluable for practitioners seeking to select the most appropriate model architecture for a given task, as it directly connects architectural design choices to functional strengths.

| Feature | Encoder-Only (e.g., BERT) | Decoder-Only (e.g., GPT) | Encoder-Decoder (e.g., T5) |
| :--- | :--- | :--- | :--- |
| **Core Architecture** | Transformer Encoder Stack Only [46] | Transformer Decoder Stack Only [51] | Full Transformer (Encoder & Decoder) [46] |
| **Attention Mechanism** | Bidirectional Self-Attention [46] | Causal (Masked) Self-Attention [45] | Bidirectional in Encoder, Causal in Decoder, plus Cross-Attention [46] |
| **Pre-training Objective** | Masked Language Modeling (MLM) [51] | Next-Token Prediction (Causal LM) [33] | Denoising / Span Corruption [51] |
| **Primary Strength** | Deep Contextual Understanding [48] | Fluent and Coherent Text Generation [48] | Sequence-to-Sequence Transformation [46] |
| **Ideal Use Cases** | Classification, NER, Sentiment Analysis, Extractive Q&A [46] | Chatbots, Content Creation, Story Writing, Code Generation [48] | Translation, Summarization, Generative Q&A [51] |
| **Key Limitation** | Not inherently generative [48] | Less deep bidirectional context understanding [48] | More complex; potential bottleneck between encoder/decoder [53] |

***

### Section 4: Scaling and Aligning Modern LLMs

Possessing a sophisticated architecture is only the first step in creating a powerful Large Language Model. The processes of scaling these models to immense sizes and then carefully aligning their behavior with human expectations are just as crucial as the underlying design. This section shifts focus from the static blueprint of the architecture to the dynamic processes that transform these models from raw predictors into capable and useful AI systems.

#### 4.1 The Power of Scale: A Primer on Neural Scaling Laws

A pivotal moment in the development of LLMs was the empirical discovery of "scaling laws." These laws describe a predictable, power-law relationship between a model's performance (typically measured by its loss on a test set) and three key resources: the number of model parameters (N), the size of the training dataset (D), and the amount of computational power used for training (C) [57].

The core finding was that as these three factors are increased, the model's performance improves in a smooth and predictable way [57]. This discovery was transformative because it turned LLM development from a highly uncertain research endeavor into a more predictable engineering discipline. It provided a strong justification for the massive investments in compute and data required to build larger models, as organizations could be reasonably confident that a larger model trained on more data would be a better model [57].

A particularly influential refinement of these ideas came from DeepMind's "Chinchilla" scaling laws [58]. Their research suggested that for a fixed computational budget, the optimal way to scale is to increase the model size and the training dataset size in roughly equal proportion. This was a crucial insight, indicating that many earlier large models (like GPT-3) were actually "under-trained" for their size—they had too many parameters for the amount of data they were trained on. The Chinchilla findings suggested that future performance gains would come as much from scaling up datasets as from scaling up model parameters, guiding the development of more compute-optimal models [58].

The concept of scaling has continued to evolve. Researchers now also consider:

* **Post-training Scaling:** Performance improvements gained after the initial pre-training, through techniques like fine-tuning, distillation, or quantization [57].
* **Test-time Scaling:** Using more computational resources at the time of inference to improve the quality of a model's output, for example, by prompting the model to generate a "chain of thought" or by generating multiple responses and selecting the best one [57].

#### 4.2 From Prediction to Execution: The Role of Instruction Tuning

A base LLM, pre-trained on a massive corpus of text with a next-token prediction objective, is a powerful text completer. It is adept at recognizing patterns and continuing a sequence of text in a statistically probable way. However, it is not inherently an "instruction-following" agent. If given a prompt like "Translate 'hello' to French," it might continue with "and 'goodbye' to 'au revoir'" instead of simply providing the translation "bonjour." It completes the pattern rather than executing the command [60].

Instruction tuning is the process designed to bridge this gap. It is a form of supervised fine-tuning (SFT) where the pre-trained model is further trained on a curated dataset of high-quality examples. These examples are typically structured as instruction-response pairs, or sometimes as (instruction, input, output) triplets [61]. For example, a data point might be:

* **Instruction:** "Summarize the following article."
* **Input:** `<article text>`
* **Output:** [A well-written summary]

The goal of this fine-tuning stage is to teach the model the general skill of following instructions. By training on a diverse set of tasks and instructions, the model learns to become a more "helpful" assistant, capable of generalizing to follow instructions it has never explicitly seen during training—a capability known as zero-shot generalization [62]. This process effectively aligns the model's behavior with explicit human intentions and commands [60].

#### 4.3 Achieving Human Alignment: Reinforcement Learning from Human Feedback (RLHF)

Instruction tuning teaches a model how to follow a command, but it does not inherently teach it about the nuanced, often implicit, qualities that humans value in a response. These qualities can include politeness, truthfulness, harmlessness, appropriate tone, and the avoidance of subtle biases. It is difficult to capture these complex preferences in a static, supervised dataset. To address this deeper alignment challenge, researchers developed Reinforcement Learning from Human Feedback (RLHF) [62].

RLHF is a sophisticated, three-step process designed to fine-tune a model's behavior to align with human values and preferences [60].

1.  **Step 1: Supervised Fine-Tuning (SFT):** The process begins with a base LLM that has already been instruction-tuned. This SFT model serves as a strong starting point, as it already understands how to follow instructions. In the language of reinforcement learning, this model is the initial "policy" [62].
2.  **Step 2: Train a Reward Model:** This is the core of the "human feedback" part of RLHF.
    * A set of diverse prompts is selected. For each prompt, the SFT model is used to generate two or more different responses.
    * Human labelers are then shown the prompt and the generated responses and are asked to rank them from best to worst based on a set of criteria (e.g., helpfulness, harmlessness, factual accuracy) [62].
    * This creates a new dataset of human preference data: (prompt, chosen\_response, rejected\_response).
    * This preference dataset is used to train a separate model, known as the "reward model" (RM). The RM is typically another LLM, initialized from the SFT model, but with its final layer replaced to output a single scalar value. It is trained to take a prompt and a response as input and output a score that predicts how highly a human would rate that response. The model learns by trying to assign a higher score to the "chosen" responses than the "rejected" ones [61].
3.  **Step 3: Optimize the LLM with Reinforcement Learning:**
    * The SFT model (the policy) is now fine-tuned further using a reinforcement learning algorithm, most commonly Proximal Policy Optimization (PPO) [61].
    * In this RL training loop, the policy is given a random prompt from the dataset. It generates a response.
    * The response is then passed to the reward model from Step 2, which calculates a scalar "reward" score for it.
    * This reward signal is used to update the weights of the policy via the PPO algorithm. The objective of PPO is to adjust the policy to generate responses that maximize the reward score from the RM [61].
    * Crucially, the PPO objective function includes a penalty term (a Kullback-Leibler divergence term) that measures how much the updated policy has diverged from the original SFT policy from Step 1. This penalty is vital to prevent the model from "over-optimizing" for the reward model and generating text that is grammatically incorrect or nonsensical but happens to get a high reward. It keeps the model's outputs grounded in high-quality language [61].

The final model resulting from this iterative RLHF process is one that is not only capable of executing instructions but is also aligned to produce outputs that are helpful, harmless, and honest, as defined by the aggregated preferences of human labelers [64].

The progression from pre-training to instruction tuning to RLHF can be viewed as a deliberate, hierarchical alignment strategy. It is a process of systematically constraining the vast, unstructured potential of a base model into a form that is both useful and safe for human interaction. Pre-training on web-scale data provides the model with a comprehensive statistical model of language, facts, and, unfortunately, human biases [66]. It is a model of **what is written** in the world, not **what should be**. Instruction tuning provides the first layer of constraint, changing the model's objective from "continue this text" to "respond to this instruction," thereby aligning it with explicit human intent. Finally, RLHF provides the most sophisticated layer of constraint. It addresses the challenge that "goodness" is difficult to define in a supervised dataset. One cannot easily write down a single "correct" response that is also polite, unbiased, and appropriately nuanced for every situation. By learning from preferences ("this response is better than that one"), the reward model is able to capture these implicit, hard-to-define qualities [62]. This three-stage funnel starts with the universe of text, channels it through the structure of instructions, and polishes it with the fine-grained filter of human values.

***

### Section 5: The Frontier of LLM Architecture

While the Transformer architecture and its specialized variants have proven immensely powerful, they are not without limitations. As models have scaled into the hundreds of billions or even trillions of parameters, two fundamental bottlenecks have become increasingly prominent: the quadratic computational cost of the attention mechanism with respect to sequence length, and the inefficiency of activating every single model parameter for every token processed. This final technical section explores the cutting edge of architectural research, focusing on two major trends that seek to overcome these challenges and define the next generation of LLMs.

#### 5.1 Scaling Beyond Density: The Mixture of Experts (MoE) Architecture

As LLMs grow, the computational resources required for training and inference become staggering. A standard "dense" model, like GPT-3, must use every one of its parameters to process every single input token [68]. This monolithic approach is computationally intensive and inefficient, as not all parameters are relevant for all inputs.

The Mixture of Experts (MoE) architecture offers a solution through sparse activation or conditional computation [68]. The core idea is to replace certain dense layers within the Transformer—most commonly the Feed-Forward Networks (FFNs)—with a collection of parallel "expert" subnetworks [69]. The key innovation is a small, learnable "gating network," also known as a "router," which dynamically determines which one or few experts are best suited to process each individual token [69].

The MoE mechanism operates as follows:

1.  For each incoming token, the gating network computes a score or probability for each available expert, indicating that expert's suitability for the given token [70].
2.  Based on these scores, the gating network selects a small subset of experts to activate, typically using a top-k algorithm. For instance, the Mixtral 8x7B model uses 8 total experts but only activates the top 2 for any given token [69].
3.  The token is then processed only by these selected experts. The other experts remain dormant for that token, saving a vast amount of computation (FLOPs) [69].
4.  The outputs from the activated experts are then combined to produce the final output for that layer. This combination is often a weighted sum, where the weights are derived from the scores assigned by the gating network [72].

The primary advantage of this sparse architecture is that it allows for a dramatic increase in the total number of model parameters without a corresponding increase in computational cost [68]. A model can have a very large "capacity" to store knowledge within its numerous experts, but the active number of parameters used for any single forward pass remains small and manageable. For example, a model like Mixtral 8x7B has a total of approximately 46 billion parameters, but only uses about 12 billion active parameters per token. This gives it the performance and knowledge capacity of a much larger dense model, but with the inference speed and training efficiency of a smaller one [69].

However, this architecture introduces new engineering challenges. One is load balancing: the gating network must be trained with an auxiliary loss function to encourage it to distribute tokens roughly evenly among the experts, preventing a few "popular" experts from becoming a bottleneck [69]. Another challenge is memory (VRAM) requirements, as all experts must be loaded into the GPU's memory, even though only a fraction are used at any given time [71].

#### 5.2 An Alternative to Attention: State Space Models (Mamba)

The other major bottleneck of the standard Transformer is its self-attention mechanism. While highly effective at capturing context, its computational and memory complexity scales quadratically with the sequence length (O(N^2)) [73]. This makes it prohibitively expensive and slow for processing very long sequences, such as entire books, high-resolution audio waveforms, or genomic data [74].

A promising alternative has emerged from a class of models inspired by classical control theory: State Space Models (SSMs). These models are designed to process sequences with a complexity that scales linearly with sequence length (O(N)) [73]. SSMs operate by maintaining a compressed, latent "state" that evolves over time, conceptually similar to an RNN. However, they are formulated in a way that allows them to be trained with high degrees of parallelism, much like a CNN, thus combining the strengths of both recurrent and convolutional approaches [73].

The Mamba architecture represents a significant breakthrough in this area. Early SSMs like S4 were powerful but had a key weakness: their dynamics were static and did not change based on the input. This made it difficult for them to handle tasks requiring content-based reasoning. Mamba introduces a selective scan mechanism, which makes the core parameters of the SSM (represented by matrices A, B, and C in the state space equations) functions of the input data itself [74]. This crucial innovation allows the model to dynamically and selectively propagate or forget information based on the content of the current token. For example, it can learn to ignore filler words but remember and propagate key pieces of information over very long distances. This gives it the ability to perform the kind of content-aware reasoning that was previously considered a unique strength of attention-based models [74].

The Mamba architecture integrates these selective SSMs into a simple, unified block that replaces both the attention and MLP blocks of a standard Transformer. In empirical evaluations, Mamba has demonstrated performance that is competitive with or even superior to Transformer models of a similar size, particularly on tasks involving very long sequences. It achieves this with significantly faster inference (e.g., up to 5x higher throughput) and linear-time scaling, making it a powerful and efficient architectural path for future sequence models [74].

The emergence of MoE and Mamba, while architecturally very different, points to a shared future direction. Both are driven by the fundamental goal of breaking the monolithic, one-size-fits-all computation of the dense Transformer. They represent two distinct philosophical approaches to achieving greater efficiency and specialization. The dense Transformer is inefficient because it applies the same massive computational load to every token, regardless of its complexity or importance. MoE's approach is to achieve specialization through routing. It maintains the core Transformer structure but breaks up the FFN component into a team of specialists, with a router deciding which token goes to which specialist. It tackles the inefficiency of parameter activation. Mamba's approach is specialization through a more efficient form of recurrence. It replaces the core attention mechanism itself, which is the source of the quadratic bottleneck for long sequences. It tackles the inefficiency of algorithmic complexity. The success of both architectures suggests that the future of LLMs lies not just in scaling the current paradigm but in discovering more efficient and specialized computational primitives that move beyond the "one model for all tokens" approach.

***

### Conclusion: A Synthesized View and Future Horizons

The architectural evolution of Large Language Models represents one of the most rapid and impactful progressions in the history of artificial intelligence. In the span of just a few years, the field has moved from the sequential, memory-limited designs of Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTMs) to the massively parallel, deeply contextual Transformer architecture. This foundational design, in turn, was deconstructed and specialized, giving rise to the dominant families of modern LLMs: the understanding-focused encoder-only models like BERT, the generation-focused decoder-only models like GPT, and the transformation-focused encoder-decoder models like T5. The subsequent development of scaling laws and sophisticated alignment techniques like Instruction Tuning and RLHF transformed these powerful architectures into capable, helpful, and increasingly safe AI systems.

Today, we stand at another inflection point, as the frontier of research pushes beyond the dense Transformer to address its inherent inefficiencies. Architectures like Mixture of Experts (MoE) tackle the cost of dense parameter activation by introducing conditional computation, while State Space Models like Mamba address the quadratic complexity of attention with a return to a more efficient, linear-time recurrent formulation. These innovations signal a move towards more specialized, efficient, and scalable computational primitives.

Looking forward, several key challenges and research directions will shape the next generation of LLM architectures [78].

* **Multimodality:** The future is undeniably multimodal. Architectures that can seamlessly process, integrate, and reason across diverse data types—including text, images, audio, and video—are a primary focus. This will require new methods for creating shared representation spaces and more complex attention or fusion mechanisms [67].
* **Reasoning and World Models:** A significant challenge for current LLMs is their tendency to "hallucinate" and their lack of true causal reasoning [80]. A major research thrust is to move beyond statistical pattern matching towards architectures that can build and maintain internal, structured "world models." Such models would be able to simulate events, understand causality, and perform more robust, grounded reasoning, representing a leap from token-level prediction to object-level understanding [57].
* **Efficiency and Deployment:** The immense size of state-of-the-art LLMs makes them difficult and expensive to deploy, especially in resource-constrained environments like edge devices. This will continue to drive intense research into architectural efficiency, including model compression, quantization, pruning, and the development of entirely new, hardware-aware architectures designed for low-latency and low-power inference [83].
* **Trust, Transparency, and Safety:** As LLMs become more integrated into society, addressing their "black box" nature is paramount. The challenges of ensuring fairness, mitigating bias, preventing malicious use, and providing explainability for model decisions are not just training problems but also architectural ones [67]. Future designs may need to incorporate structures that are inherently more transparent, verifiable, and controllable.

The trajectory of LLM architecture suggests a convergence of ideas. The next great architectural paradigm will likely not be a single, monolithic solution but a hybrid system that successfully unifies the parallel processing power and contextual richness of attention, the linear-time efficiency and recurrent memory of models like Mamba, the sparse, conditional computation of MoE, and the grounded, structured understanding of emerging world models. The quest continues for an architecture that can learn not just the patterns in language, but the underlying structure of the world it describes.

***

### Works Cited

[1] "What Are Large Language Models (LLMs)? - IBM." Accessed August 6, 2025. [https://www.ibm.com/think/topics/large-language-models](https://www.ibm.com/think/topics/large-language-models).
[2] "What is an LLM (large language model)? - Cloudflare." Accessed August 6, 2025. [https://www.cloudflare.com/learning/ai/what-is-large-language-model/](https://www.cloudflare.com/learning/ai/what-is-large-language-model/).
[3] "[www.sap.com](https://www.sap.com)." Accessed August 6, 2025. [https://www.sap.com/resources/what-is-large-language-model#:~:text=LLM%20means%20large%20language%20model,manner%3B%20and%20identifying%20data%20patterns](https://www.sap.com/resources/what-is-large-language-model#:~:text=LLM%20means%20large%20language%20model,manner%3B%20and%20identifying%20data%20patterns).
[4] "What is a large language model (LLM)? | SAP." Accessed August 6, 2025. [https://www.sap.com/resources/what-is-large-language-model](https://www.sap.com/resources/what-is-large-language-model).
[5] "What is LLM? - Large Language Models Explained - AWS." Accessed August 6, 2025. [https://aws.amazon.com/what-is/large-language-model/](https://aws.amazon.com/what-is/large-language-model/).
[6] "Understanding large language models: A comprehensive guide - Elastic." Accessed August 6, 2025. [https://www.elastic.co/what-is/large-language-models](https://www.elastic.co/what-is/large-language-models).
[7] "Recurrent neural network - Wikipedia." Accessed August 6, 2025. [https://en.wikipedia.org/wiki/Recurrent_neural_network](https://en.wikipedia.org/wiki/Recurrent_neural_network).
[8] "What is a Recurrent Neural Network (RNN)? - IBM." Accessed August 6, 2025. [https://www.ibm.com/think/topics/recurrent-neural-networks](https://www.ibm.com/think/topics/recurrent-neural-networks).
[9] "What is Recurrent Neural Networks (RNN)? - Analytics Vidhya." Accessed August 6, 2025. [https://www.analyticsvidhya.com/blog/2022/03/a-brief-overview-of-recurrent-neural-networks-rnn/](https://www.analyticsvidhya.com/blog/2022/03/a-brief-overview-of-recurrent-neural-networks-rnn/).
[10] "Introduction to Recurrent Neural Networks - GeeksforGeeks." Accessed August 6, 2025. [https://www.geeksforgeeks.org/machine-learning/introduction-to-recurrent-neural-network/](https://www.geeksforgeeks.org/machine-learning/introduction-to-recurrent-neural-network/).
[11] "What is RNN (Recurrent Neural Network)? - AWS." Accessed August 6, 2025. [https://aws.amazon.com/what-is/recurrent-neural-network/](https://aws.amazon.com/what-is/recurrent-neural-network/).
[12] "The Limitations of Recurrent Neural Networks (RNNs) and Why They Matter - Medium." Accessed August 6, 2025. [https://medium.com/@yonasdesta2012/the-limitations-of-recurrent-neural-networks-rnns-and-why-they-matter-eb0a05c90b60](https://medium.com/@yonasdesta2012/the-limitations-of-recurrent-neural-networks-rnns-and-why-they-matter-eb0a05c90b60).
[13] "A Gentle Introduction to Long Short-Term Memory Networks by the Experts - MachineLearningMastery.com." Accessed August 6, 2025. [https://machinelearningmastery.com/gentle-introduction-long-short-term-memory-networks-experts/](https://machinelearningmastery.com/gentle-introduction-long-short-term-memory-networks-experts/).
[14] "Prevent the Vanishing Gradient Problem with LSTM | Baeldung on Computer Science." Accessed August 6, 2025. [https://www.baeldung.com/cs/lstm-vanishing-gradient-prevention](https://www.baeldung.com/cs/lstm-vanishing-gradient-prevention).
[15] "Vanishing Gradient Problem : Everything you need to know - Engati." Accessed August 6, 2025. [https://www.engati.com/glossary/vanishing-gradient-problem](https://www.engati.com/glossary/vanishing-gradient-problem).
[16] "Vanishing gradient problem - Wikipedia." Accessed August 6, 2025. [https://en.wikipedia.org/wiki/Vanishing_gradient_problem](https://en.wikipedia.org/wiki/Vanishing_gradient_problem).
[17] "What is LSTM - Long Short Term Memory? - GeeksforGeeks." Accessed August 6, 2025. [https://www.geeksforgeeks.org/deep-learning/deep-learning-introduction-to-long-short-term-memory/](https://www.geeksforgeeks.org/deep-learning/deep-learning-introduction-to-long-short-term-memory/).
[18] "Long Short-Term Memory (LSTM) - NVIDIA Developer." Accessed August 6, 2025. [https://developer.nvidia.com/discover/lstm](https://developer.nvidia.com/discover/lstm).
[19] "What is LSTM? Introduction to Long Short-Term Memory - Analytics Vidhya." Accessed August 6, 2025. [https://www.analyticsvidhya.com/blog/2021/03/introduction-to-long-short-term-memory-lstm/](https://www.analyticsvidhya.com/blog/2021/03/introduction-to-long-short-term-memory-lstm/).
[20] "Long short-term memory - Wikipedia." Accessed August 6, 2025. [https://en.wikipedia.org/wiki/Long_short-term_memory](https://en.wikipedia.org/wiki/Long_short-term_memory).
[21] "What Is Long Short-Term Memory (LSTM)? - MATLAB & Simulink - MathWorks." Accessed August 6, 2025. [https://www.mathworks.com/discovery/lstm.html](https://www.mathworks.com/discovery/lstm.html).
[22] "Attention Is All You Need - Wikipedia." Accessed August 6, 2025. [https://en.wikipedia.org/wiki/Attention_Is_All_You_Need](https://en.wikipedia.org/wiki/Attention_Is_All_You_Need).
[23] "'Attention is All You Need' Summary - Medium." Accessed August 6, 2025. [https://medium.com/@dminhk/attention-is-all-you-need-summary-6f0437e63a91](https://medium.com/@dminhk/attention-is-all-you-need-summary-6f0437e63a91).
[24] "Attention is All you Need - NIPS." Accessed August 6, 2025. [https://papers.nips.cc/paper/7181-attention-is-all-you-need](https://papers.nips.cc/paper/7181-attention-is-all-you-need).
[25] "Attention Is All You Need | Request PDF - ResearchGate." Accessed August 6, 2025. [https://www.researchgate.net/publication/317558625_Attention_Is_All_You_Need](https://www.researchgate.net/publication/317558625_Attention_Is_All_You_Need).
[26] "Attention is All you Need - NIPS." Accessed August 6, 2025. [https://papers.neurips.cc/paper/7181-attention-is-all-you-need.pdf](https://papers.neurips.cc/paper/7181-attention-is-all-you-need.pdf).
[27] "Understanding Transformers & the Architecture of LLMs - MLQ.ai." Accessed August 6, 2025. [https://blog.mlq.ai/llm-transformer-architecture/](https://blog.mlq.ai/llm-transformer-architecture/).
[28] "Transformer (deep learning architecture) - Wikipedia." Accessed August 6, 2025. [https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture](https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)).
[29] "Part 2 : Transformers: Input Embedding and Positional Encoding | by Kalpa Subbaiah." Accessed August 6, 2025. [https://kalpa-subbaiah.medium.com/part-2-transformers-input-embedding-and-positional-encoding-55fe0d3d8681](https://kalpa-subbaiah.medium.com/part-2-transformers-input-embedding-and-positional-encoding-55fe0d3d8681).
[30] "A Gentle Introduction to Positional Encoding in Transformer Models, Part 1 - MachineLearningMastery.com." Accessed August 6, 2025. [https://machinelearningmastery.com/a-gentle-introduction-to-positional-encoding-in-transformer-models-part-1/](https://machinelearningmastery.com/a-gentle-introduction-to-positional-encoding-in-transformer-models-part-1/).
[31] "What is a Transformer Model? - IBM." Accessed August 6, 2025. [https://www.ibm.com/think/topics/transformer-model](https://www.ibm.com/think/topics/transformer-model).
[32] "Input Embeddings & Positional Encoding: The Forgotten Foundations of Transformers | by Sanan Ahmadov | Generative AI." Accessed August 6, 2025. [https://generativeai.pub/input-embeddings-positional-encoding-the-forgotten-foundations-of-transformers-e44e3f671f62](https://generativeai.pub/input-embeddings-positional-encoding-the-forgotten-foundations-of-transformers-e44e3f671f62).
[33] "An intuitive overview of the Transformer architecture | by Roberto Infante | Medium." Accessed August 6, 2025. [https://medium.com/@roberto.g.infante/an-intuitive-overview-of-the-transformer-architecture-6a88ccc88171](https://medium.com/@roberto.g.infante/an-intuitive-overview-of-the-transformer-architecture-6a88ccc88171).
[34] "How Transformers Work: A Detailed Exploration of Transformer Architecture - DataCamp." Accessed August 6, 2025. [https://www.datacamp.com/tutorial/how-transformers-work](https://www.datacamp.com/tutorial/how-transformers-work).
[35] "Positional Encoding in Transformers - GeeksforGeeks." Accessed August 6, 2025. [https://www.geeksforgeeks.org/nlp/positional-encoding-in-transformers/](https://www.geeksforgeeks.org/nlp/positional-encoding-in-transformers/).
[36] "Positional Embeddings in Transformer Models: Evolution from Text to Vision Domains | ICLR Blogposts 2025 - Cloudfront.net." Accessed August 6, 2025. [https://d2jud02ci9yv69.cloudfront.net/2025-04-28-positional-embedding-19/blog/positional-embedding/](https://d2jud02ci9yv69.cloudfront.net/2025-04-28-positional-embedding-19/blog/positional-embedding/).
[37] "Transformer — Attention Is All You Need Easily Explained With Illustrations | by Luv Bansal." Accessed August 6, 2025. [https://luv-bansal.medium.com/transformer-attention-is-all-you-need-easily-explained-with-illustrations-d38fdb06d7db](https://luv-bansal.medium.com/transformer-attention-is-all-you-need-easily-explained-with-illustrations-d38fdb06d7db).
[38] "Transformer Attention: A Guide to the Q, K, and V Matrices - billparker.ai." Accessed August 6, 2025. [https://www.billparker.ai/2024/10/transformer-attention-simple-guide-to-q.html](https://www.billparker.ai/2024/10/transformer-attention-simple-guide-to-q.html).
[39] "Query, Key and Value in Self-attention | by Opeyemi Osakuade | Medium." Accessed August 6, 2025. [https://osakuadeopeyemi.medium.com/query-key-and-value-in-self-attention-34bbe6fabc75](https://osakuadeopeyemi.medium.com/query-key-and-value-in-self-attention-34bbe6fabc75).
[40] "Multi-Head Attention Mechanism - GeeksforGeeks." Accessed August 6, 2025. [https://www.geeksforgeeks.org/nlp/multi-head-attention-mechanism/](https://www.geeksforgeeks.org/nlp/multi-head-attention-mechanism/).
[41] "Multi-Head Attention and Transformer Architecture - Pathway." Accessed August 6, 2025. [https://pathway.com/bootcamps/rag-and-llms/coursework/module-2-word-vectors-simplified/bonus-overview-of-the-transformer-architecture/multi-head-attention-and-transformer-architecture/](https://pathway.com/bootcamps/rag-and-llms/coursework/module-2-word-vectors-simplified/bonus-overview-of-the-transformer-architecture/multi-head-attention-and-transformer-architecture/).
[42] "Exploring Multi-Head Attention: Why More Heads Are Better Than One | by Hassaan Idrees." Accessed August 6, 2025. [https://medium.com/@hassaanidrees7/exploring-multi-head-attention-why-more-heads-are-better-than-one-006a5823372b](https://medium.com/@hassaanidrees7/exploring-multi-head-attention-why-more-heads-are-better-than-one-006a5823372b).
[43] "Understanding Multi Head Attention in Transformers | by Sachinsoni - Medium." Accessed August 6, 2025. [https://medium.com/@sachinsoni600517/multi-head-attention-in-transformers-1dd087e05d41](https://medium.com/@sachinsoni600517/multi-head-attention-in-transformers-1dd087e05d41).
[44] "Understanding Transformer model architectures - Practical Artificial Intelligence." Accessed August 6, 2025. [https://www.practicalai.io/understanding-transformer-model-architectures/](https://www.practicalai.io/understanding-transformer-model-architectures/).
[45] "Why do different architectures only need an encoder/decoder or need both? - Reddit." Accessed August 6, 2025. [https://www.reddit.com/r/learnmachinelearning/comments/1g7plvb/why_do_different_architectures_only_need_an/](https://www.reddit.com/r/learnmachinelearning/comments/1g7plvb/why_do_different_architectures_only_need_an/).
[46] "Transformer Architectures - Hugging Face LLM Course." Accessed August 6, 2025. [https://huggingface.co/learn/llm-course/chapter1/6](https://huggingface.co/learn/llm-course/chapter1/6).
[47] "machinelearning.apple.com." Accessed August 6, 2025. [https://machinelearning.apple.com/research/one-wide-ffn#:~:text=The%20Transformer%20architecture%20has%20two,transforms%20each%20input%20token%20independently](https://machinelearning.apple.com/research/one-wide-ffn#:~:text=The%20Transformer%20architecture%20has%20two,transforms%20each%20input%20token%20independently).
[48] "GPT vs BERT: Which Model Fits Your Use Case?" Accessed August 6, 2025. [https://dataengineeracademy.com/blog/gpt-vs-bert-which-model-fits-your-use-case/](https://dataengineeracademy.com/blog/gpt-vs-bert-which-model-fits-your-use-case/).
[49] "Transformer, GPT-3,GPT-J, T5 and BERT. | by Ali Issa - Medium." Accessed August 6, 2025. [https://aliissa99.medium.com/transformer-gpt-3-gpt-j-t5-and-bert-4cf8915dd86f](https://aliissa99.medium.com/transformer-gpt-3-gpt-j-t5-and-bert-4cf8915dd86f).
[50] "How does OpenAI compare to other models like BERT and T5? - Milvus." Accessed August 6, 2025. [https://milvus.io/ai-quick-reference/how-does-openai-compare-to-other-models-like-bert-and-t5](https://milvus.io/ai-quick-reference/how-does-openai-compare-to-other-models-like-bert-and-t5).
[51] "Comparing Large Language Models : GPT vs. BERT vs. T5 - Generative_AI." Accessed August 6, 2025. [https://automotivevisions.wordpress.com/2025/03/21/comparing-large-language-models-gpt-vs-bert-vs-t5/](https://automotivevisions.wordpress.com/2025/03/21/comparing-large-language-models-gpt-vs-bert-vs-t5/).
[52] "Pre-trained transformer models: BERT, GPT, and T5 | Deep Learning Systems Class Notes." Accessed August 6, 2025. [https://library.fiveable.me/deep-learning-systems/unit-10/pre-trained-transformer-models-bert-gpt-t5/study-guide/o8JLDj9oFwOSdcRt](https://library.fiveable.me/deep-learning-systems/unit-10/pre-trained-transformer-models-bert-gpt-t5/study-guide/o8JLDj9oFwOSdcRt).
[53] "Encoder-Decoder vs. Decoder-Only. What is the difference between an… | by Minki Jung | Medium." Accessed August 6, 2025. [https://medium.com/@qmsoqm2/auto-regressive-vs-sequence-to-sequence-d7362eda001e](https://medium.com/@qmsoqm2/auto-regressive-vs-sequence-to-sequence-d7362eda001e).
[54] "Encoder-Decoder Transformers vs Decoder-Only vs Encoder-Only: Pros and Cons - YouTube." Accessed August 6, 2025. [https://www.youtube.com/watch?v=MC3qSrsfWRs](https://www.youtube.com/watch?v=MC3qSrsfWRs).
[55] "11.9. Large-Scale Pretraining with Transformers — Dive into Deep ...." Accessed August 6, 2025. [https://d2l.ai/chapter_attention-mechanisms-and-transformers/large-pretraining-transformers.html](https://d2l.ai/chapter_attention-mechanisms-and-transformers/large-pretraining-transformers.html).
[56] "GPT Vs. BERT: A Technical Deep Dive | Al Rafay Global." Accessed August 6, 2025. [https://alrafayglobal.com/gpt-vs-bert/](https://alrafayglobal.com/gpt-vs-bert/).
[57] "How Scaling Laws Drive Smarter, More Powerful AI | NVIDIA Blog." Accessed August 6, 2025. [https://blogs.nvidia.com/blog/ai-scaling-laws/](https://blogs.nvidia.com/blog/ai-scaling-laws/).
[58] "Neural scaling law - Wikipedia." Accessed August 6, 2025. [https://en.wikipedia.org/wiki/Neural_scaling_law](https://en.wikipedia.org/wiki/Neural_scaling_law).
[59] "Demystifying Scaling Laws in Large Language Models | by Dagang Wei - Medium." Accessed August 6, 2025. [https://medium.com/@weidagang/demystifying-scaling-laws-in-large-language-models-14caf8ac6f80](https://medium.com/@weidagang/demystifying-scaling-laws-in-large-language-models-14caf8ac6f80).
[60] "www.ionio.ai." Accessed August 6, 2025. [https://www.ionio.ai/blog/a-comprehensive-guide-to-fine-tuning-llms-using-rlhf-part-1#:~:text=The%20goal%20of%20instruction%20fine,values%20and%20preferences%20of%20individuals](https://www.ionio.ai/blog/a-comprehensive-guide-to-fine-tuning-llms-using-rlhf-part-1#:~:text=The%20goal%20of%20instruction%20fine,values%20and%20preferences%20of%20individuals).
[61] "A Comprehensive Guide to fine-tuning LLMs using RLHF (Part-1) - Ionio." Accessed August 6, 2025. [https://www.ionio.ai/blog/a-comprehensive-guide-to-fine-tuning-llms-using-rlhf-part-1](https://www.ionio.ai/blog/a-comprehensive-guide-to-fine-tuning-llms-using-rlhf-part-1).
[62] "Instruction Tuning + RLHF: Teaching LLMs to Follow and Align | by ...." Accessed August 6, 2025. [https://medium.com/@akankshasinha247/instruction-tuning-rlhf-teaching-llms-to-follow-and-align-611a5462b1bf](https://medium.com/@akankshasinha247/instruction-tuning-rlhf-teaching-llms-to-follow-and-align-611a5462b1bf).
[63] "Fine-Tuning Language Models with Reward Learning on Policy - arXiv." Accessed August 6, 2025. [https://arxiv.org/html/2403.19279v1](https://arxiv.org/html/2403.19279v1).
[64] "How RLHF, RAG and Instruction Fine-Tuning Shape the Future | GigaSpaces AI." Accessed August 6, 2025. [https://www.gigaspaces.com/blog/rlhf-rag-and-instruction-fine-tuning](https://www.gigaspaces.com/blog/rlhf-rag-and-instruction-fine-tuning).
[65] "Reinforcement learning with human feedback (RLHF) for LLMs - SuperAnnotate." Accessed August 6, 2025. [https://www.superannotate.com/blog/rlhf-for-llm](https://www.superannotate.com/blog/rlhf-for-llm).
[66] "Scaling Laws for Fact Memorization of Large Language Models - ACL Anthology." Accessed August 6, 2025. [https://aclanthology.org/2024.findings-emnlp.658/](https://aclanthology.org/2024.findings-emnlp.658/).
[67] "The Future of Large Language Models in 2025 - Research AIMultiple." Accessed August 6, 2025. [https://research.aimultiple.com/future-of-large-language-models/](https://research.aimultiple.com/future-of-large-language-models/).
[68] "What is mixture of experts? | IBM." Accessed August 6, 2025. [https://www.ibm.com/think/topics/mixture-of-experts](https://www.ibm.com/think/topics/mixture-of-experts).
[69] "Applying Mixture of Experts in LLM Architectures | NVIDIA Technical ...." Accessed August 6, 2025. [https://developer.nvidia.com/blog/applying-mixture-of-experts-in-llm-architectures/](https://developer.nvidia.com/blog/applying-mixture-of-experts-in-llm-architectures/).
[70] "Mixture of Experts LLMs: Key Concepts Explained - neptune.ai." Accessed August 6, 2025. [https://neptune.ai/blog/mixture-of-experts-llms](https://neptune.ai/blog/mixture-of-experts-llms).
[71] "LLM Mixture of Experts Explained - TensorOps." Accessed August 6, 2025. [https://www.tensorops.ai/post/what-is-mixture-of-experts-llm](https://www.tensorops.ai/post/what-is-mixture-of-experts-llm).
[72] "What Is Mixture of Experts (MoE)? How It Works, Use Cases & More | DataCamp." Accessed August 6, 2025. [https://www.datacamp.com/blog/mixture-of-experts-moe](https://www.datacamp.com/blog/mixture-of-experts-moe).
[73] "MAMBA and State Space Models Explained | by Astarag Mohapatra - Medium." Accessed August 6, 2025. [https://athekunal.medium.com/mamba-and-state-space-models-explained-b1bf3cb3bb77](https://athekunal.medium.com/mamba-and-state-space-models-explained-b1bf3cb3bb77).
[74] "Mamba: Linear-Time Sequence Modeling with Selective State Spaces - OpenReview." Accessed August 6, 2025. [https://openreview.net/forum?id=tEYskw1VY2](https://openreview.net/forum?id=tEYskw1VY2).
[75] "[2312.00752] Mamba: Linear-Time Sequence Modeling with Selective State Spaces - arXiv." Accessed August 6, 2025. [https://arxiv.org/abs/2312.00752](https://arxiv.org/abs/2312.00752).
[76] "Mamba: Linear-Time Sequence Modeling with Selective State ...." Accessed August 6, 2025. [https://www.oxen.ai/blog/mamba-linear-time-sequence-modeling-with-selective-state-spaces-arxiv-dives](https://www.oxen.ai/blog/mamba-linear-time-sequence-modeling-with-selective-state-spaces-arxiv-dives).
[77] "state-spaces/mamba: Mamba SSM architecture - GitHub." Accessed August 6, 2025. [https://github.com/state-spaces/mamba](https://github.com/state-spaces/mamba).
[78] "(PDF) A Survey of Large Language Models: Foundations and Future ...." Accessed August 6, 2025. [https://www.researchgate.net/publication/394108041_A_Survey_of_Large_Language_Models_Foundations_and_Future_Directions](https://www.researchgate.net/publication/394108041_A_Survey_of_Large_Language_Models_Foundations_and_Future_Directions).
[79] "Large Language Models: A Comprehensive Survey on Architectures, Applications, and Challenges - Zenodo." Accessed August 6, 2025. [https://zenodo.org/records/14161613](https://zenodo.org/records/14161613).
[80] "A Review of Large Language Models: Fundamental Architectures ...." Accessed August 6, 2025. [https://www.mdpi.com/2079-9292/13/24/5040](https://www.mdpi.com/2079-9292/13/24/5040).
[81] "[2303.18223] A Survey of Large Language Models - arXiv." Accessed August 6, 2025. [https://arxiv.org/abs/2303.18223](https://arxiv.org/abs/2303.18223).
[82] "[2412.03220] Survey of different Large Language Model Architectures: Trends, Benchmarks, and Challenges - arXiv." Accessed August 6, 2025. [https://arxiv.org/abs/2412.03220](https://arxiv.org/abs/2412.03220).
[83] "Intelligent data analysis in edge computing with large ... - Frontiers." Accessed August 6, 2025. [https://www.frontiersin.org/journals/computer-science/articles/10.3389/fcomp.2025.1538277/full](https://www.frontiersin.org/journals/computer-science/articles/10.3389/fcomp.2025.1538277/full).
[84] "AI Transparency in the Age of LLMs: A Human-Centered Research Roadmap." Accessed August 6, 2025. [https://hdsr.mitpress.mit.edu/pub/aelql9qy](https://hdsr.mitpress.mit.edu/pub/aelql9qy).