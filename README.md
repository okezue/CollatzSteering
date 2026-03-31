# Steerable Control-Flow in Collatz Transformers

This project builds on the findings of Charton and Narayanan (2025), ["Transformers know more than they can tell: Learning the Collatz sequence"](https://arxiv.org/abs/2511.10811), which showed that transformers trained on the long Collatz step learn the task by mastering discrete algorithmic classes one at a time, and that their errors are structured rather than random. We extend their work from behavioral observation to causal, representation-level analysis: we locate where the model encodes its internal "loop counters," show this information is present before the model can use it, and demonstrate that editing these representations steers outputs in predictable ways.

## Background: The Long Collatz Step

The Collatz sequence starts from a positive integer $n$ and repeatedly applies:

$$c_{i+1} = \begin{cases} c_i / 2 & \text{if } c_i \text{ is even} \\ (3c_i + 1)/2 & \text{if } c_i \text{ is odd} \end{cases}$$

The **long Collatz step** collapses an entire run of odd-then-even steps into a single function. For any odd $n$, write $n = 2^k m - 1$ where $m$ is odd. Then $k$ counts the trailing 1-bits of $n$ in binary. After $k$ applications of $n \to (3n+1)/2$ (up-steps), we reach the **apex** $a = 3^k m - 1$, which is even. Let $k'= v_2(a)$ be the number of trailing zeros of $a$. After $k'$ divisions by 2 (down-steps), we arrive at the next odd number:

$$\kappa(n) = \frac{(3/2)^k(n+1) - 1}{2^{k'}}$$

The key insight from the original paper is that both $k$ and $k'$ can be determined from the binary suffix of $n$. This means the entire computation is governed by two control variables that are readable from the input's bit pattern.

## What the Original Paper Found

Charton and Narayanan trained sequence-to-sequence transformers (4-layer bidirectional encoder, 1-layer autoregressive decoder, dimension 512, 8 attention heads) to predict $\kappa(n)$ from $n$, with both numbers encoded as digit sequences in base $B$. They trained 56 models across bases 2 through 57 on roughly 300 million random pairs $(n, \kappa(n))$ with $n$ drawn uniformly from odd integers up to $10^{12}$.

Their central findings:

**Quantized learning.** Accuracy does not improve smoothly. Instead it jumps in discrete steps, with each jump corresponding to the model mastering a new residual class modulo $2^p$, which corresponds to a new pair of loop lengths $(k, k')$. Classes are learned in order of increasing $k + k'$, starting from $(1,1)$. The same learning order appears across all bases.

**Structured errors.** When the model is wrong, it is wrong in predictable ways. For roughly 70% of errors in odd-base models, the output is exactly $2^l \cdot \kappa(n)$ for some small integer $l$, meaning the model computed the right arithmetic but underestimated $k'$. For inputs with $k$ beyond the model's learned range, the output follows $r \approx (2/3)^a \cdot 2^l$, corresponding to an underestimate of $k$. The model defaults to the largest loop lengths it has learned so far.

**Arithmetic is easy, control flow is hard.** The "data-path" computation (multiplying, dividing) is largely correct even in wrong predictions. The bottleneck is inferring the correct loop lengths from the input encoding. This is what makes the model "know more than it can tell": it has internalized the arithmetic but cannot always read the control parameters that gate it.

These observations motivate a natural question: if errors arise from wrong loop lengths, is there a localized internal state encoding those loop lengths, and can we find and edit it?

## This Study

We test a specific hypothesis: the model maintains a low-dimensional, approximately linear subspace in its encoder activations that encodes $k$ and $k'$, and this subspace is causally linked to the output via the multiplicative structure of $\kappa$.

### 1. Progress Measure via Probing

At periodic checkpoints during training, we freeze the model and train linear probes on the encoder's residual stream (at each layer) to predict $k(n)$ and $k'(n)$. We also train small MLP probes as an upper bound.

The prediction is that probe accuracy for $k$ and $k'$ will rise **smoothly and before** the step-like jumps in end-to-end accuracy. If confirmed, this means the model internally represents loop-length information before it can use that information in decoding. This is analogous to progress measures studied in the context of grokking (Neel Nanda et al., 2023), but applied to control-structure inference rather than modular arithmetic.

### 2. Steering Vectors

Using contrastive activation differences, we construct steering directions $v_k$ and $v_{k'}$ for each encoder layer. For $v_{k'}$, we collect mean-pooled encoder activations grouped by $k'$ value and compute the average difference between adjacent groups. Similarly for $v_k$.

During inference, we add $\alpha \cdot v_{k'}$ to the residual stream at a chosen layer and decode normally. If the loop-length representation is truly linear and causally efficacious, this should produce predictable multiplicative changes in the output:

- Shifting along $v_{k'}$ should change outputs by factors close to $2^{\Delta}$, matching the power-of-two error signature from the original paper.
- Shifting along $v_k$ should change outputs by factors close to $(3/2)^{\Delta}$, matching the hard-error signature.

We compare against random-direction baselines to rule out generic perturbation effects.

### 3. Cross-Layer Transcoder

Following the CLT-Forge framework for mechanistic interpretability, we train a cross-layer transcoder (sparse autoencoder variant with JumpReLU activations) on encoder layer activations. This decomposes the model's computation into sparse, interpretable features. We then correlate individual features with $k$ and $k'$ to test whether the controller and executor decompose into separate feature circuits.

## Architecture

We reproduce the original paper's architecture exactly:

| Component | Specification |
|-----------|--------------|
| Encoder | 4 layers, bidirectional, dim 512, 8 heads |
| Decoder | 1 layer, autoregressive, dim 512, 8 heads |
| FFN | 2048 hidden dim |
| Optimizer | Adam, lr = $3 \times 10^{-5}$ |
| Training | ~300M random pairs per base |
| Evaluation | 100k random odd $n \in [1, 10^{12}]$, exact-match accuracy |
| Bases | 32 (best even), 24 (good composite), 11 (poor odd) |

## Running

```bash
pip install torch numpy matplotlib tqdm

# Train a model
python run.py train --base 32 --dev cuda

# Analyze errors
python run.py eval --base 32 --dev cuda

# Run probe study across checkpoints
python run.py probe --base 32 --dev cuda

# Steering experiments
python run.py steer --base 32 --dev cuda --ckpt output/b32/best.pt

# Train cross-layer transcoder
python run.py transcoder --base 32 --dev cuda --ckpt output/b32/best.pt

# Generate all plots
python run.py plots --base 32
```

Training all three bases takes roughly 2 days on a single A10G GPU.

## Curious about if...

1. **Latent control-flow precedes behavioral competence.** If $k$ and $k'$ become linearly decodable from the residual stream before the model's accuracy jumps, this gives a concrete progress measure for Collatz learning and demonstrates that internal representations lead external behavior.

2. **Causal steerability of loop lengths.** If activation steering changes outputs by factors of $2^{\Delta}$ (for $k'$ edits) and $(3/2)^{\Delta}$ (for $k$ edits), and corrects a meaningful fraction of errors without retraining, this establishes that loop-length information is not just present but causally active in a localized, editable form.

3. **Separable controller/executor.** If most failures can be corrected by editing only the loop-length subspace while leaving the arithmetic circuitry untouched, this is strong evidence for modular computation inside transformers, with the controller (loop counting) and executor (arithmetic) occupying distinct representational subspaces.

## Why This Matters Beyond Collatz

If transformers struggle less with arithmetic than with learning control flow, this connects to broader questions about LLM reasoning. Chain-of-thought and scratchpads help by externalizing intermediate control states. But if the control state is already present internally and can be extracted or steered, this suggests new training objectives (auxiliary loop-count heads, control-token supervision) that target the real bottleneck directly. It also complements findings that positional and representational design often matters more than model scale for arithmetic tasks.

## References

- Charton, F. and Narayanan, A. (2025). *Transformers know more than they can tell: Learning the Collatz sequence.* [arXiv:2511.10811](https://arxiv.org/abs/2511.10811)
- Turner, A. et al. (2023). *Activation Addition: Steering Language Models Without Optimization.* [arXiv:2308.10248](https://arxiv.org/abs/2308.10248)
- Nanda, N. et al. (2023). *Progress measures for grokking via mechanistic interpretability.* [arXiv:2301.05217](https://arxiv.org/abs/2301.05217)
- Conmy, A. et al. (2024). *How to use and interpret activation patching.* [arXiv:2404.15255](https://arxiv.org/abs/2404.15255)
- Nye, M. et al. (2021). *Show Your Work: Scratchpads for Intermediate Computation with Language Models.* [arXiv:2112.00114](https://arxiv.org/abs/2112.00114)
- McLeish, S. et al. (2024). *Transformers Can Do Arithmetic with the Right Embeddings.* [arXiv:2405.17399](https://arxiv.org/abs/2405.17399)
