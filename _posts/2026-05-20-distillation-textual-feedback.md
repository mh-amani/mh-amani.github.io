---
title: Textual feedback for reasoning, math, and code
subtitle: A literature review on distillation and natural-language critique
layout: default
date: 2026-05-20
keywords: reinforcement learning, language models, distillation, textual feedback
unlisted: true
sitemap: false
render_with_liquid: false
published: true
---

This review covers methods that train LLMs to use textual feedback — natural-language critiques, execution traces, error messages, demonstrations — for reasoning, math, and code tasks. We survey works with feedback from two sources: deterministic environments (unit tests, verifiers, execution sandboxes) and stronger LLM critics (teacher models that produce natural-language analyses of student outputs). The unifying question here will be *how to turn unstructured textual signal into training gradient*.

## Three axes

Methods sort along three roughly orthogonal axes.

1. **Feedback source.**
   - *Environment* (execution traces, unit tests, verifier outputs): RLEF [8], CTRL [6] (sandbox-grounded critic), SDPO [1] when paired with tests.
   - *LLM critic* (a stronger teacher emits natural-language analysis): Text2Grad [2], Critique-GRPO [4], CFT [3].
   - ILF [5] uses human-written feedback for the code experiments; the mechanism is identical when feedback comes from an LLM.
   - SDFT [9] uses the same student with in context demonstrations, and On-Policy Distillation [10] uses teacher logits as feedback.

2. **Coupling between text and the optimizer.**
   - Text $\to$ reward model → PPO/GRPO: Text2Grad [2], CTRL [6].
   - Text in-context $\to$ distillation into the unconditioned model: SDPO [1], SDFT [9], On-Policy Distillation [10].
   - Text $\to$ refined rollout $\to$ SFT or GRPO on the refinement: ILF [5], Critique-GRPO [4].
   - Text as raw observation for the next attempt in a multi-turn RL episode: RLEF [8].
   - Text as the SFT target itself: CFT [3].
   - Text at inference only, no weight updates: Feedback Descent [7].

3. **Credit-assignment granularity.**
   - *Sequence-level scalar* (one number per output, broadcast across tokens via standard policy-gradient or cross-entropy): CFT [3] (CE on critique target), Critique-GRPO [4] (group-relative advantage; the $f(p)$ shaping is gradient-level, not credit), ILF [5] (CE on refinement), CTRL [6] (GRPO on critic with binary downstream reward).
   - *Turn-level* (one advantage per dialogue turn, broadcast across that turn's tokens): RLEF [8].
   - *Span / token*: Text2Grad [2].
   - *Logit-level* (full or top-K distribution across vocab per position): SDPO [1] (top-K=100), SDFT [9] (full vocab), On-Policy Distillation [10] (full vocab).
   - *N/A — no gradient*: Feedback Descent [7] (artifact-level acceptance, no per-token signal).


Training with textual feedback started with execution-grounded RL on code, wiring raw `stderr` / `stdout` into the dialogue context (RLEF, late 2024) — no teacher in the loop, just verifier signal piped back to the model. The next wave used LLM-generated critiques as denser supervision than scalar verifier signal: either offline as SFT targets (CFT, ILF) or online as span/token rewards (Text2Grad) or as refined rollouts entering the GRPO group (Critique-GRPO, CTRL). Most recently, a "self-as-teacher" line collapses the teacher into the student — the same model conditioned on feedback or a demonstration produces supervision for the same model unconditioned, distilled via reverse-KL on on-policy trajectories (SDPO, SDFT, On-Policy Distillation). 

On a parallel direction we also have inference-time scaffolding (Self-Refine, Reflexion, ReAct, TextGrad, Feedback Descent [7])=.) These works do not have a training step, just the model conditioned on feedback or a demonstration updates a certain text in a loop.

The strongest empirical results in code and math today come from methods that move *all three axes simultaneously*: online, dense-credit, critique-rich. SDPO [1] is the current best demonstration on competitive code and chemistry; Critique-GRPO [4] and CTRL [6] are the strongest critique-trained-RL results on math and code respectively.


## References

<!-- Format: [n] Authors. "[Title](url)." Venue, Year. -->

[1] Hübotter et al. "[Reinforcement Learning via Self-Distillation](https://arxiv.org/abs/2601.20802)." 2026.

[2] Wang et al. "[Text2Grad: Reinforcement Learning from Natural Language Feedback](https://arxiv.org/abs/2505.22338)." May 2025.

[3] Wang et al. "[Critique Fine-Tuning: Learning to Critique is More Effective than Learning to Imitate](https://arxiv.org/abs/2501.17703)." Jan 2025.

[4] Zhang et al. "[Critique-GRPO: Advancing LLM Reasoning with Natural Language and Numerical Feedback](https://arxiv.org/abs/2506.03106)." Jun 2025 (v1) — Feb 2026 (v6). Numbers in this LITREV come from v1 unless noted.

[5a] Chen, Scheurer et al. "[Improving Code Generation by Training with Natural Language Feedback](https://arxiv.org/abs/2303.16749)." TMLR 2024. (Code experiments.)

[5b] Scheurer et al. "[Training Language Models with Language Feedback at Scale](https://arxiv.org/abs/2303.16755)." 2023. (Summarization experiments. The entry below covers both papers under the umbrella name "ILF".)

[6] Xie et al. "[Teaching Language Models to Critique via Reinforcement Learning](https://arxiv.org/abs/2502.03492)" (method: CTRL). Feb 2025.

[7] Lee et al. "[Feedback Descent: Open-Ended Text Optimization via Pairwise Comparison](https://arxiv.org/abs/2511.07919)." Nov 2025.

[8] Gehring et al. "[RLEF: Grounding Code LLMs in Execution Feedback with Reinforcement Learning](https://arxiv.org/abs/2410.02089)." Oct 2024.

[9] Shenfeld, Damani, Hübotter, Agrawal. "[Self-Distillation Enables Continual Learning](https://arxiv.org/abs/2601.19897)." 2026.

[10] Thinking machine team. "[Thinking Machines blog post: On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/)" 2026.



## Individual Paper Summaries


---
### [1] Hübotter et al. "[Reinforcement Learning via Self-Distillation](https://arxiv.org/abs/2601.20802)." 2026.

**Cluster:** Self-as-teacher distillation (with [9], [10]).

**Trained:** A single policy $\pi_\theta$ that serves *both* as student and teacher — no separate reward model or value network.

**External (not trained):** Environment feedback $f$ (e.g., runtime errors, unit test results for code; correctness signals for science QA). The key insight: the *same model* with the same weights acts as teacher when given feedback in-context, and as student when generating without feedback.

**Core mechanism — Self-Distillation Policy Optimization (SDPO):**

1. Sample rollouts $\{y_i\}_{i=1}^G \sim \pi_\theta(\cdot | x)$ (student generates without feedback)
2. Obtain environment feedback $f_i$ for each rollout
3. Compute self-teacher logits: $\pi_\theta(\cdot | x, f_i, y_{i,<t})$ — same model, but now conditioned on feedback
4. Minimize KL divergence at the *logit level*:

$$\mathcal{L}_{\text{SDPO}}(\theta) = \sum_t \text{KL}\left(\pi_\theta(\cdot | x, y_{<t}) \;\|\; \text{stopgrad}\left(\pi_\theta(\cdot | x, f, y_{<t})\right)\right)$$

The stopgrad prevents gradients through the teacher path. The per-token advantage is:

$$A_{i,t}^{\text{sdpo}}(\hat{y}_t) = \log \frac{\pi_\theta(\hat{y}_t | x, f_i, y_{i,<t})}{\pi_\theta(\hat{y}_t | x, y_{i,<t})}$$

This is dense credit assignment: for every token position, the model gets signal on *every vocabulary item* in the top-K, not just the sampled token.

**Why this works (and why it's surprising):** The "teacher" isn't a stronger model — it's the same model that just gets to see the answer/feedback. The gap between "me without feedback" and "me with feedback" is a naturally calibrated, always-improving supervision signal. As the student improves, the teacher automatically improves too (same weights), so the signal never goes stale — unlike offline distillation or fixed reward models.

**Why logit-level > token-level > sequence-level:** GRPO assigns one scalar reward per sequence. SDPO assigns a distribution-level correction at every token position across the top-K vocab items (ranked by teacher probability). This gives orders of magnitude more gradient information per rollout.

**Stability tricks:**

- **Teacher regularization:** The self-teacher uses the same weights as the student, which creates a moving-target problem — as $\theta$ updates, the teacher distribution shifts too, causing training instability. Two fixes:

  (1) *EMA teacher*: maintain a slow-moving copy of parameters updated after each gradient step:
  $$\bar{\theta} \leftarrow (1 - \alpha)\bar{\theta} + \alpha\theta, \quad \alpha = 0.01$$
  The teacher is then $\pi_{\bar{\theta}}(\cdot | x, f, y_{<t})$. This smooths the target across updates — the teacher only incorporates 1% of each new gradient step.

  (2) *Trust-region teacher*: interpolate the teacher *distribution* between the current model and a frozen copy of the initial parameters $\theta_{\text{ref}}$:
  $$q_{\text{TR}}(\cdot | x, f, y_{<t}) = (1 - \alpha) \cdot \pi_\theta(\cdot | x, f, y_{<t}) + \alpha \cdot \pi_{\theta_{\text{ref}}}(\cdot | x, f, y_{<t}), \quad \alpha = 0.01$$
  This bounds drift: the teacher can improve with training (unlike frozen $\theta_{\text{ref}}$ alone) but is always anchored toward the initial distribution. Without either regularization, training diverges (36.1% vs 50.6% with trust-region).

- **Jensen-Shannon divergence:** Standard KL divergence $\text{KL}(p \| q)$ is asymmetric — it penalizes the student heavily when $q$ is near zero but $p$ is not, but barely penalizes the reverse. This creates instability when teacher and student disagree sharply. JSD symmetrizes by averaging: $\text{JSD}(p \| q) = \frac{1}{2}\text{KL}(p \| m) + \frac{1}{2}\text{KL}(q \| m)$ where $m = \frac{1}{2}(p + q)$. This bounds the loss and prevents exploding gradients when the teacher proposes a very different distribution than the student.

- **Top-K=100 approximation:** Computing full KL over the entire vocabulary (~128K tokens) at every position is memory-prohibitive. They only compute the divergence over the top-100 logits, which captures most of the probability mass while keeping memory tractable.

**Results:**

| Setting | SDPO | GRPO | Notes |
|---------|------|------|-------|
| LiveCodeBench v6 (Qwen3-8B) | 48.8% | 41.2% | Beats Claude Sonnet 4 (40.5%) — spot-check Fig 1 |
| Chemistry (Olmo3-7B) | 76.8% | 46.8% | 5h training |
| Very hard problems (discovery@2750, test-time) | 53.2% | 35.6% (multi-turn) / 41.5% (best-of-k) | |

SDPO reaches GRPO's final accuracy in 4× fewer generations. Also produces 3–7× shorter outputs while maintaining higher accuracy.

**Key ablations:**

- **Credit assignment granularity:** Logit-level (top-100 vocab items per position) >> token-level (only the most likely token gets signal) >> sequence-level (one averaged advantage per whole response). Even sequence-level SDPO beats GRPO, showing the self-distillation mechanism itself is valuable independent of granularity. But logit-level gives the largest gains because the teacher can express "shift probability from token A to token B" — information lost when you collapse to a scalar.

- **Teacher regularization** (Best Accuracy column, Table 4): Non-regularized (raw same-step weights as teacher) → 36.1%. Trust-region → 50.6%. EMA → 49.3%. Frozen initial teacher → 48.8%. The trust-region wins because it allows the teacher to improve with training (unlike frozen) while preventing runaway drift (unlike non-regularized).

- **Feedback composition** (Best Accuracy column, Table 6): The self-teacher's context window can include different combinations of feedback. "Output + solution" (environment execution result + a correct solution from the same batch) → 48.9% (best). "Output only" → 39.8%. "Solution only" → 36.8%. Adding the student's own failed attempt to context → 44.5% (hurts). Why adding the student's attempt hurts: the teacher becomes anchored to the student's reasoning path rather than exploring alternatives, reducing the diversity of the distillation signal.

- **Scaling:** Works well at 8B (Qwen3-8B); inconsistent at <1.5B (SDPO underperforms GRPO on Qwen2.5-1.5B). Hypothesis: smaller models lack the in-context learning capacity to meaningfully shift their distribution when given feedback, so the teacher-student gap carries less signal.

**Relation to textual feedback theme:** SDPO is a clean mechanism for converting *any* textual/structured feedback into dense token-level training signal without needing a separate reward model, critic, or span annotation pipeline. The feedback enters via the teacher's context window rather than being parsed into rewards. Compared to Text2Grad [2], which explicitly extracts span-level pseudo-rewards, SDPO lets the model implicitly figure out which tokens matter by comparing its own logits with and without feedback.


Quote from the paper:
Conceptually, our work is related to “expert iteration” (Anthony et al., 2017) where a student is bootstrapped by repeatedly 
imitating an improved version of itself (called the “expert”). Canonically, the expert combines the student with test-time 
search, such as tree search (Anthony et al., 2017) or majority voting (Zuo et al., 2025). In contrast, SDPO leverages the 
student’s ability to learn from rich feedback provided in-context.

**Example interaction** (verbatim from Appendix F.2 "Examples" + Table 2 + Figure 3 / Figure 22 captions; LiveCodeBench v6 trace with Qwen3-8B):

*Self-teacher reprompt template (Table 2):*

```
User: <prompt>
      Correct solution:
      <successful_previous_rollout>
      The following is feedback from your unsuccessful earlier attempt:
      <environment_output>
      Correctly solve the original question.
Assistant: <original_response>
```

Per Table 2 caption: the `successful_previous_rollout` paragraph is skipped if no sibling rollout in the batch solved the question; the `environment_output` paragraph is skipped if the student's own attempt was successful. The point is to *re-evaluate the log-probabilities of `original_response` under the self-teacher* — the assistant turn is the student's failed code, fed back through the teacher with feedback prepended.

*Student query $x$ (LCBv6 binary-string trade problem, truncated):*

> You are given a binary string `s` of length `n` [...]. You can perform at most one trade: convert a contiguous block of `'1'`s surrounded by `'0'`s to all `'0'`s, then convert a contiguous block of `'0'`s surrounded by `'1'`s to all `'1'`s. Return the maximum number of active sections in `s` after making the optimal trade. [... 4 examples truncated ...] Your solution should have the following signature: `def maxActiveSectionsAfterTrade(s: str) -> int:`

*Student rollout $y$ without feedback (Qwen3-8B, truncated — full code in Appendix F.2):*

> `<think> </think>` ... "We'll use a sliding window technique..." ... emits nested loop that builds `temp = list(s)`, mutates `temp[j] = '0'` / `'1'`, then `count = sum(temp)` — summing a list of `str` instead of `int`.

*Environment feedback $f$ piped into the teacher's context:*

```
b'Runtime Error\nTypeError: unsupported operand type(s) for +: \'int\' and \'str\'\nLine 48 in maxActiveSectionsAfterTrade (Solution.py)\n\nLast Executed Input\n"11000"'
```

(Format matches Figure 3's `ZeroDivisionError` example and Listings 5–7 in F.3.)

*Teacher distribution shift (Figure 22 / F.4 caption, verbatim):*

> "The first row shows the tokens of the generated response. The 3 other rows show the top-$K$ logits of the self-teacher that are used during self-distillation, suggesting alternative tokens. Notably, in this example, the self-teacher identifies the error through retrospection without an explicit solution. The credit assignment on the generated sequence, and the alternative top-$K$ logits correctly show that **replacing `set` with `dict` maintains the order of elements**. Further, in the seventh shown position, the model also identifies an alternative solution path which starts with the seen token, instead of directly returning the output. The activation is sparse, identifying where mistakes happen and adjusting to the students' response distribution for specifically these few tokens."

A second concrete shift, from Figure 9's caption (re-uses the Figure 4 example):

> "Shown in blue are tokens which become more likely under the self-teacher. The self-teacher identifies how the returned range has to be modified so that it does not contain `n`."

So the per-token advantage $A_{i,t}(\hat{y}_t) = \log \tfrac{\pi_\theta(\hat{y}_t \mid x, f, y_{<t})}{\pi_\theta(\hat{y}_t \mid x, y_{<t})}$ is positive (blue) on the handful of tokens where the teacher reroutes — e.g. at the `set` position the teacher concentrates mass on `dict`; at the `return` position it concentrates mass on tokens that begin a corrected range expression — and ~zero on the long correct-by-default scaffold tokens. GRPO would smear one negative scalar over every token.

**Caveats:**
- The Appendix F.2 / Figure 22 example shows the *input side* end-to-end (prompt, student rollout, environment feedback string) but the logit-level shift itself is a **figure** (`x19.png`) that does not render in the HTML — the F.4 caption is the only verbatim description of "at position $t$, teacher shifts probability from token A to token B" (the `set` → `dict` swap and the alternative `return`-path token). The paper reports "sparse activation" qualitatively; no numeric probabilities like "0.62 → 0.31" are given.
- The Figure 22 example (`set` / `dict`, swap-order discussion) and the Appendix F.2 prompt (binary-string trade, `TypeError`) appear in the same appendix but it is ambiguous whether Figure 22 visualizes *this* prompt or a different LCB example — the trade problem has no obvious `set` usage in the visible truncated code. Treat the prompt+feedback (F.2) and the logit-shift description (F.4, Figure 22) as **two related illustrations from the same training run, not necessarily the same single token sequence**.
- `<successful_previous_rollout>` is filled from a *sibling rollout in the same batch* — the teacher's "extra context" can include a correct solution to the same question produced by another sample of the student. This is closer to in-batch self-imitation than to pure feedback-only conditioning, and Table 6 in the paper confirms "Output + solution" is the best-performing feedback composition (48.9%) vs feedback alone (39.8%).
- The pipeline conditions the teacher on $f$ *and re-scores the same student trajectory $y$* — it does **not** sample a fresh teacher rollout. The "distribution shift at position $t$" is the gap between $\pi_\theta(\cdot \mid x, y_{<t})$ and $\pi_\theta(\cdot \mid x, f, y_{<t})$ over the **top-100 vocab items at that position**, not between two generated continuations.


---
### [2] Wang et al. "[Text2Grad: Reinforcement Learning from Natural Language Feedback](https://arxiv.org/abs/2505.22338)." 2025.

**Cluster:** LLM-critique-as-RL-signal (with [4], [6]).

<img src="/assets/images/distillation-textual-feedback/text2grad01.png" alt="Text2Grad Architecture" style="zoom:33%;" />


**Two things trained:**
1. **Reward model** $R_\phi$ (Llama-3.1-8B-Instruct, finetuned) — outputs critique + span labels
2. **Policy** $\pi_\theta$ (Llama-3.1-8B-Instruct) — updated via PPO using token-level pseudo-rewards

**Pipeline:**
1. **Annotation:** GPT-4o generates paired feedback per (prompt, response) — a natural language critique $c$ and a span-level reward map $\mathcal{A}(y)$ as JSON (`good_spans`, `poor_spans` — exact quotes from response)
2. **Reward model training:** Finetune Llama-3.1-8B to jointly generate $[c; \mathcal{A}(y)]$ given $[x; y]$ via causal LM loss: $\mathcal{L}(\phi) = -\mathbb{E}[\log p_\phi(z \mid x, y)]$ where $z = [c; \mathcal{A}(y)]$
3. **NL-Gradient PPO:** At train time, reward model generates critique + spans → spans aligned to tokens → pseudo-rewards drive PPO

**Critique → token-level pseudo-rewards:**

$$\delta_t = \begin{cases} +1 & \text{if } t \in s_k \text{ and } \mathcal{A}(y)[s_k] = \text{positive} \\ -1 & \text{if } t \in s_k \text{ and } \mathcal{A}(y)[s_k] = \text{negative} \\ 0 & \text{otherwise} \end{cases}$$
~30% of tokens labeled. The "NL-Gradient": $\nabla_{\text{NL}}(c \to y) = \sum_t \delta_t \cdot \nabla_\theta \log \pi_\theta(y_t \mid x, y_{<t})$ — a token-weighted policy gradient replacing the scalar-reward version.

**Why token-level:** For $\gamma\lambda \approx 0.95$, reward 20 tokens before EOS gets ~2.8× amplification in advantage vs end-of-sequence scalar.

**Why CoT in the reward model matters:** $R_\phi$ generates critique autoregressively *before* outputting spans — CoT scaffolding improves span quality. Removing CoT drops performance 5-17% across tasks, even though the critique text never enters the policy gradient (only $\delta_t$ does).

**Results (Llama-3.1-8B-Instruct):**

| Task | Input → Output | Metric | PPO | Text2Grad | $\Delta$ |
|------|----------------|--------|-----|-----------|---|
| SLF5K | Reddit post → summary | ROUGE-L | 0.262 | 0.291 | +11% |
| KodCode | Spec → Python | Avg pass@1 | 62.5 | 66.1 | +3.6 |
| UltraFeedback | Question → answer | AlpacaEval | 32.4 | 34.7 | +2.3 |

Beats DPO, PRM-PPO, ILF, SFT+Reflection. Generalizes to Mistral-7B.

**Key ablations:**
- Dense labeling (~70% tokens) *hurts* vs span-based (~30%) — function words add noise
- Reward model span precision: 46–84% depending on task and span polarity (positive spans 58–66%, negative spans 46–84%); human-annotation agreement 82–94%
- Mistral-7B generalization: HumanEval 45.7 → 50.0, MBPP 47.1 → 49.6, AlpacaEval 19.6 → 29.4

**Example interaction** (verbatim from Appendix B.3 Listing 3 + Appendix F.1 HumanEval `special_factorial` case study; GPT-4o JSON output reconstructed from the prompt's required schema and the §4.6 `match_parens` critique style):

*Problem (HumanEval `special_factorial`, Appendix F.1 — verbatim):*

```python
def special_factorial(n):
    """The Brazilian factorial is defined as:
    brazilian_factorial(n) = n! * (n-1)! * (n-2)! * ... * 1!
    where n > 0

    For example:
    >>> special_factorial(4)
    288

    The function will receive an integer as input and should return the special
    factorial of this integer.
    """
```

*Student response $y$ (PPO baseline, Appendix F.1 — verbatim; fails unit tests because it computes plain $n!$, not the Brazilian factorial):*

> ```python
> def special_factorial(n):
>     result = 1
>     for i in range(1, n + 1):
>         result *= i
>     return result
> ```

*KodCode GPT-4o annotation prompt template $I_{\text{annot}}$ (Appendix B.3 — verbatim):*

````
Analyze the following code solution for the given problem:

Problem Description:
'''
{problem}
'''

Submitted Code:
'''
{solution}
'''

Test Results:
Passed: {passed}

{%- if not passed -%}
Test Question:
{test_question}

Error Output:
{stdout}
{%- endif -%}

Please analyze the code and identify the following in JSON format:

1. Identify any error-causing code segments directly from the submitted solution.
2. Provide detailed feedback on the code's functionality, issues, and improvement suggestions.
3. Point out any code segments from the solution that work but could be improved.

Return your analysis in this JSON structure:
```json
{
    "Code Feedback": "Provide a detailed explanation of the code's functionality, any potential issues, and suggestions for improvement. Use markdown formatting for better readability.",
    "wrong_code": ["Extract ONLY the problematic code segments FROM THE SUBMITTED SOLUTION that cause failures. Must be exact quotes. Leave empty [] if none found."],
    "improvement_code": ["Extract ONLY the working but improvable code segments FROM THE SUBMITTED SOLUTION. Must be exact quotes. Leave empty [] if none needed."]
}
```
Note: For 'wrong_code' and 'improvement_code', only include direct quotes from the submitted code above, not suggested fixes.
````

*GPT-4o critique $c$ + span annotations $\mathcal{A}(y)$ (**reconstructed** to the schema above; style mirrors the §4.6 `match_parens` critique, which is verbatim from Figure 4: "fails to check `lst[0] + lst[1]` and `lst[1] + lst[0]`"):*

> ```json
> {
>   "Code Feedback": "The function computes the plain factorial of n rather than the Brazilian factorial n! * (n-1)! * ... * 1!. The accumulator multiplies by i instead of by math.factorial(i), so special_factorial(4) returns 24 instead of 288. The loop structure is otherwise correct and can be reused.",
>   "wrong_code": ["result *= i"],
>   "improvement_code": ["for i in range(1, n + 1):"]
> }
> ```

*Derived token-level pseudo-reward $\delta_t \in \{+1,-1,0\}$ via the entry-body rule, applied span-wise to the student response tokens:*

```
def special_factorial(n):     →  0  0  0  0  0  0  0
    result = 1                →  0  0  0
    for i in range(1, n + 1): →  +1 +1 +1 +1 +1 +1 +1 +1 +1 +1    (improvement_code span → +1)
        result *= i           →  -1 -1 -1                            (wrong_code span → -1)
    return result             →  0  0
```

These $\delta_t$ feed the NL-Gradient $\nabla_{\text{NL}}(c \to y) = \sum_t \delta_t \cdot \nabla_\theta \log \pi_\theta(y_t | x, y_{<t})$ inside PPO.

*Text2Grad-trained student response after the update (Appendix F.1 — verbatim; passes tests):*

> ```python
> import math
> def special_factorial(n):
>     if not isinstance(n, int) or n <= 0:
>         raise ValueError("Input must be a positive integer.")
>
>     result = 1
>     for i in range(1, n + 1):
>         result *= math.factorial(i)
>
>     return result
> ```

**Caveats:**
- The GPT-4o JSON critique is **reconstructed** to match the prompt's required schema — Appendix F.1 prints only the four code artifacts (problem, instruct baseline, PPO solution, Text2Grad solution), never the literal critique GPT-4o emitted during training. The verbatim critique snippet in §4.6 ("*fails to check `lst[0] + lst[1]` and `lst[1] + lst[0]`*") is for a different HumanEval task (`match_parens`), whose buggy code is only shown as an image (Figure 4 / `x4.png`), not text.
- The $\delta_t$ map is **derived by hand** from the entry body's rule; the paper colours spans in Figure 4 but never publishes a numeric per-token table.
- Schema differs by task: SLF5K uses `good_spans` / `poor_spans` / `textual_feedback` (Appendix B.1); UltraFeedback uses an XML `<CritiquePrompt>` form (B.2); KodCode uses `wrong_code` / `improvement_code` / `Code Feedback` (B.3). All three map to the same $\delta_t$ at training time.
- The `improvement_code` span is correct code; whether it gets $+1$ or $0$ depends on the implementation — the paper's $\delta_t$ formula assigns $+1$ to anything tagged positive, even if the underlying tokens are not the proximate cause of success.


---
### [3] Wang et al. "[Critique Fine-Tuning: Learning to Critique is More Effective than Learning to Imitate](https://arxiv.org/abs/2501.17703)." Jan 2025.

**Cluster:** Offline critique-as-target SFT (with [5]).

**Trained:** Base LLM $\pi_\theta$ (headline result: Qwen2.5-Math-7B-base; also Qwen2.5-7B and DeepSeek-Math-7B; 32B-scale via Qwen2.5-32B-Instruct) fine-tuned to generate critiques of noisy responses.

**External (not trained):** GPT-4o-1120 generates critique annotations offline during dataset construction. Not used at inference.

**Pipeline:**
1. Sample 50K question-response pairs from WebInstruct (responses are noisy — paper reports >50% contain errors; ~56% correct, ~44% wrong)
2. For each pair $[x; y]$, GPT-4o generates a detailed critique $c$ (identifies errors, explains correct reasoning)
3. Fine-tune the base model on critique generation:

$$\theta^* = \arg\max_\theta \log P(c \mid [x; y]; \theta)$$

4. Train 1 epoch (LR 5e-6, cosine decay, warmup 0.1, batch 512); select best checkpoint via MATH-500
5. At inference, the model generates critiques that implicitly solve the problem — no separate answer head

**Why critique-as-target instead of answer-as-target:** The critique forces the model to identify *what went wrong* and articulate *why*, teaching deeper reasoning patterns than imitating clean answers. Empirically beats SFT on the same data.

**Why noisy responses are the signal source:** The method *requires* errors. When responses are correct, the critique just confirms. Learning comes from erroneous cases where the critique must diagnose and correct. Data that is *worse* for SFT (noisier) is *better* for CFT.

**Results (Qwen2.5-Math-7B-base, 50K training samples; benchmarks: MATH, Minerva-Math, GSM8K, OlympiadBench, AIME24, AMC23):**

| Task | CFT | SFT-GPT4o | Δ (CFT − SFT-GPT4o) |
|------|-----|-----------|---|
| MATH | 79.4% | 73.2% | +6.2 |
| GSM8K | 90.9% | 90.0% | +0.9 |
| Minerva-Math | 36.8% | 25.7% | +11.1 |
| Avg (6 benchmarks) | 56.0% | 50.3% | +5.7 |

(Earlier drafts listed deltas of +10.4 / +6.9 vs. "Best SFT" — those were against a weaker SFT-on-noisy-WebInstruct baseline, not SFT-GPT4o. The +6.2 / +0.9 figures above are the more meaningful comparison since both methods use GPT-4o-generated supervision.)

At 32B scale (Qwen2.5-32B-Instruct), only 4K critique examples suffice to match/beat models trained on 4× more data (Sky-T1-32B with 17K samples). Paper also highlights CFT matching SimpleRL with ~1/144 the GPU hours.

**Key ablations:**
- GPT-4o-mini critiques still beat SFT (CFT-mini avg ~51.5% vs base SFT) — teacher quality matters but even weaker critiques dominate imitation. (Earlier drafts misattributed the +11.1% Minerva-Math delta of CFT-GPT4o vs SFT to GPT-4o-mini — that was wrong.)
- WebInstruct (noisiest source) best for CFT despite worst for SFT — errors are the learning signal
- Reference (noisy) solutions slightly better than self-generated (56.0% vs 54.5%)
- Self-critique at inference does *not* help — the model's own critiques fail to materially improve answers. Note: the "~20% error rate" reported in the paper is the rate of *incorrect teacher (GPT-4o) critiques* found via manual inspection of annotations, **not** the self-critique error rate. The two were conflated in an earlier draft of this LITREV.

**Relation to Text2Grad [2] and Critique-GRPO [4]:** CFT is the simplest member of the "critiques-as-training-signal" family — pure supervised learning on critique text, no RL loop, no token-level rewards, no rollout shaping. Text2Grad converts critiques into span-level pseudo-rewards for PPO; Critique-GRPO uses critiques to generate refined rollouts entering GRPO. CFT shows that even without any RL machinery, the critique *format itself* is a stronger supervision signal than clean answers. The limitation is that CFT is offline and one-shot — the model can't improve beyond teacher critique quality, and self-critique fails. This is exactly the gap that [2] and [4] address by closing the loop online.

**Example interaction** (verbatim from Appendix A.1.3 + Appendix A.2.2 of the CFT paper):

The paper never publishes the actual GPT-4o critique-generation prompt used during dataset construction. The closest proxy is the inference-time "Stage 2" self-critique template (A.1.3), which has the same `[x; y] → critique` shape that the annotation pipeline uses:

```
Please critique whether the following solution to the question is correct.

Question: [Problem text here]
Solution: [Previous solution]

Critique:
1. [Critique point 1]
2. [Critique point 2]
...

Critique Conclusion: Correct/Incorrect
```

A concrete training triple from WebInstruct (Appendix A.2.2, "Case 2: Missed Errors in Solution"):

*Question $x$:*

> Find the roots of the following polynomial: $\frac{43x^2}{5} + 5x + \frac{28}{5}$.

*Noisy response $y$ (WebInstruct, contains an error in the final simplification):*

> Using $x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}$ with $a = \tfrac{43}{5}, b = 5, c = \tfrac{28}{5}$:
> $b^2 - 4ac = 25 - \tfrac{43 \cdot 28}{5} = 25 - 238 = -213$
> $x = \frac{-5 \pm i\sqrt{213}}{2(43/5)} = \frac{-5 \pm i\sqrt{4191}}{86}$.
> [... truncated ...] The roots are $x = \frac{-5 \pm i\sqrt{4191}}{86}$.

*GPT-4o critique $c$ (the SFT target the CFT model learns to produce):*

> Let's verify this step by step:
> The quadratic formula $\frac{-b \pm \sqrt{b^2 - 4ac}}{2a}$ was correctly applied. [... step-by-step recomputation of $a, b, c, b^2, 4ac, b^2-4ac$ ...]
> $x = \frac{-5 \pm i\sqrt{213}}{2(43/5)} = \frac{-5 \pm i\sqrt{4191}}{86}$.
> The denominator 86 comes from $2 \cdot \tfrac{43}{5} \cdot \tfrac{5}{5} = 86$, and $\sqrt{213} \cdot \sqrt{19.67} \approx \sqrt{4191}$.
> Thus, the answer $x = \frac{-5 \pm i\sqrt{4191}}{86}$ must be correct. **Critique Conclusion: correct [END]**

*Authors' meta-analysis (not part of training data — shown in the appendix to illustrate critique noise):*

> The transformation from $\frac{-5 \pm i\sqrt{213}}{2(43/5)}$ to $\frac{-5 \pm i\sqrt{4191}}{86}$ is incorrect. The denominator's 5 was handled ($2 \cdot \tfrac{43}{5} = \tfrac{86}{5}$) but this didn't propagate to the numerator. The correct simplification is $\frac{-25 \pm 5i\sqrt{213}}{86}$.

**Caveats:**

- **The GPT-4o critique-generation prompt is never published in the paper.** The template above is the inference-time self-critique prompt; the actual dataset-construction prompt likely differs. The paper says only "GPT-4o-1120 to generate detailed critiques."
- **This training triple is itself a failure case.** A.2.2 selects it to illustrate the headline limitation that ~20% of GPT-4o critiques contain errors. The critique misses the numerator bug and concludes "correct [END]" — i.e., the CFT model is trained on this as a positive target despite the critique being mathematically wrong. A "clean" training triple would have the critique catch the error.
- **Source of noisy response $y$ is external, not self-generated.** WebInstruct responses are scraped; the paper's ablation (Table) shows reference solutions slightly outperform self-generated (56.0% vs 54.5%).
- **The paper shows no side-by-side CFT-output vs. SFT-output on the same held-out test question**, and no example of the trained CFT model's inference-time critique on a test problem. The transfer from "learn to critique noisy responses" → "directly solve held-out problems via critique-shaped generation" is asserted via benchmark numbers, not concretely illustrated.


---
### [4] Zhang et al. "[Critique-GRPO: Advancing LLM Reasoning with Natural Language and Numerical Feedback](https://arxiv.org/abs/2506.03106)." Jun 2025 (v1) — Feb 2026 (v6).

**Cluster:** LLM-critique-as-RL-signal (with [2], [6]).

Authors: Xiaoying Zhang, Hao Sun, Yipeng Zhang, Kaituo Feng, Chao Yang, Helen Meng (CUHK / Cambridge / Shanghai AI Lab). Numbers below are from v1; results may differ in v6.

**One thing trained:** Policy $\pi_\theta$ (Qwen2.5-7B-Base or Qwen3-8B-Base) via modified GRPO that incorporates critique-guided refinement into the rollout group.

**External (not trained):** GPT-4o as critique model $\pi_\phi$ generating CoT critiques. Rewards are rule-based (Math-Verify): binary +1/0.

**Pipeline:**
1. **Sample & Critique:** Sample $n$ responses per prompt from $\pi_{\text{old}}$. GPT-4o generates CoT critique $c_{\text{CoT}}^{(i)} \sim \pi_\phi(\cdot | I_c, q, y^{(i)})$ for each.
2. **Refine:** Generate refined response conditioned on (question, original, critique): $y_{\text{refined}}^{(i)} \sim \pi_{\text{old}}(\cdot | I_{\text{refine}}, q, y^{(i)}, c_{\text{CoT}}^{(i)})$. Form mixed group: 7 initial + 1 refined.
3. **Shaped GRPO:** Optimize $\pi_\theta$ on the mixed group with a modified objective.

The $I$'s are predefined instructions $I_c$ and $I_{\text{refine}}$.

**Advantage (group-relative over the mixed set):**

$$A_t^{(i)} = R^{(i)} - \text{mean}\left(\{R^{(i)}\}_{i=1}^n \cup \{R_{\text{refine}}^{(i')}\}_{i'=1}^k\right)$$

No division by stdev. 

**Full objective** is a sum of two terms over the mixed group:

$$J_{\text{Critique-GRPO}}(\theta) = \frac{1}{n}\sum_{i=1}^n \sum_t r_t^{(i)}(\theta) \cdot A_t^{(i)} \;+\; \frac{1}{k}\sum_{i'=1}^k \sum_t f\!\left(\pi_\theta(y_{\text{refined},t}^{(i')} | \cdot)\right) \cdot A_t^{(i')}$$

First term: standard GRPO ratio $r_t^{(i)} = \pi_\theta / \pi_{\text{old}}$ over initial responses. Second term: shaped function over refined responses.

**Policy shaping function** (replaces the importance ratio for refined responses):

$$f(p) = \frac{p}{p + \gamma}, \quad \text{where } p = \pi_\theta(y_{\text{refined},t}^{(i')} | q, y_{\text{refined},<t}^{(i')})$$

This is *not* an importance ratio — it's a saturating nonlinearity of the policy's own probability:
- $p \ll \gamma$: $f \approx p/\gamma$ (linear — full gradient on unfamiliar tokens)
- $p \gg \gamma$: $f \approx 1$ (saturated — gradient vanishes for already-learned tokens)

Effect: for correct refinements, the model receives strong gradients on tokens it hasn't yet internalized from the critique (novel reasoning steps). Once learned, the gradient self-attenuates. For incorrect refinements ($A_t < 0$), the negative advantage combined with the shaping penalizes the model for assigning probability to wrong refined tokens.

They remove the clipping function for probability ratios and the KL-divergence penalty term. Their ratio is also very close
to 1. The approach is border line expert iteration except that there are negative samples as well.

**Selection of refined responses:** From $n$ refinements, $k$ are sampled prioritizing correct ones. If no correct refinements exist for a prompt, incorrect ones still enter with negative advantage (active penalty, not just ignored).

**Why CoT critiques specifically:** Indicative critiques ("this is wrong") yield only 2–4% valid refinement rate. CoT critiques (step-by-step error analysis) yield 36–44.7% — the critique must contain enough information to guide correction.

**Results (pass@1, greedy, avg over math+science benchmarks):**

| Model | R1-GRPO | LUFFY | Critique-GRPO | $\Delta$ vs LUFFY |
|-------|---------|-------|---------------|-------------------|
| Qwen2.5-7B | 41.66 | 43.48 | **48.07** | +4.59 |
| Qwen3-8B | 60.68 | 60.91 | **65.86** | +4.95 |

Training: 4K prompts (from OpenR1-Math-220k), 400 steps, 40× A800 GPUs.

**Key ablations:**
- Policy shaping removed: 48.07 → 44.12 (−3.95) — the shaped ratio is load-bearing
- Mixed group vs initial-only or refined-only: mixed wins — contrast between original failures and successful refinements strengthens the signal
- CoT vs indicative critique: mechanism collapses without rich critiques (2–4% vs 36–44% refinement success)

**Relation to Text2Grad [2]:** Both use textual critiques to create richer RL signals, but via different mechanisms. Text2Grad converts critique spans → token-level pseudo-rewards feeding PPO. Critique-GRPO instead uses critiques to *generate refined rollouts* that enter the GRPO group, plus a policy-shaping ratio that implicitly upweights unfamiliar correct tokens. Text2Grad operates at reward granularity; Critique-GRPO operates at rollout composition + gradient shaping.

One cool thing about this paper is that before training, they run an evaluation to find the limitations of RL with numerical feedback and the promise of natural language guidance. 

- The surprising finding: even giving the model the ground truth answer barely helps (2–4%). The model needs step-by-step reasoning about the error to actually correct itself. This is what motivates the whole approach.
-  ~29% of training problems (Qwen2.5-7B) and ~17% (Qwen3-8B) have pass@4=0 throughout training. The model never samples a correct solution
- Self-reflection is not what drives gains. They analyze which behaviors emerge during RL (Figure 3) and find that planning (subgoal-setting, summarization) drives 70–80% of newly-solved problems, while self-reflection contributes minimally. Standard GRPO doesn't teach the model to reflect productively.
(it's not rigorous. They take problems the base model couldn't solve but the RL-trained model can (87 from Minerva-Math, 37 from GPQA-Diamond for Qwen2.5-7B). They categorize six behaviours into planning or reflection...)

**Example interaction** (verbatim from Appendix B prompt templates + Appendix E worked example; refined Qwen3-8B trace after Critique-GRPO training):

*Question:*

> A sphere is inscribed in a cone, such that the area of the sphere and the area of the base of the cone are equal. Calculate the cosine of the angle of the axial section of the cone at the vertex which is also the apex of the cone.

*Initial student response $y^{(i)}$ (failed, truncated):*

> Step 1–3: defines vertex angle $2\theta$; uses similar triangles to get $r = \tfrac{Rh}{h+R}$.
> Step 4: equates areas $4\pi r^2 = \pi R^2 \implies R = 2r$.
> Step 5: substitutes back $\implies h = 2r$.
> Step 6: $\tan\theta = R/h = 1 \implies \theta = 45°$, so $\cos(2\theta) = 0$.
> Final answer: $\boxed{0}$.

*Critique prompt template $I_c$ given to GPT-4o:*

```
System: You are a science expert. A student is trying to solve a question. Please explain
briefly (step-by-step) whether the student's solution is correct or not. Finally, conclude
your judgment with: "Conclusion: correct/incorrect [END]."
User:
Question: <Question Content>
Student's Solution: <Generated Solution>
```

*CoT critique $c_{\text{CoT}}^{(i)}$ from GPT-4o (truncated):*

> Step 3: The similar-triangles equation $r = \tfrac{Rh}{h+R}$ doesn't account for the equal-area condition.
> Step 5: This substitution is unnecessary and gives the wrong simplification. The correct approach uses the inradius formula for the triangle in the axial section.
> Step 6: Therefore the angle calculation is wrong — should use the inradius relation, not $\tan\theta = R/h$ directly.
> Correct approach: apply $r = \tfrac{Rh}{\sqrt{h^2+R^2}+R}$, substitute $R=2r$, derive $h$, then use the double-angle identity. The correct final answer is: $\cos(2\theta) = \tfrac{7}{25}$.
> Conclusion: incorrect [END]

*Refinement prompt template $I_{\text{refine}}$ given to the student:*

```
System: You are a helpful assistant.
User: Given the following inputs:
Question: <Question Content>
Previous Solution: <Generated Solution>
Critique: <Critique Content>
Please re-answer by:
- Correcting potential errors identified in the critique, if they exist.
- Providing clear, step-by-step reasoning.
- Placing your final answer within \boxed.
Ensure the revised solution addresses all issues raised in the critique.
```

*Refined student response $y_{\text{refined}}^{(i)}$ (truncated):*

> Re-derives $R = 2r$, $h = 2r$, then invokes the **inradius formula** as the critique suggested: $r = \tfrac{Rh}{\sqrt{h^2+R^2}+R}$.
> Applies the double-angle identity to compute $\cos(2\theta) = \tfrac{1-\tan^2\theta}{1+\tan^2\theta} = \tfrac{7/16}{25/16} = \tfrac{7}{25}$.
> Final answer: $\boxed{\tfrac{7}{25}}$.

**Caveats on this example:**
- The critique includes the ground-truth answer ($7/25$) — strong (answer-leaking) signal, not a pure pedagogical critique.
- The refined solution is logically inconsistent at one step (re-derives $\tan\theta = 1$ but then computes with $\tan\theta = 3/4$ in the arithmetic) yet arrives at the critique-stated correct answer. Suggests the refinement is partly pattern-matching to the critique's stated answer rather than re-doing the calculation cleanly.


---
### [5] Chen, Scheurer et al. "[Improving Code Generation by Training with Natural Language Feedback](https://arxiv.org/abs/2303.16749)" (code) and "[Training Language Models with Language Feedback at Scale](https://arxiv.org/abs/2303.16755)" (summarization). TMLR 2024 + arXiv 2023. they are two papers but share the ILF framework.

**Cluster:** Offline critique-as-target SFT (with [3]).

**Trained:** The base policy $\pi_\theta$ (CodeGen-Mono 6.1B for code; GPT-3 175B for summarization). A separate refinement model $\pi_{\text{Refine}}$ is also fine-tuned for code (another CodeGen-Mono 6.1B instance).

**External (not trained via ILF):** Human annotators (provide NL feedback $f$); unit test verifier (code); FeedME (text-davinci-001) used zero-shot as $\pi_{\text{Refine}}$ for summarization and as InstructRM scorer via prompting; OPT-RM 13B (trained separately on binary preferences for best-of-N at test time).

**Pipeline (ILF — Imitation Learning from Language Feedback, Algorithm 1 for code):**
1. Sample outputs $x_0 \sim \pi_\theta(\cdot|t)$ that **fail** unit tests: $\text{Eval}(x_0, t) = 0$
2. Humans write NL feedback $f$ describing what's wrong and how to fix it
3. $\pi_{\text{Refine}}$ generates refinements $x_1 \sim \pi_{\text{Refine}}(\cdot|t, x_0, f)$ incorporating feedback
4. Filter: keep only refinements passing all tests ($\text{Eval}(x_1, t) = 1$)
5. Fine-tune $\pi_\theta$ on (task description $t$, refinement $x_1$) pairs → $\pi_{\theta^*}$

**Why NL feedback instead of just correct demonstrations:** Refinements are closer to the model's own output distribution (lower validation NLL under $\pi_\theta$), making fine-tuning more sample-efficient. They target the specific bugs the model actually produces — an on-policy signal missing from offline pre-training.

**Core formalism (one paragraph):** The objective is $\min_\theta \mathbb{E}_t[\text{KL}(\pi^*_t, \pi_\theta(\cdot|t))]$ with target $\pi^*_t(x_1) \propto \exp(\beta R(x_1, t))$. With $\beta \to \infty$, this reduces to SFT on the highest-reward refinement per task. NL feedback simply provides easier access to high-quality samples from $\pi^*_t$ than direct teacher demonstrations would. The proposal distribution composes student-attempt, human-feedback, and refinement steps, with a filter $\delta_1(\text{Eval}(x_1,t))$ that keeps only failing→passing transitions for code.

**Results (code generation, CodeGen-Mono 6.1B on MBPPTest):**

| Method | Feedback Source | Fine-tuning Data | pass@1 | pass@10 |
|--------|---------------|-----------------|--------|---------|
| **ILF** | Humans | πRefine refinements (78 examples) | **36%** | **68%** |
| Human-written refinements | — | Human-written refinements | 33% | 68% |
| Zero-Shot | — | — | 26% | 59% |
| MBPP gold programs (SFT baseline) | — | MBPP ground truth programs | 22% | 63% |
| 2-shot InstructGPT | InstructGPT | InstructGPT refinements | 25% | 59% |

Training: MBPP dataset (974 tasks). $\pi_{\text{Refine}}$ trained on 44 tasks; on held-out tasks it generates ≥1 correct refinement in 10 samples for ~61% of tasks (pass@1 ≈ 19%, pass@10 ≈ 47%). Total annotations collected: 195 triples; usable subset: 122 pieces (44 for πRefine + 78 for πθ*). Annotation cost: \$23/sample, 27 min avg.

**Results (summarization, GPT-3 175B on TL;DR):**

| Method (5K training) | Win rate vs. human summaries |
|---------------------|------------------------------|
| ILF + OPT-RM (best-of-64) | **50.8 ± 1.9%** (≈ human-level) |
| OPT-RM best-of-64 FeedME | 45.1 ± 1.9% |
| ILF: fine-tuned on refinements | 31.3 ± 1.7% |
| Fine-tuned on human summaries | 28.9 ± 1.7% |
| FeedME (zero-shot) | 22.5 ± 1.6% |

**Key ablations:**
- $\pi_{\text{Refine}}$'s pass rate declines monotonically as bugs in feedback increase (80% for 1 bug → ~10% for 5 bugs)
- Embedding similarity for refinement selection fails on diverse crowdsourced feedback (48.3% = below random); InstructRM Ensemble works (56.0%)
- Scaling model feedback (InstructGPT) doesn't match human feedback even at 200 tasks — model feedback is often vacuous ("Great job!"), irrelevant, or addresses fewer bugs (avg 1.1 vs 1.8 for humans)

**Surprising findings:**
- Fine-tuning on MBPP gold programs (22%) does *not* beat zero-shot (26%). Gold programs have higher perplexity under CodeGen — they're OOD for the model.
- ILF (36%) outperforms fine-tuning on human-written refinements (33%) despite using model-generated refinements — the model approximates the refinement distribution better than the human-written one (lower NLL on validation).
- ILF outperforms fine-tuning on human summaries at *all* data scales (100, 1K, 5K), despite human summaries being individually higher quality.

**Relation to other papers in this review:** ILF is historically important as an early formalization of training from NL feedback. Compared to Text2Grad [2], which converts critiques into span-level pseudo-rewards for PPO, ILF is simpler: feedback → refinement → SFT. No token-level credit assignment, no RL loop. Compared to Critique-GRPO [4], which uses critiques to generate refined rollouts entering GRPO with policy shaping, ILF just trains on the refinements via standard cross-entropy. Compared to SDPO [1], where the same model acts as teacher when given feedback in-context, ILF uses a separate $\pi_{\text{Refine}}$ model. The key limitation is that ILF is offline and single-pass — the policy doesn't generate new rollouts after improvement, so there's no online signal. The later papers in this review ([1], [2], [4]) all close this loop. ILF's core insight — that feedback-guided refinements are more on-distribution than gold demonstrations — foreshadows the "on-policy" motivation in all subsequent work. *Note on scope:* the code experiments use human feedback; included here because the mechanism is identical for LLM-generated feedback, as Critique-GRPO [4] demonstrates.

**Example interaction** (verbatim from `surge_annotations.jsonl` task_id=1 in the [official ILF code repo](https://github.com/nyu-mll/ILF-for-code-generation), and the $\pi_{\text{Refine}}$ prompt template from `gen_refinement_prompt` in `generate_refinements_codegen_finetuned.py`):

*MBPP task $t$ (prompt + hidden unit tests):*

```python
"""
Write a function to find the minimum cost path to reach (m, n) from
(0, 0) for the given cost matrix cost[][] and a position (m, n) in cost[][].

>>> Example: min_cost([[1, 2, 3], [4, 8, 2], [1, 5, 3]], 2, 2) = 8
"""

R = 3
C = 3
def min_cost(cost, m, n):
```
```python
# hidden tests
assert candidate([[1, 2, 3], [4, 8, 2], [1, 5, 3]], 2, 2) == approx(8)
assert candidate([[2, 3, 4], [5, 9, 3], [2, 6, 4]], 2, 2) == approx(12)
assert candidate([[3, 4, 5], [6, 10, 4], [3, 7, 5]], 2, 2) == approx(16)
```

*Initial student code $x_0$ (CodeGen-Mono 6.1B, fails the tests):*

```python
    if m == -1 or n == -1:
        return 0
    if m == 0 and n == 0:
        return cost[0][0]
    if m == 1:
        return cost[0][n] + min_cost(cost, 0, n-1)
    if n == 1:
        return cost[m][0] + min_cost(cost, m-1, 0)
    return cost[m][n] + min(min_cost(cost, m-1, n), min_cost(cost, m, n-1))
```

*Verifier output:* the three `assert ... == approx(...)` checks all return `passed: false` (binary per-test signal — the dataset records `result: "failed"` for $x_0$; no stderr/traceback is fed back to the model, only the pass/fail bit gates whether the example enters the annotation pipeline).

*Human-written feedback $f$ (Surge annotator, verbatim):*

> There are two crucial mistakes in this code. This is a recursive breadth-first search function, which is a valid approach. However, of the 4 base cases, two of them are incorrect. The last two base cases, which handle the case when the searched node is at the edge of the matrix, are the incorrect ones. They don't terminate at the correct spot; instead, they terminate at index 1, but don't add the cost of that cell to the total cost.
> This is fixed by changing those conditions to terminate at index 0 instead so that every cell is actually counted.
>
> The second mistake is that the recursive part does not account for diagonal movements, which is the only way you can get the correct cost value. This is fixed by adding one more recursive call to the min() function.

*$\pi_{\text{Refine}}$ prompt template (verbatim from `gen_refinement_prompt`; fields are filled by string concatenation, no system prompt, no chat formatting):*

```
OLD CODE:
<truncated x_0, 512 tokens>

FEEDBACK:
<feedback f>

REFINEMENT:
<re-printed MBPP prompt: docstring + first test as ">>> Example:" + function signature>
```

*Refined code $x_1 \sim \pi_{\text{Refine}}(\cdot \mid t, x_0, f)$ (passes all three tests; this is the SFT target for $\pi_{\theta^*}$):*

```python
    if m == -1 or n == -1:
        return 0
    if m == 0 and n == 0:
        return cost[0][0]
    if m == 0:
        return cost[0][n] + min_cost(cost, 0, n-1)
    if n == 0:
        return cost[m][0] + min_cost(cost, m-1, 0)
    return cost[m][n] + min(min_cost(cost, m-1, n),
                            min_cost(cost, m, n-1),
                            min_cost(cost, m-1, n-1))
```

The diff is exactly what the feedback prescribed: `m == 1` / `n == 1` → `m == 0` / `n == 0`, plus a third `min_cost(cost, m-1, n-1)` argument in the recursive `min(...)`. Edit distance vs $x_0$: 0.138.

**Caveats:**
- Feedback is *human-written*, not LLM-generated — the LITREV groups ILF with LLM-critique methods because the downstream training mechanism (refinement-as-SFT-target) is feedback-source-agnostic, but this example does not exercise the "stronger teacher" pathway.
- $\pi_{\text{Refine}}$ frequently fails: on held-out tasks it produces $\geq 1$ passing refinement in 10 samples for only ~61% of tasks (pass@10 $\approx$ 47%). Step 4 of the pipeline filters out the failures — only the passing refinements become SFT targets, so this single shown example is selection-biased toward the success case.
- The shown $x_1$ is the `unedited_annotator_completion` field — the human annotator's own fix, used as the SFT target in the "human-written refinements" baseline (33% pass@1). The actual ILF run (36% pass@1) trains on $\pi_{\text{Refine}}$-generated refinements that condition on the same `(t, x_0, f)`, not on the human's fix directly; the repo records `original_model_completion` (the failing $x_0$) and the annotator's edit but does not store a separate sampled-from-$\pi_{\text{Refine}}$ completion per row.
- The verifier signal is binary pass/fail per assert — no stderr, no traceback, no partial-credit hint enters $f$ or the refinement prompt. Everything diagnostic in $f$ comes from the human reading the code, not from the verifier. This is the cleanest contrast with RLEF [8], where stderr is the entire feedback channel.
- The repo's reported `refinement_edit_distance` (0.138) is computed against the annotator's edited completion, not against an LLM refinement; the paper's claim that refinements are closer to $\pi_\theta$'s distribution than gold MBPP programs (lower validation NLL) is *not* verifiable from this single shown row.


---
### [6] Xie et al. "[Teaching Language Models to Critique via Reinforcement Learning](https://arxiv.org/abs/2502.03492)" (method name: CTRL). Feb 2025.

**Cluster:** LLM-critique-as-RL-signal (with [2], [4]). The critic is the trained model and the generator is frozen — inverted relative to [2] and [4].

**Trained:** A critic model $Q_\theta$ (Qwen2.5-Coder-32B-Instruct) that generates textual critiques of code solutions to guide iterative refinement.

**External (not trained):** Generator model $\pi(y|x,c)$ (frozen during critic training), execution sandbox $R(y)$ for test-case evaluation, reference model $Q_\text{ref}$ for KL regularization.

**Pipeline:**

**Stage 1 — Execution-Guided Critique Synthesis (SFT):**

1. Sample initial solutions $y'$ from generator on TACO problems (18,820 filtered from 26,443)
2. Execute against test cases → map pass / fail / **partial** (= exact error or test-case detail) to hint templates $h$
3. Sample critiques from $Q_\theta(c|z,h)$ conditioned on execution hints
4. Filter out critiques that reference hints directly (removes the crutch)
5. SFT on $\{x, y', c\}$ triples with standard LM loss

**Why hint-then-filter:** Execution feedback gives the critic grounded signal about *what* went wrong, but filtering forces the critic to learn to identify issues from the code itself — bootstrapping critique ability without requiring execution at inference time. Removing the filter causes overfitting to hint format and degrades inference performance.

**Stage 2 — Reinforced Critique Generation (GRPO):**

1. For each problem-solution pair $z = (x, y')$, sample $G=8$ critiques from $Q_\theta$
2. Generator produces revised solution $y_i \sim \pi(\cdot|z, c_i)$ for each critique
3. Execute each revision: $R(y_i) \in \{0, 1\}$
4. Compute group-relative advantages: $A_i = (R(y_i) - \mu_G) / \sigma_G$
5. Update critic:

$$J(\theta) = \mathbb{E}\left[\frac{1}{G} \sum_i \min\left(\frac{Q_\theta(c_i|z)}{Q_{\theta_\text{old}}(c_i|z)} A_i,\ \text{clip}_\varepsilon(\text{ratio}) \cdot A_i\right)\right] - \beta \cdot D_\text{KL}(Q_\theta \| Q_\text{ref})$$

with $\beta = 0.001$.

**Why reward the critic via generator success:** The critic never sees the reward directly — it only gets signal through whether its critique *actually helped the generator fix the code*. This is outcome-based RL where the reward is "did my feedback work?" rather than "does my feedback look good?". Contrast with CFT [3] where critique quality is judged by a teacher offline.

**Why GRPO over PPO:** Group-relative advantages avoid training a value network (unstable for text generation). Comparing critiques within the same group for the same problem is a natural fit since critique quality is only meaningful relative to the same problem-solution pair.

**Results (Qwen2.5-Coder-32B-Instruct):**

| Task | Input → Output | Metric | Zero-shot | CTRL | $\Delta$ |
|------|----------------|--------|-----------|------|----------|
| CodeContests | problem → code | Pass@1 | 7.88% | 15.15% (3-turn) | +7.27 |
| CodeContests (GPT-4o gen) | problem → code | Pass@1 | 20.61% | 25.45% (5-turn) | +4.84 |
| JudgeBench (OOD GPT-4o slice) | response → judgment | Accuracy | — | 64.3% | competes w/ Claude-3.5-Sonnet |
| Discrimination | code → correct/incorrect | F1 | 61.19% | 69.10% | +7.91 |

Training: TACO 18,820 problems. SFT: LR $2 \times 10^{-5}$, batch 256, 1 epoch. RL: LR $1 \times 10^{-5}$, batch 1024, group size 8, 2 epochs.

**Key ablations:**
- SFT-only critic (no RL): much weaker refinement — the RL stage is essential for learning what makes feedback *actionable*
- Removing hint-filtering in Stage 1: critic overfits to hint format, degrades at inference
- Multi-turn compounding: 3-turn refinement achieves 106% relative improvement over zero-shot, with degradation rate ($\Delta\downarrow$) staying at only 0.85% in the Qwen-base 3-turn setting (3.03% in the GPT-4o 5-turn setting) — iterative critique doesn't break what already works in the trained-with regime

**Surprising findings:**
- Weak-to-strong generalization: a 32B critic successfully guides GPT-4o (a larger, stronger generator it never trained with). Critique ability transfers across model boundaries more readily than generation ability.
- The multi-turn degradation rate is remarkably low (0.85% after 3 turns), suggesting the critic learns *when not to intervene* — a property not explicitly trained for.

**Relation to other papers:** CTRL inverts the setup of Text2Grad [2] and Critique-GRPO [4]: instead of training the *generator* using critiques, it trains the *critic* using generator outcomes. The generator is frozen; the critic is optimized. This is complementary — one could combine a CTRL-trained critic with a Critique-GRPO-trained generator. Compared to CFT [3], CTRL adds the RL loop that CFT lacks: the critic improves beyond the initial SFT quality by learning from outcome feedback. The hint-then-filter bootstrap in Stage 1 is analogous to CFT's use of GPT-4o critiques as offline supervision, but Stage 2 closes the loop online.

**Example interaction** (verbatim from Appendix C.2 Table 7 hint templates + Appendix C.2 prompt listings + Appendix E Table 10 — base64-decoded from the arXiv HTML):

*Stage 1 hint templates (Table 7) — map sandbox execution outcome to the hint $h$ injected during critique synthesis:*

```
Success (100%):        The draft solution is correct. A concise and positive
                       feedback is recommended.
Failure (0%):          The draft solution is entirely wrong. A concise feedback
                       requesting a fresh restart is recommended.
Partial Success:       Input: {input}
                       Expected Output: {expected_output}
                       Actual Output: {actual_output}
Runtime Error:         The code block: '{code_block}' raised {error}.
```

*Critique prompt template (inference time, no hint — Stage 2 / deployment form):*

```
You are tasked with analyzing an answer to a problem and providing constructive
feedback. Do NOT provide direct solutions.

Problem description:
<problem>
{problem}
</problem>

Answer:
<answer>
{answer}
</answer>

Structure your response using the following format (without <format> tags):
<format>
Analysis:
{{Analysis}}

Improvement suggestions:
{{Suggestions}}

Overall judgment: {{Correct/Incorrect}}
</format>
```

*Problem $x$ (CodeContests, Table 10):*

> You had n positive integers a_1, ..., a_n arranged in a circle. For each pair of neighboring numbers you wrote down whether they are equal or not. Unfortunately, you've lost the array a. [...] Is there any array a consistent with the equality / non-equality information?

*Initial code $y'$ from frozen Qwen2.5-Coder generator (verbatim — the truncated `a[(i + 1) %` lines are exactly as printed in Table 10):*

```python
def can_form_array(s):
    n = len(s)
    a = [1] * n
    for i in range(n):
        if s[i] == 'E':
            a[(i + 1) %
        else:
            a[(i + 1) %
    if s[-1] == 'E' and a[0] != a[-1]:
        return "NO"
    if s[-1] == 'N' and a[0] == a[-1]:
        return "NO"
    return "YES"
[... main() truncated ...]
```

*Critique $c \sim Q_\theta(\cdot \mid x, y')$ from the CTRL-trained 32B critic:*

> **Analysis:** The provided solution attempts to construct an array based on the sequence of 'E' and 'N' but does not correctly handle the circular nature of the array and the constraints imposed by these characters. [...] The code assigns values based on whether the current character is 'E' or 'N' but does not account for potential conflicts that arise when the array wraps around. [...]
>
> **Improvement suggestions:**
> 1. Consider using a more systematic approach to handle the circular nature of the array. One possible method is to use a graph traversal algorithm [...]
> 2. Use a union-find data structure to manage equivalence classes of indices that must hold the same value due to 'E' connections.
> 3. Iterate through the string and apply the rules: if 'E', ensure the indices are in the same set; if 'N', ensure they are in different sets.
> 4. After processing the string, verify that no conflicts exist in the union-find structure, particularly the circular connection between the first and last elements.
> 5. If no conflicts are found, an array satisfying the conditions can be constructed; otherwise, it cannot.
>
> **Overall judgment:** Incorrect

**Caveats:**
- Table 10 shows the problem + initial solution + critique only — the *revised* code $y \sim \pi(\cdot \mid x, y', c)$ and its sandbox result are not printed in the appendix.
- The critique deliberately suggests an *algorithmic redirection* (union-find) rather than patching the syntactically truncated `a[(i + 1) %` lines — the critic is not acting as a syntax linter.
- The deployed critique prompt has no hint slot: Stage 1 synthesis uses the hinted template over TACO solutions; critiques explicitly referencing the hint were filtered before SFT, and Stage 2 GRPO never injects hints. So the inference-time interaction shown above is hint-free even though training was hint-bootstrapped.
- Per §4.4 the same 32B critic also guides GPT-4o as the generator (5-turn weak-to-strong setting, +4.84 Pass@1) — the depicted Qwen2.5-Coder pairing is the headline same-family setting, not the weak-to-strong one.



---
### [7] Lee et al. "[Feedback Descent: Open-Ended Text Optimization via Pairwise Comparison](https://arxiv.org/abs/2511.07919)." Nov 2025.

**Cluster:** Test-time scaffolding (no weight updates). *Included as the inference-time counterpoint to the training methods above — the cleanest articulation of "use textual feedback as the optimization signal" without an outer training loop.*

**Trained:** Nothing — no model weights are updated. The "optimization" happens over text artifacts (SVG code, prompts, SMILES molecules) at inference time only.

**External (frozen):** Mutator model $\mathcal{M}$ (generates improved candidates), Evaluator $\mathcal{E}$ (provides binary preference + textual rationale). These are multi-model across tasks — GPT-4o-mini, Qwen3-8B, GPT-4.1-mini, Claude for various mutator roles; GPT-5-mini for the SVG evaluator; domain-specific tools (docking scores + RDKit) for molecules. The headline framing is **pairwise comparison** as the evaluator interface (vs. absolute scoring).

**Pipeline:**

1. Initialize artifact $x_0$ via prompted generation
2. **Mutate:** $x_t = \mathcal{M}(x^*_t, \mathcal{R}_{t-1})$ — generate candidate conditioned on current best + accumulated feedback
3. **Evaluate:** $(p_t, r_t) = \mathcal{E}(x_t, x^*_t)$ where $p_t \in \{0,1\}$ is binary preference, $r_t \in \mathcal{S}$ is textual rationale
4. **Update:** always append $(x_t, r_t)$ to $\mathcal{R}$. If $p_t = 1$, set $x^*_{t+1} = x_t$ **and reset** $\mathcal{R} \leftarrow \emptyset$; otherwise $x^*_{t+1} = x^*_t$ and keep $\mathcal{R}$.
5. Repeat for $T$ iterations or until $k$ consecutive non-improvements

No loss function. No gradients. No backpropagation. Improvement emerges from the LLM's ability to translate accumulated textual feedback into directional improvements in semantic space.

**Why reset history on acceptance:** Once an improvement is accepted, past feedback (which described flaws of the now-discarded artifact) becomes stale. Resetting keeps the context focused on what's wrong with the *current* best.

**Why this differs from TextGrad:** TextGrad proposes improvements *pointwise* — conditioning only on the latest artifact with no memory. Feedback Descent maintains a trajectory-level buffer of comparative feedback accumulated since the last successful update. On molecule optimization (1000+ steps), TextGrad plateaus while Feedback Descent continues improving. TextGrad is essentially Feedback Descent with the history mechanism ablated.

**Results:**

| Task | Input→Output | Metric | Baseline | Feedback Descent | $\Delta$ |
|------|-------------|--------|----------|-----------------|---|
| SVG generation | description→SVG | Win rate vs init | — | 80–100% | — |
| IFBench prompts (Qwen3-8B) | task→optimized prompt | Accuracy | GEPA | 38.78% | best |
| HoVer prompts (Qwen3-8B) | task→optimized prompt | Accuracy | GEPA | 60% | best |
| Molecule ADRB1 | seed SMILES→optimized | −Vina−10(1−QED) | TextGrad: 8.531 | 10.623 | +2.09 |
| Molecules (6 targets) | seed SMILES→optimized | Combined score | Graph GA, REINVENT, TextGrad | >99.9th pctile of 260K compounds | — |

Molecule optimization: batch size 8, top-k=10, three trivial seeds (acetamide, pentane, benzene). Prompt optimization: 10–15 iterations, temperature 0.6–1.0, early stopping patience 0–5.

**Key ablation:** TextGrad (= no accumulated history) does not scale to high iteration budgets — pointwise conditioning plateaus while trajectory-level feedback continues improving.

**Surprising findings:**
- On some targets, discovered molecules surpass the *best compound* in the entire 260K database — despite starting from trivial seeds (acetamide, pentane, benzene).
- The theoretical result (dimension-free convergence) is a rare formalization of the intuition that "LLMs understand semantic directions" — it grounds the claim that textual feedback is fundamentally richer than scalar reward.

**Example interaction** (verbatim from Sections 2.2–2.3 "running SVG example" + Algorithm 1 + Appendix C.1 "SVG Code Optimization"; the paper does *not* print any complete literal artifact end-to-end — see Caveats — so this entry reproduces the verbatim feedback snippets and pseudocode, with the artifact slots described).

*Task spec (Section 5.2):* render an SVG illustration of a unicorn that wins a pairwise comparison under one of six judge rubrics (Ink Wash / Minimalist / Realism / Retro Arcade / Stained Glass / Anatomy). Mutator and judge are both `gpt-5-mini`. The judge "outputs both a binary preference and short textual feedback".

*Initial artifact $x_0$:* generated by "prompting a language model with the task description alone (e.g., ``Generate SVG code for a unicorn'')". The literal SVG bytes are not printed; only rendered PNGs appear in Figure 2 (unicorn realism progression) and Figure 3 (six rubric variants).

*Algorithm 1 (exact pseudocode):*

```
Input: Initial text x_0, Language model M, T
Current best: x* ← x_0,  Rationale history: R ← ∅
for t = 1 to T do
    x_t ← M(x*, R)                    ▷ Propose
    p_t, r_t ← Compare(x_t, x*)       ▷ Compare
    R ← R ∪ {(x_t, r_t)}
    if p_t = 1 then
        x* ← x_t,  R ← ∅              ▷ Update + reset
return x*
```

*Examples of evaluator rationale $r_t$ for the unicorn task (Section 2.2, quoted verbatim):*

> "adjust the stroke width"

> "make sure the legs are connected to the body"

> "add a shadow to the unicorn's mane"

> "needs more defined horn shape"

> "legs disconnected from body; increase stroke width"

*Accumulated rationale buffer $\mathcal{R}$:* the paper specifies the update rule $\mathcal{R}_{t+1} = \mathcal{R}_t \cup \{(x_t, r_t)\}$ with a hard reset $\mathcal{R} \leftarrow \emptyset$ on every accepted candidate. So at iteration $t$, $\mathcal{R}$ contains every (candidate, rationale) pair since the last acceptance — e.g. after three rejected mutations, $\mathcal{R} = \{(x_{t-2}, \text{``adjust the stroke width''}),\ (x_{t-1}, \text{``legs disconnected from body; increase stroke width''}),\ (x_t, \text{``needs more defined horn shape''})\}$. No example $\mathcal{R}$ contents are printed in full.

*Mutator's next candidate $x_{t+1}$:* paper does not print SVG code. Visual evidence is Figure 2, captioned: "Iterative progression of SVG unicorn optimization under the realism judge. **Feedback Descent produces gradual, semantically meaningful improvements through accumulating directional cues.**" The mutator + judge implementation: "tournament-style approach where `gpt-5-mini` generates SVG/TikZ code that gets rendered to PNG images for pairwise aesthetic comparisons by a separate instance of the same model acting as judge. The system maintains a ``champion'' design that only updates when both A-vs-B and B-vs-A orderings consistently agree on a winner, accumulating winning rationales into the generation prompt to guide aesthetic improvements across iterations." (Appendix C.1)

*Molecule trajectory (Section 5.4 + Appendix C.1, for the longer-horizon counterpart):* seeds are the literal SMILES strings `CC(N)=O` (acetamide), `CCCCC` (pentane), `c1ccccc1` (benzene). For ADRB1 over 1000 steps, the combined score $-\text{Vina} - 10(1 - \text{QED})$ improves from the trivial seed level to **10.623** (vs. TextGrad's **8.531**, Graph GA's **9.145**, the previous-best baseline). No intermediate SMILES strings are printed in the paper.

**Caveats:**
- *No literal artifact trace is printed end-to-end.* The paper's case studies are figures (rendered PNGs for SVG, scatter plots for molecules), not text dumps. The Appendix C.3 prompt-template listings are typeset as figure-images in the HTML/PDF, so even the mutator and judge prompts cannot be extracted verbatim from `arxiv.org/html/2511.07919v1`. Anything reproducing actual SVG/SMILES strings here would be fabrication.
- *Multi-model mutator.* "These are multi-model across tasks — GPT-4o-mini, Qwen3-8B, GPT-4.1-mini, Claude for various mutator roles; GPT-5-mini for the SVG evaluator" (from this LITREV's [7] header). The example feedback snippets all come from the SVG/gpt-5-mini setting; behavior with weaker mutators is not characterized at the same fidelity.
- *Reset-on-acceptance.* The buffer $\mathcal{R}$ in any printable example would necessarily be from a *rejection streak* — the paper's mechanism erases history at the moment of progress, so "the accumulated buffer at convergence" is empty by construction.
- *Feedback snippets are illustrative, not from a logged run.* The five quoted rationales in Section 2 are the paper's own gloss of "examples of feedback" for the running unicorn example — they are presented as plausible evaluator outputs rather than as a transcript of a specific iteration $t$.
- *Tournament + consistency check.* Appendix C.1 reveals that an accepted update actually requires *both* A-vs-B and B-vs-A orderings to agree — a detail elided from the headline Algorithm 1, which presents acceptance as a single $p_t = 1$ check.

---
### [8] Gehring et al. "[RLEF: Grounding Code LLMs in Execution Feedback with Reinforcement Learning](https://arxiv.org/abs/2410.02089)." Oct 2024.

**Cluster:** Execution-grounded RL. Standalone in this LITREV — the only entry where the entire feedback signal is raw verifier output, no teacher or critic in the loop.

**Trained:** A code-generating policy $\pi_\theta$ (Llama 3.1 Instruct 8B/70B) and a separate turn-level value function $V_\psi$ (same backbone, new scalar head).

**External (not trained model, but frozen environment):** Python 3.10 interpreter, the problem's *public* test set (used for intra-episode feedback), and the *private* test set (used only to score the final answer).

**Pipeline (per episode, up to $T=3$ turns):**
1. Prompt the model with problem $o_0$; sample code $a_0 \sim \pi_\theta(\cdot|c_0)$.
2. Run $a_t$ on public tests. Serialize pass/fail status, stderr, stdout/expected diffs into a natural-language observation $o_{t+1}$.
3. Append $(a_t, o_{t+1})$ to the dialogue context $c_{t+1} = (o_0, a_0, o_1, \dots, o_{t+1})$ and sample a revised solution.
4. Terminate on all-public-pass or at $t=T-1$. Submit the *last* code to private tests for the terminal reward.
5. Optimize with PPO: 1024 rollouts/update, 4 epochs of 256-sequence minibatches.

**Reward:**

$$r(s_t, a_t) = \begin{cases} +1 & \text{terminal, all private tests pass} \\ -1 & \text{terminal, any private test fails} \\ -0.2 & \text{non-terminal turn with invalid (uncompilable/unrunnable) code} \end{cases}$$

Plus per-token KL to the SFT init folded into the reward: $R(s_t,a_t) = r(s_t,a_t) - \beta \log \frac{\pi_\theta(a_t|c_t)}{\rho(a_t|c_t)}$, with $\beta=0.05$.

**Objectives (PPO-clip):**

$$\mathcal{L}^\pi(\theta) = -\mathbb{E}_t\big[\min(\rho_t(\theta)\hat{A}_t,\ \mathrm{clip}(\rho_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t)\big], \quad \epsilon=0.2$$

$$\mathcal{L}^V(\psi) = \tfrac{1}{2}\,\mathbb{E}_t\big[\max((V_\psi(c_t)-R_t)^2,\ (\mathrm{clip}(V_\psi, V_{\psi_\text{old}} \pm 0.2) - R_t)^2)\big]$$

with undiscounted return $R_t = \sum_{i=t}^{T} R(s_i,a_i)$ and turn-level advantage $\hat{A}_t = R_t - V_\psi(c_t)$ **applied uniformly to every token in turn $t$**.

**Why turn-level value, token-level policy:** Reward only arrives at episode end and is scored per *turn* (the code that ran), not per token. A token-level critic has to smear a far-away binary signal across thousands of tokens; a turn-level critic gets cleaner regression targets at decision boundaries. The policy still updates token-by-token because that's where gradients are cheapest. The authors report this hybrid beat both pure-token and pure-turn configurations in ablation.

**Why KL-in-reward rather than in loss:** Treats the KL as just another shaping term inside the advantage, so the value function learns the KL-penalized return directly — avoids needing a second regularizer coefficient at the loss level and lets PPO clipping handle policy updates uniformly.

**Why the $-0.2$ for invalid code:** Without it, the model can burn intermediate turns on garbage that never even executes, producing no learnable feedback. The penalty nudges the policy to at least emit runnable code so subsequent turns have real observations to condition on.

**Why public-only feedback during the episode:** If the agent saw private tests during rollouts it would just overfit to them. The public/private split is what makes "use feedback to generalize" a nontrivial objective — analogous to train/test in classical ML, embedded inside a single episode.

**Results (CodeContests test set):**

| Setting | Base | Method | Δ |
|---|---|---|---|
| 1@3, 8B | Llama 3.1 8B instruct: 10.5 | +RLEF: 16.0 | +5.5 |
| 1@3, 70B | Llama 3.1 70B instruct: 27.5 | +RLEF: 40.1 | +12.6 |
| 10@100, 70B | Instruct: 50.3 | +RLEF: 54.5 | +4.2 |
| SOTA comp. | AlphaCodium+GPT-4: 29 | RLEF 70B 1@3: 40.1 | +11.1 at ~1000× fewer samples |

HumanEval+ / MBPP+ (1@3, multi-turn): 70B HumanEval+ 75.0 → 80.4; MBPP+ 70.2 → 72.2.

**Key ablations (CodeContests test, 8B):**

| Ablation | Test |
|---|---|
| Few-shot baseline | 8.5 |
| SFT on solutions | 10.0 |
| Single-turn RL (no feedback loop) | 10.9 |
| RL with feedback *withheld at train time* | 10.9 |
| Token-level value function | 13.7 |
| Full RLEF | **16.0** |

Random-feedback probe: swapping the execution-feedback string with noise at eval collapses performance — the policy actually *conditions on the feedback content*, not just on "I got another turn."

**Surprising findings:**
- Sample-efficiency jump is massive — 8B RLEF with 3 samples beats AlphaCode 9B with 1000 samples. The gain isn't from a stronger prior but from cheap feedback substituting for expensive resampling.
- Single-turn RL barely beats SFT (10.9 vs 10.0 on 8B) — the delta over baseline is almost entirely the multi-turn feedback mechanism, not "RL on code."
- Feedback-withheld RL ties with single-turn RL — you have to actually let the model see the stderr/stdout during *training*. The policy has to learn to parse and act on executor output, and that only happens if it's in the context during training, not just at inference.
- Instruct-tuned bases improve more than pretrained bases — the model needs pre-existing instruction-following to parse structured feedback; RL can't bootstrap that from scratch at this scale.

**Relation to other papers in this review:** RLEF is the "environment-grounded" counterpoint to the critique-based methods. Where Text2Grad [2], Critique-GRPO [4], and CTRL [6] all rely on LLM-generated textual critiques (GPT-4o, Claude, or a trained critic) to densify reward, RLEF's "feedback" is just raw Python stderr/stdout — no teacher model in the loop. The feedback is low-semantic-density but high-reliability. Mechanism-wise it's closest to CTRL [6]: both target code, both use a sandbox as ground truth, both are multi-turn. The split is that CTRL *trains a critic* to narrate execution results into actionable critiques while the generator stays frozen, whereas RLEF trains the generator directly and uses raw executor output verbatim. Compared to SDPO [1], which builds a self-teacher from in-context feedback and distills into the student's logits, RLEF keeps feedback external (in the dialogue) and uses scalar rewards for credit assignment — simpler but coarser. RLEF is the cleanest demonstration that when verifier signal is cheap and reliable (unit tests), you don't need critiques or teachers at all — just wire the environment output back into the context and run PPO.

**Example interaction** (verbatim from Appendix D, first example; RLEF-trained Llama 3.1 70B on CodeContests valid set, $T=3$, temperature 0.2, top-p 0.95).

*Problem $o_0$* (CodeContests "a mod b" task — find the largest $a \bmod b$ over pairs with $r \geq a \geq b \geq 1$):

> Provide a Python solution for the following competitive programming question: You are given two integers $l$ and $r$, $l \leq r$. Find the largest possible value of $a \bmod b$ over all pairs $(a, b)$ of integers for which $r \geq a \geq b \geq 1$. [... input format: $t$ test cases, each with $l, r$ up to $10^9$; expected outputs `0, 1, 12, 499999999` for the four sample cases ...] Your code should be enclosed in triple backticks like so: ```` ```python YOUR CODE HERE ``` ````.

*Initial code $a_0$* (turn 1) — branches on `r // 2 + 1 >= l` and prints `r // 2` or `r % l`:

```python
t = int(input())
for _ in range(t):
    l, r = map(int, input().split())
    if r // 2 + 1 >= l:
        print(r // 2)
    else:
        print(r % l)
```

*Observation $o_1$* — the executor concatenates the failing public test's input, the expected `stdout` block, and the actual `stdout` block into one natural-language frame:

> Your code failed the following tests:

```
- input `4
1 1
999999999 1000000000
8 26
1 999999999
` failed:
Expected output `0
1
12
499999999
` but got `0
1
13
499999999
`

Give it another try.
```

*Revised code $a_1$* (turn 2) — switches to a brute-force scan, semantically correct but $O(r-l)$:

```python
t = int(input())
for _ in range(t):
    l, r = map(int, input().split())
    max_mod = 0
    for b in range(l, r + 1):
        a = r
        mod = a % b
        max_mod = max(max_mod, mod)
    print(max_mod)
```

*Observation $o_2$* — same template, now flagged as timeout instead of wrong-answer (no stderr because nothing crashed; just the sandbox's wall-clock verdict):

> Your code failed the following tests:

```
- input `4
1 1
[... truncated ...]
` failed:
Execution took too long

Give it another try.
```

*Final code $a_2$* (turn 3, submitted to private tests) — keeps the if/else skeleton from $a_0$ but adds the `b = (r // 2) + 1` step that fixes the wrong-answer case from $o_1$, in $O(1)$:

```python
t = int(input())
for _ in range(t):
    l, r = map(int, input().split())
    if r == l:
        print(0)
    else:
        b = (r // 2) + 1
        if b >= l:
            print(r % (b))
        else:
            print(r % l)
```

*Result:* passes public and private tests.

**Caveats:**
- The observation template (Appendix C.1) is a fixed string per failure mode: wrong answer → `Expected output \`...\` but got \`...\``; exception → `${stacktrace}`; timeout → `Execution took too long.`; OOM → `Out of memory.` Every failing public test gets one bullet, then a literal `Give it another try.` plus the formatting reminder. No LLM is in the loop — it's a deterministic Python f-string over the sandbox result.
- Only *public* tests appear in $o_t$. The terminal scalar reward comes from *private* tests the model never sees. Episode ends early on all-public-pass; the last $a_t$ is what gets scored.
- $T=3$ is fixed (one initial attempt + up to two repairs). KL to the SFT init is folded into the reward, not the loss ($\beta=0.05$).
- Light re-formatting of the initial prompt was applied "for readability" — the actual training-time prompts had less whitespace structure.


---
### [9] Shenfeld, Damani, Hübotter, Agrawal. "[Self-Distillation Enables Continual Learning](https://arxiv.org/abs/2601.19897)." 2026.

**Cluster:** Self-as-teacher distillation (with [1], [10]).

**Trained:** Student parameters $\theta$ of a single LLM $\pi_\theta$.

**External (not directly trained):** The teacher is the *same model* conditioned on a demonstration $c$ in its prompt. There are two orthogonal design choices: (a) the teacher's *parameters* — frozen base, current student, or an EMA copy of the student with $\phi \leftarrow \alpha\theta + (1-\alpha)\phi$, $\alpha \in \{0.01, 0.02, 0.05\}$; the paper ablates all three and EMA wins; (b) the teacher's *context* — same input $x$ as the student, plus the demonstration $c$ that the student does not see. The supervision signal is the gap between "self without demo" and "self with demo"; the parameter smoothing only stabilizes the moving target. Demonstration set $\mathcal{D}=\{(x_i, c_i)\}$ — a query and an in-context example or document — is the only external signal.

**Pipeline (per step):**
1. Sample $(x, c) \sim \mathcal{D}$.
2. **Student rollout:** $y \sim \pi_\theta(\cdot \mid x)$ — *no* demo in context.
3. **Teacher distribution:** evaluate $\pi_\phi(\cdot \mid x, c)$ token-wise on the same $y$ — EMA weights, demo *in context*.
4. Backprop reverse-KL; update $\theta$ via AdamW; update $\phi$ via EMA.

**Loss:**

$$\mathcal{L}(\theta) = \mathbb{E}_{y \sim \pi_\theta(\cdot \mid x)} \big[ D_{\mathrm{KL}}\big( \pi_\theta(\cdot \mid x) \,\|\, \pi_\phi(\cdot \mid x, c) \big) \big]$$

Computed analytically per token over the full vocabulary at each position along the on-policy trajectory.

**Why the model is its own teacher (ICL as supervision):** A demonstration in context shifts next-token probabilities toward task-correct behavior *without weight updates*. Distilling toward that ICL-shifted distribution is implicitly trust-region: capability moves toward the demo while staying close to the base model, so prior knowledge is preserved. SFT loses this because expert tokens can be arbitrarily off-policy from the model's manifold.

**Why EMA teacher (vs. frozen base or current student):** Frozen base never improves, so distillation plateaus once the student matches base+ICL. Using the *current* student causes instability — the target moves at every step. EMA smooths the target so it tracks the student's gains without thrashing.

**Why on-policy:** Sampling $y$ from $\pi_\theta$ keeps gradients on the student's own output manifold. The teacher only redirects toward the demo-conditioned distribution where the student's own trajectory was suboptimal.

**Why reverse KL:** Mode-seeking — the student concentrates on high-probability teacher modes rather than averaging across all continuations the demo-conditioned model entertains.

**Results (Qwen2.5-7B-Instruct unless noted):**

| Task | Input → Output | Metric | Base | SFT | SDFT | Δ vs SFT |
|---|---|---|---|---|---|---|
| Sci Q&A (SciKnowEval chem L-3) | question → answer | acc | 32.1 | 66.2 | **70.2** | +4.0 |
| Tool Use (ToolAlpaca) | query+API → call | regex acc | 42.9 | 63.2 | **70.6** | +7.4 |
| Medical (HuatuoGPT-o1, EN) | question → answer | GPT-5-mini judge | 30.1 | 35.5 | **40.2** | +4.7 |
| Knowledge (2025 disasters wiki) | question → fact | strict acc | 0 | 80 | **89** | +9 |
| " | " | OOD acc | 0 | 80 | **98** | +18 |
| Reasoning, answer-only — medical (Olmo-3-7B-Think) | problem → answer | acc | 31.2 | 23.5 | **43.7** | +20.2 |
| " | " | avg gen tokens | 4612 | 3273 | 4180 | — |

**Forgetting (avg across 6 retention benchmarks: HellaSwag, HumanEval, IFEval, MMLU, TruthfulQA, Winogrande):** Base 65.5 → SFT 53.4–60.2 → SDFT **64.5–65.4**. SDFT essentially closes the forgetting gap.

Compute: ~2.5× FLOPs, ~4× wall-clock vs. SFT.

**Key ablations:**
- *Teacher parameterization:* EMA > frozen base, EMA > current student (the latter is unstable).
- *Teacher context:* text+answer demo (89% strict) > text-only context (75%) — demonstration content matters, not just extra prompt material.
- *On-policy vs. offline distillation* from the same teacher: on-policy wins consistently.
- *Pass@k invariance:* SDFT's gain over base is uniform across $k\in[1, 128]$ → genuine new skill, not entropy collapse / mode sharpening of capabilities the base already had.
- *Scaling:* gap over SFT grows with model size — stronger ICL ⇒ stronger teacher ⇒ better distillation target.

**Surprising findings:**
- **Reasoners can be trained on answer-only data.** Olmo-3-7B-Think + SDFT on (problem, final-answer) pairs lifts accuracy 31.2 → 43.7 *and preserves long chain-of-thought* (4180 tokens). SFT on the same data *degrades* it to 23.5 with collapsed CoT (3273 tokens). The demo-conditioned teacher implicitly produces reasoning during student rollout, so the student is effectively distilled on its own elicited traces.
- **Knowledge OOD 80 → 98.** SFT memorizes; SDFT generalizes. CPT gets only 7% OOD. Because the on-policy student must *produce* the fact in its own phrasing rather than copy the training string, the fact is encoded as knowledge instead of a surface pattern.

**Relation to SDPO [1]:** Same group (Hübotter is on both), sequential arxiv IDs, same core mechanism — the model itself is the teacher when conditioned on extra context, the student when not. Where they differ:
- *Feedback type.* SDPO feeds *outcome feedback* $f$ (test results, error traces, a correct solution from the same batch) into the teacher's context — useful when ground truth or environment signal is available. SDFT feeds *demonstrations* $c$ (in-context examples or source documents) — useful when no verifier exists but worked examples do. The same dial: "what extra context makes the model better at this query?"
- *KL granularity.* SDPO computes top-K (=100) logit-level KL plus uses JSD for symmetry. SDFT uses full-vocabulary reverse KL. SDPO needs symmetrization because outcome feedback can shift the teacher far from the student; SDFT's demonstration-conditioned teacher stays closer, so vanilla reverse KL is stable.
- *Use case.* SDPO targets RLVR-style settings where rule-based rewards exist — the gain over GRPO is dense credit assignment. SDFT targets continual learning where the failure mode is forgetting, not credit assignment — the gain over SFT is the implicit trust region from ICL-conditioned supervision.

Together they argue for a unified picture: *in-context-augmented self* as a universal substitute for an external teacher or reward model. The "feedback" can be a test result (SDPO), a demo (SDFT), or any context that improves the model's next-token distribution — and on-policy reverse-KL distillation converts that conditional improvement into a weight-space update without forgetting.

<img src="/assets/images/distillation-textual-feedback/self-dist-enable-cont-learn-01.png" alt="Self-Distillation Enables Continual Learning" style="zoom:33%;" />

**Example interaction** (partly verbatim from §3 + §4.5 + Appendix B.3; the per-task instantiation is reconstructed — the paper does not publish a worked end-to-end example):

This example follows the §4.5 setup — Olmo-3-7B-Think distilled on the HuatuoGPT-o1 medical dataset using **answer-only** demonstrations $c$ (no chain-of-thought in the demo). The query $x$ and demo $c$ below are illustrative reconstructions in the format the paper describes; the teacher-context template and the shift narrative are anchored on verbatim text.

*Demonstration set element $(x, c)$ — illustrative, format per §4.5 ("contains no explicit chain-of-thought annotations" — each $(x,c)$ is a clinical-question / short-final-answer pair):*

```
x = "A 62-year-old man presents with sudden onset crushing
     substernal chest pain radiating to the left arm, diaphoresis,
     and ST-segment elevation in leads II, III, aVF. Which coronary
     artery is most likely occluded?"

c = "Right coronary artery."
```

*Student rollout $y \sim \pi_\theta(\cdot \mid x)$ (no demo in context) — reconstructed:*

> Let me think. ST elevation in II, III, aVF indicates an inferior wall MI. The inferior wall is supplied by [... truncated reasoning ...]. Therefore the answer is the **left circumflex artery**.

Base Olmo-3-7B-Think emits a long CoT (~4612 avg. tokens per §4.5 / Table 2) but lands on the wrong vessel here.

*Teacher context $\pi_\phi(\cdot \mid x, c)$ — verbatim template from §3:*

```
<Question>
This is an example for a response to the question:
<Demonstration>
Now answer with a response of your own, including the thinking process:
```

Filled in: `<Question>` ← $x$; `<Demonstration>` ← `"Right coronary artery."`. The demo is **answer-only** — the teacher must produce the reasoning chain itself, conditioned on knowing where it should land.

*Teacher distribution shift along the student's $y$ — reconstructed, per the §4.5 mechanism narrative:*

At the position where the student writes "the **inferior wall** is supplied by", the demo-conditioned teacher puts most of its mass on continuations like " the right coronary artery in roughly 80% of patients" or " the RCA, which gives off the posterior descending branch", while the student spreads mass over "the left circumflex" and "the RCA". At the final-answer position where the student commits to "left circumflex", the teacher concentrates on "right coronary" and hedges like "Let me reconsider — inferior MI most commonly involves the RCA". Reverse-KL pulls $\pi_\theta$ toward those teacher modes at every token along $y$.

*Reported outcome (Table 2, §4.5):* SDFT on this answer-only dataset lifts Olmo-3-7B-Think from 31.2% → **43.7%** accuracy *while preserving long CoT* (4180 avg. gen tokens). SFT on the identical $(x, c)$ pairs collapses CoT (3273 tokens) and drops accuracy to 23.5%.

**Caveats:**
- **The paper publishes no worked end-to-end example.** Only the high-level teacher-context template (the boxed block above, verbatim from §3) and dataset descriptions are given. The clinical question, the demo, and the student rollout are **plausible reconstructions** in the format §4.5 describes.
- **The teacher distribution shifts are reconstructed**, not empirical. The paper reports only aggregate KL (Fig. 2 right: SDFT teacher 0.68 nats vs SFT 1.26 nats from base) and final-task accuracy, not per-position logit shifts.
- **Demo format choice.** §4.5 reports "text+answer" demos (89% strict accuracy on the 2025-disasters task) > "text-only" context (75%). Answer-only is shown here because it makes the mechanism most striking — the teacher must hallucinate reasoning to bridge $x$ and a one-line $c$.
- **EMA teacher.** $\pi_\phi$ above is an EMA of $\theta$ with $\alpha \in \{0.01, 0.02, 0.05\}$ (§3); the demo-conditioned forward pass uses these smoothed weights, not $\theta$ itself.
- **Reasoning-trace preservation is the headline qualitative claim**, validated only via average generation length and accuracy — no published trace shows the student's CoT before vs. after SDFT.



---
### [10] Thinking Machines team. "[On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/)" 2026.

**Cluster:** Self-as-teacher distillation (with [1], [9]). The three form a tight family: on-policy student rollouts + per-token reverse KL to a teacher distribution. SDFT [9] is the closest mechanical sibling (full-vocabulary reverse KL, EMA-ish teacher, demonstration-conditioned). SDPO [1] swaps the demonstration for environment/outcome feedback. On-Policy Distillation generalizes the same recipe and applies it as a distillation tool for both math reasoning and continual learning.

*Note: the verification agent could not fetch this blog from the sandbox, so the numbers below were not re-checked against the source. The editorial commentary on `discount=0` and on `gradient drift` is the LITREV author's synthesis, not direct quotes — treat accordingly.*

#### Core Idea

Sample trajectories from the *student*, have the *teacher* score each token via reverse KL. Combines the on-policy relevance of RL with the dense reward signal of distillation.

| Method | Sampling | Reward | Bits/episode |
|---|---|---|---|
| SFT (off-policy distillation) | off-policy | dense | O(N) |
| Reinforcement learning | on-policy | sparse | O(1) |
| On-policy distillation | on-policy | dense | O(N) |


#### Loss Function

**Per-token reverse KL:**
```
KL(π_θ || π_teacher) = E_{x ~ π_θ} [log π_θ(x_{t+1}|x_{1..t}) − log π_teacher(x_{t+1}|x_{1..t})]
```
Advantage = `−reverse_KL`. Plugged into standard importance-sampling RL loss. Discount factor = 0.

The blog frames discount=0 as "less mathematically correct" by classical RL standards, where the return at step `t` should be the discounted sum of all future rewards — meaning early tokens would also be penalized for the downstream divergence they caused. But this framing is arguably the wrong lens. The cleaner way to think about it: the reward model here is simply the teacher's log-prob at each token, a dense local signal that directly answers "was this token a good choice given where we are?" Discount=1 would conflate that with "did bad things happen after this token?", smearing credit across the sequence in a way that muddies the signal. A token can be locally correct but followed by mistakes, or locally wrong but recoverable — per-token KL is the more principled reward definition, not a degenerate special case of a discounted sum. This is exactly the same choice made in dense-reward RL: when you have a shaped reward at every step (e.g. a process reward model), you don't accumulate it into a discounted return — you treat each step's reward as a local signal. Discount=1 only makes sense when rewards are sparse and a late signal needs to be propagated back to early decisions that caused it.

Properties of reverse KL: mode-seeking (learns one specific behavior), unhackable (low KL always corresponds to desirable teacher behavior), reduces exposure bias.

#### Experiment 1 — Math Reasoning

- **Student:** Qwen3-8B-Base
- **Teacher:** Qwen3-32B (log-probs only, no gradients through teacher)
- **Dataset (SFT init):** [OpenThoughts-3](https://huggingface.co/datasets/open-thoughts/OpenThoughts3-1.2M) — 400K prompts, responses generated by QwQ-32B
- **Benchmark:** AIME'24

Training the student (Qwen3-8B-Base) on 400k prompts with full fine-tuning achieves a score of 60% on AIME'24. LoRA lags behind full fine-tuning at large scale.

<img title="" src="/assets/images/distillation-textual-feedback/2026-04-15-12-32-37-image.png" alt="" width="583">

We treat the 400K SFT checkpoint as a starting point and compare post-training approaches to go from 60% → 70%:

| Method | AIME'24 | Teacher FLOPs | Student FLOPs | Cost vs SFT-2M |
|---|---|---|---|---|
| SFT-400K (init) | 60% | 8.5×10²⁰ | 3.8×10²⁰ | — |
| SFT-2M (extrapolated) | ~70% | 3.4×10²¹ | 1.5×10²¹ | 1× |
| Reinforcement learning | 68% | — | — | ≈1× |
| On-policy distillation | 70% | 8.4×10¹⁹ | 8.2×10¹⁹ | **9–30×** |

**Compute reduction breakdown:**
- **9×** when the SFT dataset already exists (teacher FLOPs for off-policy not counted)
- **18×** in GPU hours (teacher log-prob computation parallelizes cheaply)
- **30×** when including full cost of generating the off-policy dataset from scratch

#### Experiment 2 — Dense Supervision vs RL (Direct Comparison)

1. Start with Qwen3-8B-Base (no SFT).
2. Run RL on **DeepMath** (LoRA rank 128) → this becomes the teacher.
3. On-policy distill the RL-trained teacher back into the base model.

![](/assets/images/distillation-textual-feedback/2026-04-15-13-08-33-image.png)

**Result:** On-policy distillation matches teacher performance in ~7–10× fewer gradient steps → **~50–100× less total compute** (distillation works at shorter context lengths and smaller batch sizes, compounding the savings).

RL should be thought of as *search over semantic strategies*; once a strategy is found, distillation is a shortcut to learn it without replaying the entire RL curriculum.


#### Experiment 3 — Personalization / Continual Learning

- **Student:** Qwen3-8B (post-trained) → mid-trained on internal company documents
- **Teacher for distillation:** original Qwen3-8B (pre-midtrain)
- **Mid-train dataset:** internal docs + [Tulu3](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) prompts re-sampled from Qwen3-8B (on-policy background data as forwards-KL regularizer)
- **Distillation dataset:** Tulu3 prompts
- **Benchmarks:** Internal QA eval (knowledge), IF-eval (instruction following)

**Problem:** mid-training on new knowledge degrades post-training behavior (IF-eval). No mix ratio of doc/chat data fully preserves IF-eval. LoRA also insufficient.

**On-policy distillation recovers post-training behavior:**

After fine-tuning on a 70-30 mix of internal document data and chat data, on-policy distillation recovers nearly full IF-eval performance without losing knowledge:

| Model | Internal QA (Knowledge) | IF-eval (Chat) |
|---|---|---|
| Qwen3-8B | 18% | 85% |
| + midtrain 100% docs | 43% | 45% |
| + midtrain 70% docs | 36% | 79% |
| + midtrain 70% + distill | 41% | **83%** |

**Key insight for continual learning:** Alternate between phases of fine-tuning on new data and on-policy distillation to recover behavior. The earlier version of the model acts as the teacher to "re-invoke" lost capabilities.

Note: SFT on a model's own samples (KL=0 in expectation) still degrades performance because finite batches introduce gradient drift → the model goes off-policy over time. On-policy distillation always stays on-policy since the teacher is fixed.


#### Key Takeaways

- **Data efficiency:** Can train many epochs on even a single prompt (minimizes full distribution, not a single answer).
- **Continual learning:** On-policy distillation is the recommended tool for updating models without catastrophic forgetting.
- **Practical recipe:** mid-train for knowledge → on-policy distillation for behavior recovery → repeat.

**Example interaction** (verbatim from the blog's `example-kl-illustration.svg` figure and the surrounding paragraph; SimpleBench worked example used to illustrate per-token reverse-KL):

*Setup:* Student $\pi_\theta$ = Qwen3-4B-Instruct-2507. Teacher $\pi_{\text{teacher}}$ = Qwen3-235B-A22B-Instruct-2507. Task drawn from [SimpleBench](https://simple-bench.com/). Blog framing: "Below we see a real example of an incorrect student trajectory graded by a teacher [...] the correct answer is 'B. 0' since ice cubes will melt in a frying pan."

*Question $x$ (verbatim from the figure):*

```
Choose the most realistic answer. Beth places four whole ice cubes in a frying pan at
the start of the first minute, then five at the start of the second minute and some
more at the start of the third minute, but none in the fourth minute. If the average
number of ice cubes per minute placed in the pan while it was frying a crispy egg was
five, how many whole ice cubes can be found in the pan at the end of the third minute?

A. 30   B. 0   C. 20   D. 10   E. 11   F. 5
```

*Student trajectory $y \sim \pi_\theta(\cdot \mid x)$ (verbatim from figure; "(Start of assistant response truncated)" tag in original):*

> [... truncated ...]
> **Step 3: Total ice cubes in the pan at the end of the third minute.**
> This means all ice cubes placed by the end of minute 3 — minutes 1, 2, and 3.
> So: 4 (min 1) + 5 (min 2) + 11 (min 3) = **20**.
> Note: The ice cubes are not removed during the process — the pan is "frying a crispy egg", but we are only counting those that were added.
> Based on the context we assume that ice cubes are not melted or removed during the frying process — otherwise we would have to account for melting, which is not mentioned.
> Therefore the total number of ice cubes in the pan at the end of the third minute is: 4 + 5 + 11 = **20**.
> **Final Answer: C. 20.** `<|im_end|>`

*Per-token reverse-KL signal $\mathrm{KL}\big(\pi_\theta(\cdot \mid x, y_{<t}) \,\|\, \pi_{\text{teacher}}(\cdot \mid x, y_{<t})\big)$ visualized as red shading; darker = larger teacher-vs-student disagreement at position $t$. Highest-penalty tokens (verbatim from figure, "darker red" cells):*

- `"Note"` — start of the phrase that asserts ice cubes are not removed; teacher places much lower mass here (would instead open a reasoning step about melting).
- `"not removed"` / `"melted or removed"` — spans that lock in the wrong physical assumption.
- `"frying process"`, `"those that were added"`, `"after being added"` — phrase-initiating tokens that commit the student to the math-only frame.

Lighter / no penalty on the arithmetic itself (`4 + 5 + 11`, `**20**`) and on the final answer `**C. 20**` — once the wrong premise is set, the conclusion is "entirely predictable conditional on the whole preceding sequence" (blog quote), so the teacher assigns it high probability too and the KL is near zero.

*Resulting gradient:* advantage $= -\mathrm{KL}_t$ per token, discount $= 0$, plugged into the standard importance-sampling RL loss. Gradient mass concentrates on the "forking tokens" (the blog cites [Wang et al. 2025, *Beyond the 80/20 Rule*](https://arxiv.org/abs/2506.01939)) — the student is pushed to lower probability of starting phrases like `"Note: The ice cubes are not removed"`, not to memorize `"B. 0"`.

**Caveats:**
- This is the **only worked example** in the blog and it comes from the *motivating illustration* section (before the three experiments), not from the AIME / DeepMath-RL / continual-learning runs themselves. Those experiments report only aggregate benchmark numbers, no traces.
- The blog does **not** publish numerical per-token logprobs for either model. The relative ordering of "darker = higher KL" tokens above is read off the SVG's red-shading intensity rather than from numbers in the post — the *which-tokens-are-flagged* list is verbatim from the figure, the *underlying logprob deltas are not given*.
- Reverse-KL framing: advantage is $-\mathrm{KL}(\pi_\theta \,\|\, \pi_{\text{teacher}})$ (mode-seeking, student-sampled expectation), same direction as SDFT [9] but opposite to what standard SFT effectively optimizes.
- The student model in this illustration (Qwen3-4B-Instruct-2507 → Qwen3-235B-A22B) is a *different* pair than the three headline experiments (which all use Qwen3-8B-Base as student, Qwen3-32B or a DeepMath-RL-trained Qwen3-8B as teacher, or Qwen3-8B-pre-midtrain as teacher).
- Blog post (not a peer-reviewed paper) — no appendix, no released traces, no released code for the figure.


