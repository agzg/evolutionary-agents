![Cover](images/cover.png)

# Because Mom Said So: Priors in Evolutionary Agents

A multi-agent POMDP testbed in which LLM agents explore a noisy graph world, compress what they observe into short natural-language priors under a hard memory bottleneck, and pass those priors to their offspring. The system lets us study how knowledge, biases, and conventions form, transfer, and drift across generations of agents.

This repository began as a graduate agents-class final project (paper: `report.pdf`). This README documents what is here, and lays out a concrete plan to turn it into a viable submission for the COLM 2026 Workshop on Agent Behavior (https://www.aiagentbehavior.com/), including the literature connections that motivate the pivot.

---

## Table of Contents

1. [What this is](#1-what-this-is)
2. [Repository structure](#2-repository-structure)
3. [Installation and usage](#3-installation-and-usage)
4. [The core mechanism](#4-the-core-mechanism)
5. [What the current paper shows](#5-what-the-current-paper-shows)
6. [Honest review of the current state](#6-honest-review-of-the-current-state)
7. [Reframing for the Workshop on Agent Behavior](#7-reframing-for-the-workshop-on-agent-behavior)
8. [Concrete transformations and new experiments](#8-concrete-transformations-and-new-experiments)
9. [Literature: connections and expansions](#9-literature-connections-and-expansions)
10. [Expanding the context and memorization work](#10-expanding-the-context-and-memorization-work)
11. [Submission logistics and timeline](#11-submission-logistics-and-timeline)
12. [Limitations and risks](#12-limitations-and-risks)

---

## 1. What this is

The environment is a partially observable Markov decision process over a random geometric graph. A handful of nodes are doors. One door is the goal. Each door emits signals: hints that are internally consistent and truthfully point at the goal (its colour, region, nearby landmarks), and distractors that contradict each other and point at wrong answers. The agent never sees a label telling it which is which. To succeed it must learn that consistent signals are trustworthy and contradictory ones are noise.

Agents run an observe, reason, act loop with a two-tier LLM setup. A rolling 750-token context buffer holds recent (observation, action, reasoning) entries; when it overflows, the oldest half is summarized, which creates a recency gradient and a lossy compression pressure. When an agent reproduces, its recent experience is compressed into a prior of at most 150 words and handed to a child, which starts fresh at the same node with only that prior. Performance differences between generations therefore come almost entirely from the content of the inherited prior.

The scientific framing of the original report is cumulative cultural evolution: agents act like a transmission chain that distills heuristics, develops shorthand conventions, and occasionally self-corrects bad inherited information.

## 2. Repository structure

```
src/
  config.py          TrialConfig dataclass, all experiment knobs
  environment.py     RGG construction, door theming, hint and distractor generation, observation
  agent.py           LLM agent, context buffer, summarization, prior compression, parent Q and A
  beliefs.py         Optional Bayesian belief tracker over door identity
  reproduction.py    Birth logic, lineage tracking, birth-event records
  runner.py          Trial execution loop, reproduction triggers, logging
  metrics.py         Text similarity, drift, signal precision and recall, belief and repro stats
  skill_library.py   Voyager-style shared convention store with categories and dedup
  cloaking.py        Potential-theory active cloaking overlay (DTN operator on the graph Laplacian)
  logger.py          Full transcript logging
experiments/
  a_prior_ablation.py     Experiment A: inherited prior vs blank slate
  b_parent_interaction.py Experiment B: prior plus multi-step parent dialogue (viability, weak)
  c_lexical_shortcuts.py  Experiment C: convention and lexical drift across three scales
  e_skill_library.py      Experiment E: shared skill library on top of priors
  h_cloaked_goals.py      Experiment H: cloaked goals (intervention on the observation, currently future work)
  i_fertility.py          Experiment I: fertility ablation, the central experiment
run_experiments.py   CLI entry point
misc/prompts.md      Every LLM prompt in the system, documented
lit/                 Background papers (see Section 9)
report.pdf           The current writeup
results/             Output, gitignored
```

## 3. Installation and usage

The code calls models through `langchain-dartmouth`. Set up your Dartmouth Chat API key and LangChain credentials (via developer.dartmouth.edu) in a `.env` file. An example file is provided. Note that `.env.example` currently lists `ANTHROPIC_API_KEY`; the actual runtime uses the Dartmouth provider configured in `src/config.py` (reasoning model `openai.gpt-4.1-mini`, utility model `vertex_ai.gemini-2.0-flash-001`). Reconcile these before a run.

```bash
uv sync
uv run python run_experiments.py --exp all --trials 50   # viability sweep
uv run python run_experiments.py --exp a --trials 250     # main prior ablation
uv run python run_experiments.py --exp i --trials 50      # fertility ablation
```

Useful flags: `--max-steps`, `--reasoning-model`, `--utility-model`, `--output`, `--seed`. If a `models.txt` file is present at the repo root, model names are validated against it; if absent, validation is skipped. Results land in `results/exp_<id>/` as `results.json`, `summary.json`, `report.txt`, plus full transcripts.

## 4. The core mechanism

The single most important loop to understand is in `src/runner.py` and `src/agent.py`:

1. The agent observes its node, its neighbours, and a random subset of nearby door signals.
2. The reasoning model picks a move; the (observation, action, reasoning) triple is appended to context.
3. When context exceeds 80 percent of 750 tokens, the utility model compresses the oldest half into a 2 to 3 sentence summary.
4. On a reproduction trigger (periodic, on-success, or novelty), the reasoning model compresses the last 12 entries into a prior of at most 150 words.
5. The child inherits the prior via its system prompt and starts with an empty context buffer.

Three optional modules layer on top: a Bayesian belief tracker over door identity (`beliefs.py`), a shared convention store (`skill_library.py`), and a potential-theory cloaking overlay that attenuates real hints far from the goal and replaces them with distractors (`cloaking.py`).

## 5. What the current paper shows

From `report.pdf`, across 250-trial experiments:

- Prior inheritance reaches 93 percent success versus 78 percent for memoryless agents, with median steps dropping from 34.5 to 12.0.
- On the hardest graph instance a third-generation agent solved the task in 5 steps while a full-state oracle took 98, because the inherited prior encoded experiential route knowledge rather than coordinates (the grandchild phenomenon).
- Fertility, defined as the mean number of reproductive events per agent, has a sweet spot. Novelty-triggered reproduction at Jaccard threshold 0.7 generalizes to a harder unseen graph (degrading only 37 percent) where fixed-interval strategies degrade by up to 180 percent.
- Lineages spontaneously develop stable naming conventions whose drift increases with environment complexity, and compression bottlenecks sometimes act as error correction (a child overrode a wrong inherited target with "URGENT: inherited target is WRONG. DISCARD.").

## 6. Honest review of the current state

Strengths. The codebase is clean, modular, and reproducible with fixed seeds. The environment is a genuinely controllable knob-rich testbed. The fertility result is novel and crisp. The cultural-evolution framing is coherent and well cited. There is a working but unused intervention engine (cloaking) that is more valuable than the report treats it.

Gaps for a behavior venue. The paper is framed and measured as a capability study: the headline numbers are success rate and steps to goal. The Workshop on Agent Behavior is explicitly about how agents behave rather than what they achieve. The current metrics do not directly answer behavioral questions such as which kinds of misleading signals move an agent, how susceptible the agent is relative to a baseline, whether a memorized signal becomes a heritable bias, or how interventions reliably steer behavior. Experiment B (parent dialogue) was inconclusive, Experiment H (cloaking) has no results, and the Bayesian module is implemented but underused. The good news is that the existing machinery already produces the raw material for all of these behavioral measures; what is missing is the reframe and a handful of new metrics on top of data the system already logs.

## 7. Reframing for the Workshop on Agent Behavior

The workshop, organized in part by Nikhil Singh, Manuel Cherep, and Pattie Maes, advances the scientific study of how agents do and should behave. Two of the papers already in `lit/` are the organizers' own: ABXLAB (`agent-behavior.pdf`) and the hypersensitivity study (`hypersensitive.pdf`). Aligning with that line of work is the highest-leverage move available, and it requires almost no new infrastructure.

The central reframe in one sentence: distractors are nudges, and an inherited prior is a behavioral intervention that changes how susceptible an agent is to those nudges across generations.

Why this is a natural fit:

- ABXLAB formalizes an environment with an intervention set I of functions that alter an observation before the agent sees it, and measures how much agent choices shift in response. This project's distractor injection and cloaking overlay are exactly such functions on the observation. The environment is already a man-in-the-middle behavioral testbed; it just has not been described or measured as one.
- ABXLAB and the hypersensitivity paper find that agents are strongly biased choosers, that they are hypersensitive to nudges relative to humans, and that prompt-level fixes (chain of thought, few-shot) do not resolve that sensitivity. This project can contribute a complementary, positive result: that cultural transmission of distilled priors is an intervention that does reduce susceptibility, where in-context strategies alone do not. That is a genuinely new claim in their framework.
- The flip side is equally publishable and arguably more striking: a misleading signal that gets memorized into a prior becomes a heritable bias that steers every descendant. ABXLAB warns that agentic consumers may inherit and amplify human biases; this testbed can demonstrate multi-generational bias amplification mechanistically, and connect it to the iterated-learning bias-amplification literature (Section 9).

Mapping to the five workshop topics:

1. Behavioral evaluations. Replace success-centric reporting with a behavioral test suite over the distractor taxonomy already in `environment.py` (spatial, colour, relational, narrative, pattern). Measure per-type susceptibility, the behavioral analog of ABXLAB's percentage-point choice shifts.
2. Agentic interactions. The parent-child transmission, convention formation, and self-correcting priors are social and relational behaviors. Frame them as transmission-chain dynamics with cultural attractors.
3. Interventions. The project has three distinct steering mechanisms: prior inheritance (in-context steering), cloaking (environment design on the observation), and the skill library and memory bottleneck (structural constraints). The workshop explicitly asks for post-training, environment design, and structural constraints.
4. Social and ethical implications. Heritable bias, inherited misinformation, and the question of when distilled priors help versus entrench errors map onto accountability for delegated decisions.
5. Behavior foundation models. Emergent conventions and lexical drift are behaviors that arise and propagate without central coordination.

There is also a benchmark proposal track (1 to 2 pages, proposal only, no results needed). The RGG hint-versus-distractor world is a clean, procedurally generated, fully controllable benchmark for epistemic robustness and susceptibility to misleading context. This is a low-risk parallel option (Section 11).

## 8. Concrete transformations and new experiments

Each item below is scoped so it reuses the existing simulator and logs. Effort tags are rough.

T1. Susceptibility metric per distractor type (low effort, high value).
The environment already tags every signal as hint or distractor and assigns a `signal_type`. Add a per-step record of which signals were present and which way the agent moved, then define susceptibility as the probability that the agent's action follows a distractor's recommendation given that the distractor was visible, decomposed by signal type. This produces a table directly comparable in spirit to ABXLAB Table 2: a behavioral fingerprint of the agent showing, for example, that it is more swayed by authoritative narrative distractors than by pattern distractors. No new LLM calls beyond what already runs.

T2. Priors as a de-biasing intervention (medium effort, this is the headline).
Run the susceptibility metric from T1 across generations and conditions: no prior, blank prior, random prior, inherited prior, and abstract cross-world prior (all already supported by `config.py` and the Experiment A and H scaffolding). The claim to test: inherited priors lower susceptibility to distractors over generations. Contrast against the hypersensitivity finding that chain of thought and few-shot do not fix nudge sensitivity, by adding a chain-of-thought-only control. A clean positive result here is the paper.

T3. Heritable bias and inherited misinformation (medium effort, most striking).
Deliberately inject a single persuasive distractor into a parent's observation stream (an intervention function in the ABXLAB sense), measure whether it enters the compressed prior, and then measure how many descendants inherit the wrong belief and how many steps it costs them. This demonstrates multi-generational bias amplification and ties directly to iterated-learning theory (Ren et al., bias amplification) and to the self-correction phenomenon already observed. The compression bottleneck becomes the variable that determines whether bias is amplified or filtered.

T4. Cloaking as a parametric intervention engine (medium effort, revives dead code).
Promote `cloaking.py` from future work to a centerpiece. The cloak radius is a continuous intervention strength that controls how much real information reaches the agent. Sweep it and plot susceptibility and reliance on inherited route knowledge as a function of intervention strength. Frame the cloak formally as a member of the ABXLAB intervention set I that maps observations to observations.

T5. Memory faithfulness analysis (medium effort, connects context and memorization).
Borrow ABXLAB Appendix D directly: use an LLM-as-judge to analyze what the 750-token context and the compressed prior actually retain. Measure whether the agent's memory faithfully represents the true signal structure or over-retains distractors, and whether the prior preserves generalizable strategy versus memorized graph-specific routes. This is the bridge to the memorization story in Section 10.

T6. Optimal and adversarial priors (higher effort, optional).
The hypersensitivity paper derives optimal nudges from a resource-rational model. By symmetry, search for the prior text that maximally steers a child's behavior, in either a helpful or an adversarial direction. This frames priors as a tunable control signal and connects to the moral-alignment and reward-shaping work in `lit/`.

Suggested minimum viable paper: T1 plus T2 plus T3, reusing the Experiment A, H, and I harnesses, reported as a behavioral study with susceptibility as the primary outcome and the fertility result kept as a secondary contribution about when transmission helps.

## 9. Literature: connections and expansions

### 9.1 Papers already in `lit/`

- ABXLAB, A Framework for Studying AI Agent Behavior (`agent-behavior.pdf`, Cherep, Ma, Xu, Shaked, Maes, Singh, ICLR 2026). A man-in-the-middle framework that intercepts and modifies web content with an intervention set, then measures how agent choices shift. Agents are shown to be strongly biased choosers, more susceptible than humans, with rule-like switching under explicit preferences. This is the template to imitate: formalize the environment as states, actions, observations, transitions, and interventions, and report behavior shifts rather than only task success.
- LLM Agents Are Hypersensitive to Nudges (`hypersensitive.pdf`, Cherep, Maes, Singh). Canonical nudges (default, suggestions, highlighting, optimal) move agent choices far more than human choices, and chain of thought and few-shot do not fix it. This is the gap this project can address with a transmission-based intervention.
- Moral Alignment for LLM Agents (`llm-morality.pdf`, Tennant, Hailes, Musolesi, ICLR 2025). Intrinsic-reward fine-tuning to encode explicit values in an iterated prisoner's dilemma, with generalization across games. Relevant to T6 and to framing priors as transparent, explicit value or strategy carriers rather than opaque preference data.
- Reinforcement World Model Learning for LLM-based Agents (`rl for llm.pdf`, Yu, Peng, ... Singh et al.). Self-supervised world-model learning aligning predicted next states to realized ones. Relevant to the memorization story: a prior that encodes route knowledge is a compressed, transmissible world model, and the memorization-versus-generalization tension in T5 mirrors token-level fidelity versus semantic equivalence here.

### 9.2 Cultural evolution and iterated learning (new, from the web)

These directly underpin the report's framing and give it theoretical teeth:

- Ren et al., Bias Amplification in Language Model Evolution: An Iterated Learning Perspective (https://openreview.net/pdf?id=BSYn7ah4KX). A Bayesian iterated-learning account with proven monotonic bias amplification across generations. This is the formal backbone for T3.
- When LLMs Play the Telephone Game: Cultural Attractors in Multi-turn Settings (ICLR 2025, https://proceedings.iclr.cc/paper_files/paper/2025/hash/dbdea7859f1d2fc10f2c9e79b8f5ae54-Abstract-Conference.html). Transmission chains converge to attractor states; open-ended instructions attract more strongly. The convention-drift result in Experiment C is an instance of this; reframe drift as attraction.
- Cultural evolution in populations of LLMs (https://arxiv.org/html/2403.08882v1). Punctuated equilibria and attractors in transmission chains over network topologies, including chain and fully connected graphs. Directly relevant since this project already runs lineages over a graph.
- LLMs show human-like content biases in transmission chain experiments (PNAS, https://www.pnas.org/doi/10.1073/pnas.2313790120). Content biases survive serial reproduction. Motivates checking which distractor content types preferentially survive into priors (a content-bias version of T1).
- Model Collapse as Cultural Evolution (https://arxiv.org/html/2605.23054). Compression pressure from a bottleneck plus communication pressure are needed for structure; compression alone degenerates. This is precisely the 750-token bottleneck plus the need to be useful to offspring, and it predicts when conventions become degenerate.

### 9.3 Memory and distractor susceptibility (new, from the web)

These connect the project to current agent-behavior evaluation and to the context and memorization theme:

- Lost in the Noise: How Reasoning Models Fail with Contextual Distractors, and the NoisyBench benchmark (https://arxiv.org/pdf/2601.07226). Up to 80 percent performance drops from contextual distractors, distractors can trigger emergent misalignment, naive prompting and context engineering do not help, and agentic workflows amplify errors by over-trusting context. Rationale-aware rewards help. This is the strongest contemporary anchor for the susceptibility framing and the claim that a transmission-based intervention is worth studying.
- MemoryAgentBench, Evaluating Memory in LLM Agents (https://openreview.net/forum?id=DT7JyQC3MR). Four competencies: accurate retrieval, test-time learning, long-range understanding, and selective forgetting. The summarize-and-inherit pipeline here is a memory system that can be scored on exactly these axes.
- MemFail, Stress-Testing Failure Modes of LLM Memory Systems (https://ar5iv.labs.arxiv.org/html/2605.26667). Formalizes memory as summarization, storage, and retrieval, and isolates failure modes of each. The agent's pipeline (summarize oldest half, store in buffer, retrieve last six entries, compress to prior) maps cleanly onto this taxonomy, so the project can report which operation introduces or filters bias.
- Context rot (Chroma 2025, summarized at https://atlan.com/know/llm-context-window-limitations/). Semantically similar but irrelevant content actively misleads models, and effects compound with multiple distractors. Supports the design choice of consistent hints versus contradictory distractors and the recency-gradient memory.

## 10. Expanding the context and memorization work

The 750-token bottleneck and prior compression are, at bottom, a study of what an agent chooses to remember and how that choice shapes downstream behavior. Five concrete expansions, ordered from cheapest to most ambitious:

1. Memorization versus generalization in priors. Compare the specific prior prompt against the abstract cross-world prior prompt (both already in `agent.py`). Measure transfer to held-out graphs. The hypothesis: specific priors memorize route knowledge that wins on the training graph (the grandchild phenomenon) but fails to transfer, while abstract priors carry strategy that transfers. This is a clean memorization-versus-generalization curve, and it reuses the Experiment H cross-world machinery.

2. The bottleneck as a controllable variable. Sweep `max_context_tokens` and `max_prior_tokens`. Connect to Model Collapse as Cultural Evolution: too little compression preserves noise, too much produces degenerate conventions. Plot convention stability, susceptibility, and transfer against the bottleneck size to locate the structure-forming regime.

3. Memory faithfulness audit (T5 above). Use an LLM-as-judge, as in ABXLAB Appendix D, to label what the context and the prior retain relative to the ground-truth signal set. Report precision and recall of hints versus distractors in memory over time, and whether summarization (the MemFail summarize operation) is where distractors leak in.

4. Heritable memorized bias (T3 above). Show that a memorized distractor is not a one-step error but a persistent, inheritable state. This is the memorization angle on ABXLAB's bias-inheritance warning, made mechanistic by the lineage structure.

5. Choice architecture of memory. The summarization and prior-compression prompts are themselves a choice architecture imposed on the agent's own memory. Treat the compression prompt as an intervention and show that editing it changes descendant behavior. This unifies the workshop's intervention topic with the context and memorization theme: environment design applied inward, to memory rather than to the world.

Taken together, these turn an implementation detail (a token budget) into the paper's mechanism: memory compression is the lever that determines whether an agent lineage learns robust strategy or inherits and amplifies bias.

## 11. Submission logistics and timeline

The workshop call (https://www.aiagentbehavior.com/) lists an OpenReview submission deadline of June 23, 2026 (Anywhere on Earth), double-blind, non-archival, with preprints and concurrent submissions explicitly encouraged. Two tracks are relevant:

- Papers, 4 to 9 pages in the COLM template. Realistic given that the simulator, logs, and one strong result (fertility) already exist. The fastest credible path is T1 plus T2 plus T3 on top of the existing Experiment A, H, and I runs, reframed with susceptibility as the primary outcome.
- Benchmarks, 1 to 2 pages, proposal only, no implementation or results required, using the provided LaTeX template. The RGG hint-versus-distractor world is a ready-made proposal for a controllable epistemic-robustness and susceptibility benchmark. Accepted benchmarks receive credits, a harness, and support toward an open-source version and a shared evaluation suite.

Given the date on the call, the benchmark proposal is the lowest-risk option and can be drafted from this repository alone. If pursuing the paper track, prioritize T1 and T2, which need new metrics rather than new infrastructure, and keep T3 as the striking result if time permits.

## 12. Limitations and risks

- The reframe must be more than relabeling. Susceptibility has to be measured and reported as the primary outcome, not appended to a success-rate paper.
- Cost. New per-condition runs add LLM calls; T1 and the faithfulness audit are the cheapest because they mostly post-process existing logs.
- Statistical care. ABXLAB sets a high bar with cluster-robust inference; even at workshop scale, report effect sizes with proper baselines (no prior, blank prior, random prior, chain of thought) and a clear susceptibility definition.
- Non-archival venue. This does not preclude a later full paper, which is an argument for investing in the benchmark and the susceptibility metric now, since both are reusable.
- Provider mismatch. Reconcile `.env.example` (Anthropic key) with the Dartmouth provider actually used in `config.py` before any reviewer or collaborator tries to run the code.

---

### Citation

Original report: Sikder, R. and Azam, A. Because Mom Said So: Priors in Evolutionary Agents. Dartmouth College. See `report.pdf`.
