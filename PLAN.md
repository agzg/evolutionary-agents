# Plan: From a Cultural-Evolution Capability Study to an Agent-Behavior Paper

This document has two parts. Part A gives detailed summaries of the papers that matter for the pivot. Part B is the major change plan: what to keep, add, and cut across the framing, the formal model, the codebase, and the ablations. It also answers two specific questions: whether to keep the POMDP, and what to say about the behavioral aspect.

Target venue: COLM 2026 Workshop on Agent Behavior. Deadline June 23, 2026 (AoE), non-archival, 4 to 9 pages, plus a separate 1 to 2 page benchmark proposal track.

---

## Part A. Detailed paper summaries

### A.1 ABXLAB: A Framework for Studying AI Agent Behavior (Cherep, Ma, Xu, Shaked, Maes, Singh; ICLR 2026) [local: lit/agent-behavior.pdf]

What it is. A man-in-the-middle framework that turns any website into a controllable behavioral testbed for LLM agents. It formalizes the environment as a tuple of state, action, and observation spaces, a deterministic transition function, and an intervention set I, where each intervention is a function that rewrites an observation before the agent sees it. The agent acts on the modified observation, so the experimenter has clean causal control over what cue the agent is exposed to.

Setup. A realistic shopping environment (OneStopMarket via WebArena and AgentLab). A binary forced choice (2AFC): the agent sees two product pages in two tabs and must add the better one to the cart. The action space is nine web actions; the observation is pruned HTML; the agent keeps an explicit chain-of-thought and a short-term memory stream. Episodes cap at 10 actions.

Interventions. Five nudge categories injected as text under a product title: authority (expert or Wirecutter recommended), social proof (best seller, 50,000 customers), scarcity (limited edition, buy in the next hour), negative framing (newer version available, final sale), and incentives (free shipping, buy one get one free). Three conditions: no nudge, first product nudged, second product nudged. Attribute-matching regimes: original, matched ratings (MR), matched ratings and prices (MRaP). They also test explicit user-preference profiles.

Scale and analysis. 17 models, over 80,000 trials, around 2.5B tokens. Effects estimated with linear probability models, trial fixed effects, cluster-robust standard errors by nudge text and category, Benjamini-Hochberg correction. A multinomial logit robustness check correlates at r around 0.93. They also collect a human baseline (30 Prolific participants, 50 decisions each).

Key findings.
- Agents are strongly biased choosers. Ratings shifted choice by 30 to 80 percentage points across most models; some near-deterministic (o4-mini at 81 pp).
- Price effects strong and intensify when ratings are matched (Llama 4 Maverick at 93 pp toward cheaper), and largely vanish when both ratings and prices are matched, showing agents respond to the actual attributes, not correlates.
- Order effects are brittle and heterogeneous: GPT-4.1 Nano had a 90 pp preference for the first item, Claude 3.5 Haiku a 35 pp penalty against it.
- Nudges shift choice by 10 to 60 pp even when price and rating are matched. Text-level wording matters: the Wirecutter authority nudge was strongest.
- Humans in the same task showed modest effects (around 4 to 10 pp), so agents are 3 to 10 times more susceptible.
- User profiles act like categorical switches: declaring one preference suppresses competing attributes almost entirely.
- An LLM-as-judge analysis of the agents' thought and memory streams (Appendix D) checks which attributes agents mention, with the caveat that stated reasoning may not be faithful.

Why it matters here. This is the template to imitate. The lessons are: (1) add an intervention set I that rewrites observations, (2) measure behavior shift in percentage points with a proper baseline, (3) prefer a clean forced choice to maximize causal identification, (4) decompose by intervention type and even by exact wording, (5) audit the memory and reasoning streams with an LLM judge.

### A.2 LLM Agents Are Hypersensitive to Nudges (Cherep, Maes, Singh) [local: lit/hypersensitive.pdf]

What it is. A controlled case study porting a human resource-rational decision task (Callaway et al. 2023) to LLMs. Agents pick among baskets of hidden cells, paying a cost to reveal information, trying to maximize reward minus reveal cost. This is a meta-level decision task: deciding how to decide.

Nudges studied. Four canonical choice architectures: default option (accept or decline a pre-selected basket), suggested alternatives (early or late), information highlighting (cheaper to reveal one prize), and an optimal nudge derived by optimizing the resource-rational model. Conditions: base, zero-shot chain of thought, and few-shot with real human game traces. Eight models, around 1B tokens, temperature 0.2.

Key findings.
- Three behavioral axes are measured, all independent of raw success: information acquisition strategy (how many cells revealed before deciding), net earnings, and nudge-following rate.
- Information acquisition diverges from humans (two-sample KS tests, D often 0.4 to 0.98). GPT-3.5 Turbo often chooses without revealing anything; GPT-4o Mini over-reveals at high cost in multiples of five (simplistic row or column strategies). Stronger models with few-shot get closer to humans.
- Hypersensitivity: models follow the nudged option far more often than humans do, even when it is suboptimal.
- Crucially, chain of thought has minimal effect and few-shot only partially helps; neither resolves nudge sensitivity. This is the gap.
- An optimal nudge designed to help humans also raises performance for some models.

Why it matters here. It establishes that prompt-level interventions (CoT, few-shot) do not fix susceptibility. That is exactly the opening for a transmission-based intervention: show that inherited, distilled priors reduce susceptibility where CoT does not. It also gives a menu of behavioral metrics to copy: information acquisition (exploration before commit), divergence from a normative or human baseline via KS tests, and nudge-following rate.

### A.3 Moral Alignment for LLM Agents (Tennant, Hailes, Musolesi; ICLR 2025) [local: lit/llm-morality.pdf]

What it is. Instead of RLHF on opaque human preferences, they fine-tune LLM agents with explicit intrinsic rewards that encode moral values, evaluated in the Iterated Prisoner's Dilemma. They use Deontological rewards (for example, do not defect against a cooperator) and Utilitarian rewards (maximize collective payoff). They use a small model (Gemma2-2b-it) and PPO with a KL penalty.

Key findings. Agents learn aligned strategies; a previously selfish strategy can be unlearned (re-prioritized); and some moral strategies generalize to other matrix games. They deliberately obscure the game (action1 and action2 instead of cooperate and defect) so the model uses general decision-making rather than memorized Prisoner's Dilemma responses, and they test transfer across environments.

Why it matters here. Two transferable ideas. First, an explicit, transparent, transmissible value or strategy carrier (their intrinsic reward) is analogous to the natural-language prior in this project: both are explicit steering signals rather than opaque weights. Second, the memorization-versus-generalization test (obscuring the task, then checking transfer to new games) is directly reusable for testing whether a distilled prior carries general strategy or memorized specifics.

### A.4 Reinforcement World Model Learning for LLM-based Agents (Yu, Peng, Xu, Shen, He, Nath, Singh, Gao, Yu) [local: lit/rl for llm.pdf]

What it is. A self-supervised RL method (RWML) that trains an LLM agent to predict the next state given an action and history, rewarding semantic similarity between predicted and realized next states in an embedding space (rather than token-level next-state prediction, which causes model collapse). Uses GRPO, a binary similarity reward, and subsamples easy examples. Evaluated on ALFWorld and tau2-bench.

Key findings. World-model pretraining improves base agents by large margins with no expert data or task-success reward, and stacks with task-success RL. Token-level fidelity (SFT) is worse than semantic-similarity reward, which resists reward hacking.

Why it matters here. It frames a prior as a compressed, transmissible world model. The token-fidelity-versus-semantic-equivalence distinction is exactly the memorization-versus-generalization tension in the prior compression step: a prior that memorizes exact routes (token-fidelity analog) versus one that captures transferable structure (semantic analog). It also legitimizes embedding-based similarity as a metric for comparing what priors encode.

### A.5 Cultural evolution and iterated learning (web)

- Ren et al., Bias Amplification in Language Model Evolution: An Iterated Learning Perspective (openreview BSYn7ah4KX). A Bayesian iterated-learning framework with a proof of monotonic bias amplification across generations of LLMs. This is the formal backbone for the heritable-bias claim: a small prior bias, when transmitted and re-distilled, grows.
- When LLMs Play the Telephone Game (ICLR 2025). Transmission chains of LLMs converge to cultural attractor states; open-ended instructions attract more strongly than constrained ones; different text properties (toxicity, length) have different attraction strength. Reframes convention drift as attraction toward fixed points.
- Cultural evolution in populations of LLMs (arXiv 2403.08882). Transmission chains over network topologies (chain, fully connected, and others) show punctuated equilibria and attractors. Directly relevant because this project already runs lineages over a graph.
- LLMs show human-like content biases in transmission chain experiments (PNAS). Some content types (negative, social, threat-related, stereotype-consistent) survive serial reproduction preferentially. Motivates measuring which distractor content types preferentially survive into priors.
- Model Collapse as Cultural Evolution (arXiv 2605.23054). Structure emerges only with both compression pressure (a learning bottleneck) and communication pressure (the need to be expressive); compression alone degenerates. This is precisely the 750-token bottleneck plus the need for a prior to be useful to offspring, and predicts when conventions become degenerate.

### A.6 Memory and distractor susceptibility (web)

- Lost in the Noise and NoisyBench (arXiv 2601.07226). Up to 80 percent performance drops from contextual distractors; distractors can trigger emergent misalignment even without adversarial intent; naive prompting and context engineering do not help; agentic workflows amplify errors by over-trusting context; rationale-aware rewards help. The strongest contemporary anchor for the susceptibility framing.
- MemoryAgentBench (openreview DT7JyQC3MR). Four memory competencies: accurate retrieval, test-time learning, long-range understanding, and selective forgetting. The summarize-and-inherit pipeline can be scored on these axes.
- MemFail (arXiv 2605.26667). Formalizes memory as summarization, storage, and retrieval, and isolates failure modes of each. The agent pipeline (summarize oldest half, store in buffer, retrieve last six, compress to prior) maps onto this taxonomy, so the project can attribute where bias enters or gets filtered.
- Context rot (Chroma 2025). Semantically similar but irrelevant content actively misleads models and compounds with multiple distractors. Supports the consistent-hint versus contradictory-distractor design and the recency-gradient memory.

---

## Part B. The major change plan

### B.0 One-line repositioning

From "inherited natural-language priors help agents navigate a noisy POMDP (cultural evolution)" to "a controllable testbed for measuring how LLM agents behave under misleading information, and how memory and generational transmission change that behavior, including reducing susceptibility and, conversely, amplifying heritable bias."

Primary outcome variable changes from success rate and steps to susceptibility (probability the agent follows a misleading signal), reported in percentage points against explicit baselines.

### B.1 Should you keep the POMDP?

Short answer: keep it, but for a sharper reason than navigation, and add a controlled probe layer on top.

The honest tension. For a pure behavioral-science result, a long-horizon navigation POMDP is a liability. It injects confounds (graph topology, path length, exploration luck) that add variance and make it hard to attribute a behavior shift to a specific cue. ABXLAB deliberately strips all that away and uses a single binary forced choice precisely to get clean causal identification of each intervention.

Why the POMDP still earns its place. The thing that distinguishes this project from ABXLAB and the hypersensitivity work is memory and transmission. A single-shot forced choice cannot study any of the questions that make this project interesting: there is no sequential memory pressure, no compression bottleneck, and no parent-to-child transfer. The POMDP plus the bounded context buffer is the minimal substrate in which compression of experience is forced, and only because compression is forced can you study how memory shapes behavior and how bias is inherited. So the principled justification is: partial observability over a horizon is what creates the compression pressure that is the paper's actual subject. State that explicitly. Do not justify the POMDP as a navigation challenge; justify it as a memory-and-transmission generator.

The concrete fix: add behavioral probes inside the POMDP. At controlled decision points, present the agent with a clean local choice in which exactly one factor is manipulated (for example, inject a single authority-style distractor pointing at a wrong door, holding everything else fixed), and log the counterfactual decision. This gives ABXLAB-style percentage-point susceptibility measurement on top of the richer, ecologically valid environment. You keep the memory story and gain causal cleanliness.

Also reduce confounds you do not need: continue using identical graph instances per condition per trial (already done), fix start and goal across matched conditions where possible, and report effects within trial (trial fixed effects), exactly as ABXLAB does.

If a reviewer pushes back, the fallback is a minimal mode: a degenerate one-shot version of the same environment (one observation, a forced door choice) used only for the cleanest susceptibility numbers, with the full POMDP used for the memory and transmission results. You can offer both.

### B.2 What to talk more about: the behavioral aspect

Right now the paper measures task outcomes. A behavior paper should measure and discuss the following, all of which the simulator can already produce or can produce with small additions.

1. Susceptibility by manipulation type. Map the existing distractor taxonomy onto choice-architecture categories and report per-type effect sizes:
   - spatial and color distractors behave like direct false claims,
   - narrative distractors ("a trustworthy guide left a note") behave like authority or social proof,
   - alarmist or urgent distractors behave like scarcity or negative framing,
   - pattern distractors ("the first door is never right") behave like injected pseudo-rules.
   This is the behavioral fingerprint, the analog of ABXLAB Table 2.

2. Brittleness to irrelevant variation. Does shuffling signal order or rephrasing a hint (same content, different wording) change the decision? ABXLAB found order and wording effects. Measuring decision flips under content-preserving perturbations is a clean behavioral robustness result.

3. Information acquisition and decision horizon. Copy the hypersensitivity metric: how much does the agent explore before committing to a door? Premature commitment versus over-exploration is a behavioral signature independent of success.

4. Deviation from a normative baseline. Repurpose the Bayesian module (currently an optional feature) as a rational reference agent. Measure how far LLM updating diverges from ideal Bayesian updating given the same signals (a KS-style or KL-style divergence). This is the ABXLAB and hypersensitivity move of comparing to a normative or human baseline.

5. Faithfulness of memory and reasoning. Use an LLM-as-judge (ABXLAB Appendix D) to check whether the agent's stated reasoning and its compressed prior faithfully reflect the true signal structure, or whether they over-retain distractors. Ties to the faithfulness literature.

6. Heritable bias and multi-generational amplification. The novel behavioral contribution: a memorized misleading signal is not a one-step error but a persistent, inheritable state that steers all descendants. Quantify propagation depth and cost. Anchor to Ren et al. iterated-learning bias amplification.

7. Epistemic autonomy and self-correction. When and how do agents override an inherited prior (the observed "URGENT: inherited target is WRONG. DISCARD." behavior)? Quantify override rate as a function of evidence strength and prior confidence.

8. Robustness of the intervention. Show priors reduce susceptibility where CoT does not (direct dialogue with the hypersensitivity result), establishing transmission as an intervention class.

### B.3 What to keep, add, and cut

Keep:
- environment.py (RGG, the hint and distractor taxonomy is an asset; just relabel categories to choice-architecture terms in the paper).
- agent.py memory pipeline (buffer, summarization, prior compression) since it is the subject of study.
- runner.py, reproduction.py, lineage tracking.
- metrics.py drift and similarity functions.
- cloaking.py, promoted from future work to a first-class intervention.
- The fertility result, demoted to a secondary section answering "when does transmission help."

Add:
- src/interventions.py: a formal intervention set I of functions Observation to Observation. Wrap and unify the existing scattered manipulations: distractor injection, cloaking attenuation, single-nudge injection, signal reordering, hint rephrasing or reframing. This is the ABXLAB intervention engine and the spine of the new framing.
- A probe mechanism in runner.py: at configured steps, apply a controlled intervention and log a counterfactual decision record (which signals present, types, the action taken, whether it followed each distractor type, whether it matched the consistent hints).
- src/behavior.py (or extend metrics.py): susceptibility, decision-following, brittleness under perturbation, exploration horizon, faithfulness hooks, divergence from the Bayesian reference.
- A normative reference agent: reuse beliefs.py as an ideal-observer baseline, not an agent feature.
- New experiment scripts (Section B.5).

Cut or de-emphasize:
- Experiment B (parent dialogue): inconclusive, move to an appendix or drop.
- The skill library as a headline: keep as a secondary structural intervention, not a main result.
- Heavy cultural-evolution narrative as the primary frame: keep transmission as the mechanism but lead with behavior and intervention. Cultural evolution becomes the explanatory theory, not the headline.
- The grandchild route-knowledge anecdote as a capability win: reframe it as a memorization phenomenon (overfitting to a specific graph), which is more interesting behaviorally.

### B.4 Codebase change list (concrete)

1. New file src/interventions.py
   - class Intervention with apply(observation) returns observation.
   - Concrete interventions: InjectDistractor(type, target_door), CloakSignals(strength), ReorderSignals(seed), ReframeHint(style), DropHints(p). Each tagged with a category label for reporting.
   - An InterventionSet that composes these and records what was applied per step.

2. Refactor environment.observe to accept an optional intervention pipeline, so all manipulations go through one auditable path (cloaking currently lives inside observe; move it behind the intervention interface).

3. New logging in runner.py: a per-decision behavioral record (step, signals present with type and hint flag, action, door chosen, followed-hint bool, followed-distractor-by-type, intervention applied, exploration count so far). Persist as a flat table for analysis.

4. New file src/behavior.py
   - susceptibility(records, by="signal_type")
   - decision_flip_rate(records_paired) for perturbation pairs
   - exploration_horizon(per_agent)
   - bayes_divergence(agent_history, reference) using beliefs.py
   - faithfulness_audit(prior, true_signals, judge_llm)

5. Repurpose beliefs.py as a standalone IdealObserver that consumes the same signal stream and produces the normative posterior, decoupled from the agent.

6. config.py: add intervention configuration (which interventions, strength, probe steps), and a probe_mode flag for the controlled single-factor experiments.

7. New experiments (Section B.5), each emitting susceptibility tables rather than only step counts.

8. Optional minimal mode: a one-shot environment variant (single observation, forced door choice) for the cleanest causal numbers.

### B.5 New ablation and experiment matrix

Primary outcome for all: susceptibility (probability of following a misleading signal), reported in percentage points with baselines and trial fixed effects.

Baselines suite (run once, reused everywhere): random walk, no prior, blank prior, random prior, chain-of-thought only, ideal Bayesian observer. Effect sizes are always relative to these.

E1. Susceptibility by manipulation type (the behavioral fingerprint).
Controlled probes inject one distractor of a known type at a time. Report per-type pp shift. Mirrors ABXLAB Table 2. Cheap: mostly post-processing of existing runs plus the probe layer.

E2. Priors as a de-biasing intervention (headline).
Conditions: no prior, blank, random, inherited, abstract cross-world, plus CoT-only control. Measure susceptibility across generations. Claim: inherited and abstract priors lower susceptibility over generations where CoT does not. Direct dialogue with the hypersensitivity paper.

E3. Heritable bias and amplification (most striking).
Inject a single persuasive distractor into a parent stream. Measure whether it enters the prior, how many descendants inherit it, and the cost. Sweep the bottleneck size to show compression can either filter or amplify the bias. Anchor to Ren et al.

E4. Intervention-strength sweep (revives cloaking).
Treat cloak radius as continuous intervention strength. Plot susceptibility and reliance on inherited route knowledge versus strength. Frame cloaking as a member of I.

E5. Memory faithfulness and failure attribution.
LLM-as-judge on context and priors (faithfulness), plus a MemFail-style attribution of where distractors enter (summarization versus retrieval versus compression). Connects to MemoryAgentBench competencies.

E6. Brittleness under content-preserving perturbation.
Decision-flip rate when signal order is shuffled or hints are rephrased. A clean robustness result.

Secondary (kept from the original):
- Fertility ablation, reframed as "when transmission helps," not a headline.
- Convention drift, reframed as attraction toward cultural attractors (telephone-game framing).

### B.6 Paper structure (4 to 9 pages)

1. Intro: agents act on misleading context; capability metrics hide behavior; we build a controllable testbed and study susceptibility, memory, and transmission.
2. Framework: states, actions, observations, transitions, and an intervention set I (ABXLAB-style). Justify the POMDP as a memory-and-transmission generator (B.1).
3. Behavioral metrics: susceptibility, deviation from ideal observer, faithfulness, exploration horizon, heritable bias.
4. Experiments: E1 fingerprint, E2 de-biasing (headline), E3 heritable bias, E4 intervention strength, with E5 and E6 as support.
5. Connections: cultural evolution and iterated learning as explanation; memory and distractor literature as context.
6. Discussion and limitations.

Benchmark-track alternative (1 to 2 pages): propose the RGG hint-versus-distractor world plus the intervention set I as a benchmark for epistemic robustness and susceptibility, with the susceptibility metric and the ideal-observer baseline as the scoring protocol. No results required. Lowest-risk option for the deadline.

### B.7 Timeline given the deadline

- Days 1 to 2: implement interventions.py, the probe layer, and behavior.py. Run E1 and E2 baselines on small trial counts.
- Days 3 to 4: E2 and E3 at full small-scale; draft framework and metrics sections.
- Day 5: E4 and E5 if time; otherwise appendix. Draft results.
- Day 6: writing, figures, baseline effect-size tables, polish.
- Parallel safety net: draft the 1 to 2 page benchmark proposal from this repository alone in case the paper-track experiments slip.

### B.8 What success looks like

A behavior paper whose primary figure is a susceptibility table (per manipulation type, per condition, in percentage points, against baselines), whose headline is that distilled inherited priors reduce susceptibility where chain of thought does not, and whose most memorable result is that a single memorized misleading signal becomes a heritable, amplifying bias across a lineage. The POMDP and memory bottleneck are justified as the minimal machinery that makes the memory and transmission questions askable at all.
