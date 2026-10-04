# Feedback on the tools

Submission field. Everything below was measured on this account while building the
project, not recalled from documentation.

## Nebius Token Factory

**The OpenAI-compatible API is the best thing about it.** One client, one retry policy,
one JSON contract, and swapping a model is a string change. Measuring six models
against each other on the same task cost nothing in integration work, which is why this
project has comparisons in it at all.

**Image support is invisible until you try.** Of the 24 models the account serves, only
five accept an image. Every NVIDIA Nemotron size refuses one, and so do most of the
frontier text models: DeepSeek V4-Pro, Qwen3.5-397B, GLM-5.1 through 5.3, Hermes-4-405B
and gpt-oss-120b all return `400 This model does not support image input`. Finding that
out took a probe script that sends one small picture to every model on the account.

A capability flag in the models list, something as small as `"modalities": ["text",
"image"]`, would save every team doing vision work the same morning. Right now the only
way to know is to spend a call and read the error.

**Empty completions happen, and they are indistinguishable from caution.** On one run,
four frames came back with nothing at all in the content, twice each, including after a
retry with a larger token budget. Scored naively that reads as the model declining to
judge, which credits it with care it never exercised. We now retry an empty completion
once at the same settings and record it separately. It would help if a blank completion
carried a finish reason that distinguished it from a deliberate refusal.

**`reasoning_effort` is rejected by some models with a 400 rather than ignored.** Our
client drops the option and retries when the error mentions reasoning, which works, but
a per-model list of accepted options would be better than discovering it at runtime.

**Latency was never the problem.** Vision at 1280x720 ran about 9 seconds a frame and
Nemotron Nano about 8 seconds a station, consistently, with no rate limiting at the
volumes we used. Model availability and behaviour varied far more than speed did.

## NVIDIA Nemotron

**The four sizes are genuinely different models, and the difference is not a smooth
quality gradient.** The most useful thing we learned on this project is that Nemotron
3.5 Lightning, which drives a tool-calling supervisor perfectly well, **cannot produce
structured output for a short planning task at all**. Given one station of probe
geometry it writes its reasoning out in prose, re-reads the prompt, argues with itself
about which list is which, and never reaches the JSON object. That behaviour is
identical at a 300 token budget and at 3000, so it is not a truncation problem.

Nemotron 3 Nano 30B-A3B and Nemotron 3 Super 120B both answer the same prompt cleanly
and immediately. We default the planner to Nano, which is the smallest and cheapest of
the three, and it beats a hand-written arithmetic baseline on the task.

If there is one thing worth publishing per model, it is JSON reliability under a short
budget. It is the difference between a model being usable in a control loop and not,
and it does not correlate with size.

**Nano punches above its price.** It took 88 of 89 routing decisions correctly in the
welding cell against ground truth. On the tank's probe planning it is beaten by a
plain coverage rule, 9 to 12 of 28 against a deterministic 16, which is a result about
the task rather than about the model: sparse hidden defects on a fixed budget reward
systematic coverage over reasoning. Nothing in this
project needed to escalate to a larger Nemotron, and we removed the escalation path
rather than leave a feature in that never fired.

**No NVIDIA model on Token Factory accepts images**, which forced the vision half onto
a non-NVIDIA model. For a hackathon with a Physical AI track this is the single most
limiting gap, since embodied work is mostly cameras. A hosted Nemotron or Cosmos vision
model would have changed the architecture of this entry.

## Blender and the RTX 4070, for the simulation

Not a Nebius or NVIDIA service, but worth recording because it cost real time.

Cycles reads which GPU to use from user preferences rather than from the scene file, so
a render launched outside the interactive session can silently fall back to the CPU and
report nothing wrong. We now set the compute backend explicitly in a startup script and
print what it resolved to, because a render that is twenty times too slow for no stated
reason is a bad afternoon.

OptiX denoising at 192 samples gave us clean 1080p frames at about 25 seconds each on a
laptop 4070, which made a rendered dataset and an animated shot practical on hardware
that was never going to path trace anything at full sample counts.

## What we would ask for, in order

1. A modality flag in the models list.
2. A per-model note on structured output reliability under a short token budget.
3. A hosted NVIDIA vision model.
4. A finish reason that separates an empty completion from a refusal.
