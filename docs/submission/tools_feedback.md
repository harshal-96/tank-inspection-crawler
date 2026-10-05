# Feedback on the tools

Submission field. Everything below was measured on this account while building the
project, not recalled from documentation.

## Nebius Token Factory

**The OpenAI-compatible API is the best thing about it.** One client, one retry policy,
one JSON contract, and swapping a model is a string change. Measuring six models
against each other on the same task cost nothing in integration work, which is why this
project has comparisons in it at all.

**Image support is invisible until you try.** Of the 23 models the key reaches, 15
refused an image outright, all four NVIDIA Nemotron sizes among them; 8 accepted it,
and 6 of those judged the bead (`data/vision_field.jsonl`). Among the refusals are
frontier text models such as DeepSeek V4-Pro, Qwen3.5-397B, GLM-5.1 through 5.3,
Hermes-4-405B and gpt-oss-120b, which all return `400 This model does not support image
input`. Finding that out took a probe script that sends one clean and one defective
bead to every model on the account.

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

**The documented structured output controls did nothing.** Structured output is
documented as `extra_body={"guided_json": ...}`, with `response_format` accepting
`json_object` and `text`. Measured against Nemotron 3.5 Lightning, neither had any
effect: both were accepted without an error and the model still answered in prose. What
worked was `reasoning_effort="none"`, which cut 900 tokens of preamble to 86 tokens of
correct JSON on the welding cell's view prompt. A parameter that is silently ignored is
worse than one that is rejected, because nothing tells you to look.

**Latency was never the problem.** Vision at 1280x720 ran about 9.2 seconds a frame and
Nemotron Nano about 7.1 seconds a station, consistently, with no rate limiting at the
volumes we used. Model availability and behaviour varied far more than speed did.

## NVIDIA Nemotron

**The sizes do not agree on where the answer goes, and the disagreement is silent.**
Nemotron 3.5 Lightning puts its answer in `content` and leaves `reasoning` empty.
Nemotron 3 Nano does the reverse: `content` comes back empty and the whole answer, JSON
and all, arrives in `reasoning`. Reading only `content` does not raise or warn on Nano;
it returns an empty string, and any caller with a sensible fallback for an unparseable
reply applies that fallback to every call. An evaluation built that way measures the
fallback and reports it as the model's judgement, which is what happened here before we
read the answer from either field: Nano appeared to score perfectly on a decision it
was never asked. The fix is a few lines (`brain/nebius.py`, `reply_text`), but only once
you know.

**Reasoning has to be switched off explicitly for a control loop.** Left at its default,
Lightning reasons at length in plain prose before reaching an answer, and `"low"` was
not enough to stop it; `"none"` was. The welding cell's view decisions and its
corrective action briefs set it. The tank's probe planner did not, and its recorded Nano runs were made with default
reasoning.

Nemotron 3 Nano 30B-A3B and Nemotron 3 Super 120B both answered the probe planning
prompt cleanly. We default to Nano, which is the smallest and cheapest of the sizes we
used, though on the tank's probe planning it was never ahead of a short coverage rule
(below).

If there is one thing worth publishing per model, it is which response field carries
the answer and how reliably it returns JSON under a short budget. That is the
difference between a model being usable in a control loop and not, and it does not
correlate with size.

**Nano punches above its price.** It took 88 of 89 per-part view decisions correctly in
the welding cell against ground truth. On the tank's probe planning it found 9 to 12 of
28 rejectable points against the coverage rule's 16, but given only the state the model
is shown, the same rule finds 11, so on matched information the two are level. We use
the rule there because it is no worse, deterministic, free and adds no hours, not
because the model was shown to reason worse. Nothing in this project needed to escalate
to a larger Nemotron, and we removed the escalation path rather than leave a feature in
that never fired.

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
5. A per-model note on which response field carries the answer, `content` or
   `reasoning`, or one field that always does.
6. `guided_json` and `response_format` either honoured or rejected with an error, never
   silently ignored.
