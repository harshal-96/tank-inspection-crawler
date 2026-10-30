# Video script and shot list

The film is 2:58, 1920×1080 at 30 fps, narrated, with subtitles. The Physical AI track
needs at least one minute showing the key application modules in action when there is
no hardware: chapters 03 to 07 run from 0:53 to 2:09, 77 seconds of the crawler's run,
the recorded inspection, the probe map and planner, and the dashboard. Chapter 08 adds 20
seconds of the welding cell's model roles.

Everything in it runs in simulation, and every model result in it is a recorded run
replayed against the answer key. No model is called to make the film, and the film says
so on screen.

It opens on the problem, not the robot: the confined-space figures, then an illustrated
fall inside a tank. The crawler first appears on the title card, after both.

---

## Cold open, 0:00 to 0:14

**On screen.** A dark field. "1,030 workers killed in confined spaces" counts up beside a
grid of 1,030 dots, one for each, with "88 in the lowest year, 166 in the highest" below
it. On "two hundred and five", 205 of the dots light blue and "205 inside tanks, bins and
vats" counts up. On "a hundred and fifty-six", those go back and 156 light orange, with
"156 in falls to a lower level". The two groups never light together, because the
Bureau's categories can overlap (a fall inside a tank is in both). The source line stays
on screen throughout: US Bureau of Labor Statistics, "Fatal occupational injuries
involving confined spaces, 2011 to 2018", published 2020.

**Narration.**
> In eight years, a thousand and thirty American workers died in confined spaces. Two
> hundred and five inside tanks, bins and vats. A hundred and fifty-six in falls.

## 01 The confined space, 0:14 to 0:27

**On screen.** An illustration, tagged as one for as long as it is shown. Inside the
project's own tank scene, a worker in coverall, helmet and headlamp works the wall from a
scaffold, beside the lit manway he came in through. Shortly after "scaffold" he sways
back, goes limp and falls off the open back of the scaffold, head first, to the tank
floor, where he stays. The fall is simulated with rigid-body physics on the figure's own
skeleton, so it falls the way a body does, not along a drawn path. No crawler is in the
shot. It illustrates the hazard the robot exists to remove. It is not a recorded event,
and no number comes from it. The avatar and its motion capture are Microsoft Rocketbox
(MIT).

**Narration.**
> Before anyone goes in, the tank is drained, cleaned, ventilated and certified. Then a
> person climbs in and works the wall from a scaffold.

**Chips.**
- Drained, cleaned, ventilated and certified, before anyone climbs in.
- Falls to a lower level: 156 of the 1,030. US Bureau of Labor Statistics. (After he
  lands.)

## Title, 0:27 to 0:36

**On screen.** The dark title card, "Tank Inspection Crawler". The tank is drawn as
courses and plates, with the crawler working one seam and probe points appearing behind
it. This is the first time the crawler is seen.

**Narration.**
> Tank Inspection Crawler. A robot for the welds inside a steel tank, so fewer people
> have to climb in.

## 02 Two instruments, 0:36 to 0:53

**On screen.** The crawler hero render. On "ultrasonic" it cuts to the probe, simulated
(`tank/ultrasound.py`): two cross-sections through the wall on one clock, a sound 10 mm
plate and the thinnest place the survey probes (station 9 at +0.30 m, 3.86 mm left). The
same 5 MHz pulse goes in under the probe and the pressure wave is drawn as it travels and
bounces between the two faces. Under each, the trace the gauge records draws itself, with
the back-wall echoes marked as they arrive: 3.38 µs apart on the sound plate, 1.30 µs
apart over the hidden loss. The readings follow: 10.00 mm, and 3.86 mm, 39% of the plate,
under the 60% limit. The inner face, the one the camera sees, is labelled and identical
in both. A line on screen says what it is: simulated, a finite-difference pressure wave,
thickness = velocity × echo gap ÷ 2.

**Narration.**
> The crawler follows the weld seams, the lines where the steel plates are joined. It
> carries two instruments: a camera that judges the weld, and an ultrasonic probe that
> measures the steel left behind it. Neither can do the other's job.

## 03 One seam, in simulation, 0:53 to 1:04

**On screen.** The rendered seam shot, all 240 frames.

**Narration.**
> Here it works one seam in a simulated tank, twenty metres across. Every fault was
> placed by hand, so the answer key comes from the scene, not from a model.

## 04 Reading the weld, 1:04 to 1:23

**On screen.** The 24 survey frames, a station at a time. The answer key's box sits on
each seam fault in view. A panel shows the recorded verdict, the fault the model named,
the key, the model's own stated reason and the running tallies. The film then returns to
station 12, the miss, with the crack boxed at the edge of the frame.

**Narration.**
> At each of twenty-four stations, a vision model on Nebius Token Factory reads the
> frame and decides whether a person has to go in. It got that call right on
> twenty-three of twenty-four, with no false alarms. The one miss is a crack the frame
> before had already flagged.

**On screen.**
- Walk-in call right 23 / 24.
- False alarms 0.
- Clean frames passed 13 / 13.
- Fault frames flagged 10 / 11.
- DeepSeek V4.1 Flash on Nebius Token Factory. Recorded verdicts, re-marked against the answer key. A development set.

## 05 What the camera cannot see, 1:23 to 1:39

**On screen.** The wall along the run: 24 stations by 7 probe offsets, coloured by the
remaining wall from the manifest. The camera's band of plus or minus 83 mm and its call
at each station sit above it. The visible rust patch is outlined, and the failing places
pulse.

**Narration.**
> But the worst wall loss leaves no mark on the surface. The rust the camera sees is
> inside the limit. Of a hundred and sixty-eight places the probe can reach,
> twenty-eight fail, and twenty sit behind frames the camera passed.

**On screen.**
- 168, 28, 20, with a line under the 168: each read by the simulated gauge, within 0.05 mm,
  the same 28 fail, even under a 0.5 mm lining.
- 5 of 5 hidden patches fail; the visible rust, at 69% of the wall, does not.
- The hidden loss is set heavier than a real shell would carry, so that planners can be told apart.

## 06 Where to probe, 1:39 to 2:00

**On screen.** The coverage rule's 48 probes land station by station, and its count runs
to 16. Then the planner panel shows each arm as a range on a 0 to 28 axis:
- best possible: 23;
- the rule over all 5040 tie-break orders: median 12, 7 to 18;
- the rule shown only the model's view: median 13;
- Nemotron 3 Nano's three recorded runs: 9, 10 and 12;
- random picks: 4 to 12.

**Narration.**
> There is time for two measurements in seven at each station. In no recorded run was
> NVIDIA Nemotron ahead of a simple coverage rule: typically twelve or thirteen for the
> rule, nine to twelve for the model, and no difference is significant. So arithmetic
> drives the arm, and the model stays on the picture.

**Footnote on screen.** Paired exact p 0.14 to 0.34 per run, none significant. The hidden
patches were placed by hand, where coverage does well.

## 07 The inspection record, 2:00 to 2:09

**On screen.** The tank dashboard, captured full page from the running service. It holds
on the scoreboard, then scrolls to the station viewer.

**Narration.**
> Every station lands in the inspection record: the frame, the call and its reason, the
> probe readings, and what goes to a person.

## 08 The same loop, on a welding cell, 2:09 to 2:30

**On screen.** The seam result chart beside a panel of the cell's results:
- Tracker: 22.8 to 0.67 mm, 50 of 50 parts under 1 mm on exact rendered depth. The whole chain comes to about 2.8 mm against 1.5 mm.
- Nemotron 3 Nano: 88 / 89, against 86 / 89 for the same limits as arithmetic, p 0.625.
- Nemotron 3.5 Lightning: the supervisor and the briefs.

**Narration.**
> The same loop first ran on a robotic welding cell, where part variation moves the
> joint. There, NVIDIA Nemotron Nano decides for every part whether a measurement is good
> enough to weld on: eighty-eight of eighty-nine right. Nemotron Lightning supervises the
> cell and writes the corrective briefs.

## 09 From simulation to steel, 2:30 to 2:46

**On screen.** Frame 3 with the undercut boxed: "reported each time as the corroded plate
beside it". Then the bench rig, path traced from the repository's own model of it
(`tank/sim/blender_bench.py`, every part at the size it is sold in): the small crawler on
the welded test plate, the thickness gauge with its probe and couplant, and the laptop
running the dashboard, as the camera moves in. It is tagged "Render · designed, not yet
built" for as long as it is on screen, and the chip below it says the same.

**Narration.**
> Everything here runs in simulation, and one fault still needs work: a groove at the
> weld toe, reported four times as the rust beside it. Next, a bench crawler on a real
> plate, with hidden wall loss to find.

## Close, 2:46 to 2:58

**On screen.** The tank motif, with this line: "The model on the picture. Arithmetic on
the arm. A person for whatever it cannot defend." Below it: "All results in simulation ·
recorded model runs, replayed".

**Narration.**
> Tank Inspection Crawler. The model on the picture, arithmetic on the arm, and a person
> for whatever it cannot defend.

---

## Rules check

- **Length.** 2:58, under three minutes.
- **Modules in action.** 77 seconds in chapters 03 to 07, plus 20 in chapter 08. That
  satisfies the no-hardware branch of the Physical AI rule. No hardware is claimed: the
  bench rig is a render, tagged "Render · designed, not yet built" on screen however real
  it looks, and "In simulation" sits on every chapter.
- **Every number spoken or shown is printed by the repository.**
  - 23 of 24, 13 of 13, 10 of 11, 0 false alarms and the station 12 miss: `tank/score_survey.py --replay` and `tank/significance.py`.
  - 168, 28, 20, the 16, the median 12 and 13, 9 to 12 and the chance band: `tank/score_probes.py --baselines` and `--tiebreaks`.
  - p 0.14 to 0.34: `tank/significance.py`.
  - 3.86 mm, 1.30 and 3.38 µs, 39%, within 0.05 mm and the same 28, and the same under a
    0.5 mm lining: `tank/ultrasound.py`, which writes `tank/data/ultrasound.json`.
  - 88, 86 and p 0.625: `scripts/eval_router_truth.py --rule`.
  - 22.8 mm, 0.67 mm and the 2.8 mm chain: the README's cell results.
  - 1,030, 88 to 166, 205 and 156: the US Bureau of Labor Statistics fact sheet
    "Fatal occupational injuries involving confined spaces, 2011 to 2018" (2020), cited in
    `devpost.md` and named on screen.
- **Public-content rules.** No em dashes. The seam is glossed. Joint shift is attributed
  to part variation. The seam result chart keeps its "exaggerated for visibility" note.
  No vendor name appears in a chapter title.
- **No live clip.** The film shows no live call, so it makes no claim that one is
  running. Every number in it comes from a replay.
- **One illustration.** The falling worker in chapter 01 is tagged ILLUSTRATION on
  screen for as long as he is shown. It carries no result, and the video description
  says the same. Avatar credit: Microsoft Rocketbox (MIT), in the description.
