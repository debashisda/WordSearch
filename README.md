# WoordSearch

Solves an Android Word Search game over ADB. Screenshot in, multi-point drag out, verified after every move.

```
screencap ──► geometry ──► glyphs ──► board ──► solver ──► player ──► swipe
              (lattice)   (letters)  (+words)  (8 dirs)  (+verify)
```

Nothing is hardcoded to a grid size. The lattice is re-fitted on every frame,
so a 6×7 board and a 12×14 board take the same code path.

---

## Setup

```bash
pip install -r requirements.txt
python -m uiautomator2 init          # optional but recommended, see Backends

adb devices                          # confirm the phone is listed
python scripts/calibrate.py          # confidence check — see what it detects
python run.py --launch --forever     # open the game, then play on
python run.py --dry-run              # read and solve, send nothing
python run.py                        # actually play the level on screen
python run.py --levels 10            # play ten levels
python run.py --forever              # keep playing until you press Ctrl-C
```

Between levels the bot finds the green continue button on the COMPLETED screen
and taps it, so a long run needs no attention. Ctrl-C stops cleanly and prints
what it got through.

---

## Layout

| File | Stage |
|---|---|
| `wsbot/adb.py` | capture frames, send gestures; `ReplayDevice` for offline work |
| `wsbot/geometry.py` | **1.** find the panel, fit the cell lattice, derive rows × cols |
| `wsbot/glyphs.py` | **2.** crop → binarize → tight-crop → resize; **3.** gated match, library, unknown-glyph tokens |
| `wsbot/board.py` | assemble a board: letters, word list, found flags |
| `wsbot/solver.py` | locate every target word, all 8 directions, tokens included |
| `wsbot/lexicon.py` | work out what unknown glyphs are, from the word list + grid |
| `wsbot/screens.py` | the screens between levels: find the continue button |
| `wsbot/player.py` | placement → swipe path → execute → verify → retry |
| `wsbot/config.py` | optional `config.json`; every field has a default |
| `run.py` | the play loop |
| `scripts/calibrate.py` | one-time check, `--save` to pin geometry |
| `scripts/label_glyphs.py` | teach the library new letters |
| `scripts/debug_view.py` | annotated overlay — run this first when something breaks |
| `scripts/selftest.py` | whole pipeline against saved frames, no phone needed |
| `scripts/benchmark.py` | where the time per word goes: cable, capture, gesture |
| `scripts/audit_glyphs.py` | find and quarantine corrupted samples in the library |
| `scripts/launch_game.py` | open the game and press through to a board |

---

## How the three stages work

### 1. Geometry — why the grid size is detected, not configured

The panel is the most reliable landmark on screen: a near-white rounded
rectangle. Found-word colour stripes punch holes in the white mask, so the mask
is flood-filled from the border before taking the largest solid blob. On your
recording this returned the identical rectangle on an untouched board and on
one with six words already solved.

Grid size then comes from scoring, not from measuring gaps. Cells are square
and the panel wraps them tightly, so a candidate `(rows, cols)` fixes the pitch
as `panel_w/cols` and `panel_h/rows`; candidates whose two pitches disagree by
more than 5% are discarded. The winner minimises area-weighted squared distance
from glyph centroids to cell centres.

That beats the two obvious alternatives, both of which were tried and rejected:

- **Clustering centroid gaps** got the shape right but the pitch wrong (176.7
  instead of 172.0), because stray blobs at the panel edge dragged the first
  and last clusters outward.
- **Projection profiles** were exact on a clean board and fell apart on a
  solved one, where colour stripes merge the profile into noise.

Candidate scoring survives both. On your level-7 frames the correct 7×6 lattice
scored **17× better** than the runner-up, on the solved board as well as the
clean one.

### 2 & 3. Glyphs — binarize, normalise, match behind a structural gate

The one thing worth knowing: **ink is the minority class, not the dark one.**
The panel carries a radial gradient, and a found word flips its cells to white
letters on saturated colour. Neither absolute brightness nor border sampling
survives that — an early version blew out a third of the cells. Ink is about
20% of a cell in both polarities, so Otsu plus "take the smaller class" is
correct either way, with no colour reasoning at all.

After that: keep the largest component, tight-crop to its bounding box, scale
to fit a 32×32 canvas preserving aspect, centre it. Matching is
`cv2.matchTemplate` with `TM_CCOEFF_NORMED` against every stored sample.

**Correlation alone is unsafe when the library is incomplete**, and that is the
normal case — a level only shows the letters it happens to use. Holding each
letter out of the library in turn, an out-of-vocabulary glyph matched the
*wrong* letter at a median score of 0.783 and a peak of 0.897. That is how
`THOMAS` became `IHOMAS`: with no T stored, the T correlated well enough
against I to pass. Raising the threshold cannot fix it on its own, because
true matches drop to ~0.894 once a grid grows and glyphs render smaller — the
two distributions overlap.

So candidates are gated on two **scale-invariant** structural features before
their score is even considered:

| feature | what it is | separates |
|---|---|---|
| `holes` | topological hole count, area-filtered | B(2) vs D(1); U(0) vs O(1); K(0) vs R(1) |
| `aspect` | width ÷ height of the tight crop | I (0.17) vs T (~0.79) — a 4.6× gap |

Measured across a 2× range of render sizes, a letter's aspect drifts by at most
**1.11×** and its hole count not at all, so the gate is safe at a 1.45
tolerance. With the gate in place every out-of-vocabulary match falls to ≤0.877
while true native matches sit at ≥0.961, so `min_score` is set to **0.90** and
the two no longer overlap. Leave-one-out now refuses **17/17** unknown letters
instead of guessing 13 of them.

A refusal is safe; a wrong letter is not. Across 80 frames the matcher returns
**zero wrong letters** — the only imperfect reads are glyphs a finger was
covering, which multi-frame voting clears.

Word-list pills use the same library — same typeface, and normalisation makes
the smaller point size irrelevant. That is why there is no Tesseract here: on
isolated capitals in a known font, template matching is both more accurate and
much faster.

---

## An incomplete alphabet is no longer fatal

`glyphs/` ships with **17 of 26 letters** (54 samples) taken from the
recording. Missing: `B F J K Q T U X Z`. A level using any of those used to
produce silent corruption. It no longer can, for two reasons.

**1. Unknown glyphs become tokens, and tokens still solve.**

A glyph the library cannot name does not have to be *identified* to be
*useful*. The grid and the word list share one typeface, so the same unknown
glyph can be recognised as the same symbol in both without ever learning its
name. Unreadable glyphs are clustered and given tokens `?1`, `?2`, … shared
across the whole board:

```
words: ?1N?1RGY, P?1AC?1, G?2LD, ?3ISD?2M, CALM, VI?1?3, ?3IN
solved 7/7 anyway
```

`?1ACO?2` has exactly one place it can sit in the grid, so the drag is correct
whether or not anyone establishes that `?1` is a J.

A token constrains only against *letters*, never against another token. That
distinction is what makes doubled letters work. The same glyph is often
recognised at one position and refused at another, so MUFFINS read as
`MUF?1INS` — one F named, one not — while both F cells in the grid were `?1`.
Requiring a token to line up with the same opposing *symbol* made that look
self-contradictory, and the word could not be placed anywhere, so the level
never finished; EIFFEL failed identically. Pairing two unknowns says nothing
about which letter either is, so it now implies nothing, while a token paired
with a named letter still binds to it for the whole word. Tested by deleting letters
from the library: with **7 of 17 letters removed**, all 7 words still placed
correctly. Confirmed on a real level-8 board (`Names`), where 14 of 48 cells
were letters the library had never seen: all six words — including `THOMAS` on
a SW diagonal and `MARK` reading backwards — were located correctly.

Clustering deliberately errs toward splitting. On that level-8 board the six
`J` glyphs fell into two groups whose tight crops differed by a single pixel,
correlating 0.99+ within each group and 0.874-0.888 across them. Since
genuinely different letters also correlate up to ~0.88, no threshold separates
"same letter, rendered slightly differently" from "two different letters". So
the solver absorbs it instead: identical tokens match for free, different
tokens may still match but cost a wildcard, and every token must stand for one
thing consistently across a word. Verified correct at cluster thresholds from
0.80 to 0.95.

**2. Unknown glyphs are usually identified automatically.**

A bundled word list (196,906 words, 580 KB gzipped, includes first names) turns
the tokens into letters with no input from you. Two constraints do it:

* the pattern must match a real word — `?1HOMAS` has exactly one reading;
* **a token cannot be a letter the library already holds**, because tokens
  exist precisely because matching refused them.

The second constraint is what makes it decisive. `?2ACO?5` matches five
dictionary words — BACON, JACOB, MACON, RACON, VACOA — but with the library
holding `ACDEGHILMNOPRSVWY`, only JACOB assigns both tokens to letters it
lacks. Tokens are then tied across the whole board, and the result is checked
against the grid before being accepted: the dictionary proposes, the board
disposes.

It iterates, because resolving one letter constrains the next. On the real
level-8 board the first pass got T, J, B and U but left `MAR?3` split between
MARK and MART — T was still unknown, so still a legal token value. Learning T
and re-reading eliminated MART, and MARK resolved on the second pass:

```
resolved ?1=T, ?2=J, ?4=J, ?5=B, ?6=U
resolved ?1=K
identified BJKTU on its own; learned 15 glyph sample(s)
```

Pass `--no-lexicon` to turn this off.

It cannot help with every level. Themed categories are full of proper nouns —
a "Musicians" level wants BEEGEES, TOMPETTY and LILWAYNE, none of which are in
any dictionary — so tokens there stay unresolved. That costs nothing: the level
still solves and plays through them, and `--words` pins the letters down if you
want them learned.

**3. The library completes itself as you play.**

A confirmed word reveals the true letter of every cell it crossed *and* every
glyph in its own pill, so both render sizes get learned. Play a few levels and
the alphabet fills in. `--no-learn` turns this off.

To resolve the unknowns immediately, type the level's words in once:

```bash
python run.py --words THOMAS JACOB CAROL RYAN JOSHUA MARK
```

Each placement names the letter in every cell it crosses, so `T J B U K` are
learned permanently and the next level reads them normally. You rarely need
this now that the lexicon runs automatically — it is the fallback for a themed
level whose words are not in any dictionary.

Learning refuses ambiguity on purpose. A word is only learned from when it
sits in exactly one place in the grid *and* matches exactly one word pill.
That guard matters: once JACOB's pill read `?4A?2O?7`, CAROL was also
compatible with it, and learning from that pairing wrote J as C, C as R and B
as L into the library. Corrupting the reference set is worse than learning
nothing.

Or label by hand, asking only about cells it could not read:

```bash
python scripts/label_glyphs.py --only-unknown
```

---

## Starting the game

```bash
python run.py --launch --forever      # open it, then play until Ctrl-C
python scripts/launch_game.py         # just open it, to check the sequence
python scripts/launch_game.py --restart   # force-stop first, for a clean start
```

Launching is not one step. Traced from a recording of a cold start, the app
goes: blue splash, then a dark loading screen with a progress bar for several
seconds, then a main menu, and only then a board. So the launcher watches
rather than launching and hoping — it polls, waits out anything with nothing
to press, and stops when a board appears.

The menu is the part that needs care. It stacks three large buttons: a red
PLAY carrying the level number, a yellow THEMES, and a green INFINITE.
Pressing the wrong one starts a different mode — and the first version did
exactly that, pressing THEMES, because the red PLAY button was never detected
at all. Red sits at both ends of the hue scale, so fixed hue bands cut it in
half, and THEMES was simply the topmost button that could be seen. See
`wsbot/screens.py` for what that took to fix.

With all three detected, PLAY is both the largest (221k px against THEMES'
125k) and the topmost, so the launcher requires those two signals to agree
before pressing, and waits rather than guessing if they ever disagree. The
level-complete screens have a single button, so the same rule covers them.

`monkey -p <package> -c android.intent.category.LAUNCHER 1` does the launching,
which resolves the activity itself, so no activity name has to be kept up to
date; if that fails it falls back to asking the package manager. The package
(`com.blackout.word`) is in `config.json` as `package`, or pass `--package`.

## Playing continuously

Clearing a level replaces the board with a COMPLETED screen and a large green
button that starts the next one. The loop handles this itself, deciding what is
on screen each poll rather than tapping hopefully:

| what it sees | what it does |
|---|---|
| a board with new letters | start playing it |
| a between-level screen | tap its button, whatever the colour |
| anything else | wait — an ad or a transition |

More than one screen can stand between levels. Besides COMPLETED with its green
button, a photo card with a purple COLLECT button sometimes appears first and
leads to COMPLETED once dismissed. The loop taps whatever it finds and keeps
polling, so it walks the chain without needing to know its length.

The button is therefore found by shape, not colour — hardcoding green meant the
COLLECT screen was invisible and the run stalled. Nor by position: an earlier
version tapped four fifths of the way down the screen, which on a real
completion screen lands 400px below the button.

Two details make it reliable:

* **Each hue band is searched separately.** One mask over "any saturated
  colour" fails on the COLLECT screen, because the purple button touches the
  turquoise sea in the photo above it and they merge into a ragged blob filling
  39% of its box. Per-band, the button comes out alone at 92%.

* **No board may be on screen.** Shape alone also matches a found word's
  coloured stripe: a 4- or 5-letter horizontal one has the same size, fill and
  aspect as a button, and *was* matched in testing, which would have made the
  bot tap a word it had just found. The board test decides it — but looking for
  a pale panel is not enough, because the photo card's bright border passes for
  one. Fitting the cell lattice settles it: real boards score at least 0.86
  (80 frames, five levels from 6x4 to 10x8), the photo card 0.11.

Across 80 frames of live play there are zero false detections.

A level that does not fully clear is re-read and retried, up to three times,
rather than waiting for a next level that is not coming.

**The transition costs one capture per screen, and nothing else.** Three things
used to inflate it:

* a fixed one-second sleep on every poll, whether or not anything had changed;
* a two-frame board read inside the wait, whose result the caller discarded
  before re-reading the board properly itself — two captures thrown away;
* a full-resolution hue scan on every poll, at ~178 ms.

Now the board is read from the frame already in hand, the scan runs on a
quarter-size copy (~12 ms, same answer to within a pixel), and the frame that
proves the level is over is also used to press the first button. The new level
is recognised by having an *unplayed* word rather than by comparing grid text:
the level just finished has every pill struck through, so one unfound pill is
proof the next has loaded.

| from last word to next board | captures | at 0.4 s each | at 1.0 s each |
|---|---|---|---|
| COMPLETED only | 2 | 1.2 s | 2.4 s |
| COLLECT then COMPLETED | 3 | 2.1 s | 3.9 s |

That is the floor: one capture to see each screen. Anything beyond it is
capture latency, which `scripts/benchmark.py` measures.

## When a word is not in the grid

The game does not list words it has not hidden, so a word that cannot be found
anywhere is a misread pill, not a missing word. VINEGAR once read as VGNEGAR —
one letter wrong — which appears nowhere, so the level could never finish and
was replayed until the bot gave up.

The grid is a far stronger constraint than the pill image, so it is used to
correct the reading rather than the other way round. Blank out one position and
search again: if exactly one placement survives, the grid itself says what that
letter must be. Of the 175 single-letter variants of VGNEGAR, exactly one
places, and it is VINEGAR.

Two positions are tried only if one fails, and then the answer must be
unambiguous or a dictionary word — blanking more of a short word makes
coincidences easy, and KGWG admits both KIWI and RGWS. Measured on the real
level-59 grid:

| misread | repaired to |
|---|---|
| `VGNEGAR` | VINEGAR |
| `VGNEGAB` | VINEGAR (two positions) |
| `STBAWBERRY` | STRAWBERRY |
| `KGWG` | KIWI (dictionary breaks the tie against RGWS) |
| `ZZZZZZZ` | nothing — correctly refused |

## Matching does not slow down as the library grows

Scoring a glyph used to call `cv2.matchTemplate` once per stored sample, so a
board read grew in step with everything the bot had learned. On a 10x8 board:
0.22 s at 53 samples, 1.95 s at 1060 — and since each level starts with a
three-frame consensus read, that was almost six seconds of waiting before the
first word, growing every session.

For templates and images of the same size, `TM_CCOEFF_NORMED` at the single
valid position *is* the Pearson correlation of the two read as vectors. So each
sample is stored zero-meaned and scaled to unit length once, and scoring a
glyph against the entire library becomes one matrix-vector product.

| library | before | after |
|---|---|---|
| 53 samples | 221 ms | 159 ms |
| 530 samples | 1099 ms | 203 ms |
| 1060 samples | **1952 ms** | **216 ms** |

The grid lattice is also fitted once per consensus read rather than per frame,
since it cannot move between frames of the same level. Together that takes the
read at the start of a level from **5.9 s to 0.5 s** of CPU at 1060 samples,
and it is now nearly flat as the library grows. Letter choices are unchanged:
3342 correct, 0 wrong, 18 refused across the 80-frame set, exactly as before.

## Keeping the library clean

Because the library teaches itself, a sample stored under the wrong letter is
permanent, and it poisons the structural gate for that letter from then on: the
gate passes a candidate if *any* stored sample matches on holes and aspect, so
a single narrow glyph filed under G is enough for G to start matching narrow
glyphs. That is how an I came to be read as G.

Two defences:

* **`add()` refuses a structurally inconsistent sample.** A glyph is only
  stored under a letter whose existing samples agree with it on hole count and
  aspect ratio. A letter with no samples yet is always accepted, so a new
  letter can still be learned. Rebuilding the shipped library with this guard
  active rejected 858 of 887 candidate pill samples as duplicates or
  inconsistent.

* **`scripts/audit_glyphs.py`** finds samples already stored. Hole count is a
  property of the letterform, so it is asserted outright — unless it would
  condemn every sample of a letter, which means the table is wrong for that
  typeface rather than the samples being bad. Aspect is judged against the
  letter's own siblings. Suspects move to `glyphs/_quarantine/` rather than
  being deleted.

If your library has been learning for a while, run it:

```bash
python scripts/audit_glyphs.py          # report
python scripts/audit_glyphs.py --fix    # quarantine
```

It found two corrupted samples in the library shipped here — an "I" whose
binarisation left a hole covering 28% of the glyph, and a half-width "V".

## Verification, and why it matters

Every drag is checked before moving on; a word counts as played only once its
pill flips to the found state. A missed drag retries, each attempt slower and
overshooting the final cell slightly:

| Attempt | ms per cell | Overshoot | Direction |
|---|---|---|---|
| 1 | 55 | none | forward |
| 2 | 90 | 18% of a cell | forward |
| 3 | 90 | 18% of a cell | **reversed** |
| 4 | 140 | 30% of a cell | forward |

The reversed rung exists because the game accepts a word dragged from either
end — `BOCAJ` is `JACOB`. Same cells, so it costs nothing and removes a whole
class of failure.

Note this is *only* about the drag. **Backwards placements need no special
search**: looking for a word going `W` is identical to looking for its reverse
going `E`, and all eight directions are already searched. On the Names board,
`CAROL`, `RYAN` and `MARK` are all right-to-left in the grid and all found.

This is the difference between a demo that works once and something you can
leave running.

**How the check is made matters as much as making it.** Verification reads one
number: the brightness of the word's own pill, at a box already known from the
initial board read. It does not re-read the grid and does not re-OCR the pill
text.

That is not only for speed. Re-reading the text is actively *wrong*: during the
strike-through animation an extra component appears inside the pill, so WISDOM
reads as `WISDOM?2` and CALM as `CALM?2`. A check that compares text then
decides a word it had just found "did not register", and drags it again — four
times, on a word that was already complete. Reproduced on frame 12 of the
sample recording, where the old text comparison returns False and the
brightness check correctly returns True.

Brightness cannot fail that way: unfound text peaks at ~252 and found text at
~129, and the box never moves.

The saving is large, because a screencap is by far the most expensive thing the
bot does — the phone PNG-encodes a 1080x2340 frame and pushes it over USB,
which dwarfs the ~75 ms of recognition work.

| per word | old | new |
|---|---|---|
| screencaps | 3 | 1 |
| fixed sleep | 0.77 s | none — polls, returns on response |
| grid cells re-read | 48 | 0 |
| cost of the check itself | ~75 ms | 0.075 ms |

**Deciding the level is over** uses the same idea. Reading the board again
after the last word does not work: by then the COMPLETED screen has replaced
it, so the read finds zero word pills, and `complete` requires at least one —
nine words all landing still reported "not cleared", and the level was replayed
from the start. Completion is now settled by either signal:

* the board is gone, i.e. the COMPLETED screen is up — definitive;
* every pill is struck through, by brightness at the known boxes.

It polls briefly for either, since the screen does not change the instant the
last word lands.

The grid cannot change during a level, so it is read once at the start and not
again. `settle` in `config.json` is now a
*budget* (how long to wait before calling a drag failed), not a delay that is
always paid.

Board reads use 3-frame majority voting, because a finger still on the glass
can hide a letter for one frame. Single frames read your recording's grid
exactly 65/80 times; **3-frame consensus read it exactly 78/78.**

---

## Why a word takes as long as it does

Two things cost a round trip to the phone: making the gesture and taking a
capture. Everything else is noise — recognition is about 75 ms.

Checking each word as it is played means one of each, per word. So by default
all the drags go out first, in a single adb invocation, and **one** capture
settles the whole board. On a seven-word level that is 2 round trips instead of
15. Anything that did not land is retried individually with per-word checking,
so a missed drag costs no more than before. `--no-batch` restores the
word-at-a-time behaviour if you need to isolate one.

What remains is the gesture itself, and on the `adb` backend that is dominated
by something no cable can fix: **`input` is a Java program**, so Android starts
a JVM for every `input swipe` — commonly 0.5-1.5 s before the drag begins.
`uiautomator2` talks to an agent already running on the device and skips it
entirely, which is usually the single largest speedup available:

```bash
pip install uiautomator2 && python -m uiautomator2 init
```

`scripts/benchmark.py` measures all of this on your own setup — adb round-trip
latency, PNG vs raw capture, the JVM cost, and the implied USB throughput, so
you can tell a slow cable from a slow phone.

## Screen capture

A screencap is the most expensive thing in the loop, so there are two ways to
take one and the bot times both at startup and keeps the faster:

| | how | trade-off |
|---|---|---|
| `png` | `screencap -p` | the phone PNG-encodes the frame first — often over a second on a 1080x2340 screen |
| `raw` | `screencap` | no encoding at all; ~10 MB instead of ~2 MB, but transfer is usually far cheaper than the encode |

The raw payload is three little-endian uint32s (width, height, format) followed
by RGBA, with a fourth uint32 on newer Android — 12- or 16-byte header, decided
by which one makes the length add up. A payload that does not add up is
rejected and PNG is used instead, which also catches the old Windows adb bug
that rewrote newlines inside binary output.

Startup prints what it chose, e.g. `screen capture: using raw (png 1180ms,
raw 240ms)`.

## Backends

| | How it works | When to use |
|---|---|---|
| `u2` | `uiautomator2.swipe_points()` — a real multi-point gesture | default; needed for reliable long and diagonal words |
| `adb` | `adb shell input swipe` — two points, Android interpolates | fallback with no extra dependencies |

`adb shell input swipe` sends only a start and an end. Android fills in MOVE
events between them, which tracks a straight line well enough, but the sampling
rate is not guaranteed and long words can drop a letter. `swipe_points` feeds
every cell centre as its own sample. Pick it with `--backend u2`, or let `auto`
find it.

---

## What was measured, and what was not

Verified against your 40-second recording (1080×2340, level 7, 80 frames):

- panel at `(24, 584) 1032×1204`; `1032/6 == 1204/7 == 172.0` exactly
- grid shape detected correctly on **80/80** frames
- grid letters exact on **78/78** 3-frame windows
- word list exact on **77/80** single frames, including found/unfound state
- **0 wrong letters** in 3360 cell reads; 19 refusals, all finger-occluded
- out-of-vocabulary letters refused **17/17** in leave-one-out (was 4/17)
- **236/237** individual placements correct on single frames; the one failure
  came from a word pill caught mid-strike-through
- **229/229** correct under 3-frame consensus, every word located in **78/78**
  windows — which is why consensus is the default
- **0/80** false continue-button detections during live play
- 2 of 80 frames read a found pill's text wrongly mid-animation, which is what
  made the old text-comparison check retry completed words; the brightness
  check is correct on both

Verified on a level-8 board (`Names`, 8x6) with nine letters absent from the
library:

- grid read with **0 wrong letters**, 14 cells as tokens
- all 6 words located, every placement correct
- worst true match **0.918**, best refusal **0.704** — a 0.214 gap either side
  of the 0.90 threshold, on real device output rather than simulation
- the lexicon identified **B J K T U** unaided (17 -> 22 of 26 letters), after
  which the board read with 47 of 48 cells resolved and no user input at all
- across 24 scenarios on both real boards, with up to 6 further letters deleted
  from the library: **0 wrong cells, 0 unsolved words**
- found-state threshold is wide: unfound text sits at ~252, found at ~129
- all 7 words located, including `PEACE` as a **diagonal** (r1c0 → SE)

Not observed, so handled defensively rather than precisely:

- **A grid other than 6×7.** The detector is built to generalise and its
  scoring is decisive, but your recording only ever shows one size. Run
  `scripts/calibrate.py` on a bigger level and check the chosen lattice before
  trusting it unattended.
- **Ads and other interstitials.** The COMPLETED and COLLECT screens are
  handled, and any future one with a solid coloured button should be too, since
  nothing about those two is hardcoded. A full-screen ad whose close control is
  a small grey X would not be, and the bot would wait it out rather than
  dismiss it.
- **Four letters** (`F Q X Z`) are still unseen. Levels using them solve
  through tokens, and `--words` or normal play learns them. The structural gate
  is validated by holding out letters that *are* present; absent ones are
  reasoned about, not measured.

---

## When something goes wrong

```bash
python scripts/debug_view.py                 # from the device
python scripts/debug_view.py saved_shot.png  # from a file
```

Writes `debug/debug.png` with the panel outline, the lattice, the letter read
from each cell, the word pills, and an arrow per placement. Nearly every
failure is obvious in that one image.

To test a change without a phone:

```bash
python scripts/selftest.py frame1.png frame2.png frame3.png
```

Common cases:

| Symptom | Cause |
|---|---|
| "Grid panel not found" | not on a level screen, or a popup is covering it |
| wrong rows × cols | check `calibrate.py`'s candidate table; pin with `--save` if needed |
| `?1 ?2` tokens in the grid | letters not in the library yet. Harmless — words still solve. Run `--words` once to learn them |
| the same letter gets two tokens | expected; sub-pixel rendering splits a cluster. The solver handles it, no action needed |
| several words run together as one | fixed: the pill mask no longer uses a wide horizontal close. A warning now fires if it ever recurs |
| letters refused that used to read | `min_score` too high for your render size; `calibrate.py` prints the score spread, lower it in `config.json` |
| words read but drags miss | use `--backend u2`; raise `ms_per_point` in `config.json` |
| a completed word is dragged again | fixed: the check no longer compares pill text, which corrupts mid-animation |
| "could not locate" a word you can see | a misread pill; the grid now repairs it. If it persists, run `scripts/audit_glyphs.py` |
| words with a doubled letter never place | fixed: token bindings now constrain against letters, not against other tokens |
| each word takes seconds | run `scripts/benchmark.py`. Usually the JVM start behind `adb shell input swipe`; installing uiautomator2 removes it |
| the level replays after every word lands | fixed: completion is judged by the COMPLETED screen, not by re-reading a board that is no longer there |
| "the next level did not appear" on a board you can see | was a confidence artefact on grids whose size has an integer factor (6x4 against 18x12); harmonics are now excluded |
| `screencap` returns nothing | the game or ROM blocks capture; you would need scrcpy |

---

## config.json

Entirely optional — the bot runs without it. Pin geometry only if
auto-detection ever picks wrong, and note that pinning stops it adapting to a
new grid size.

```json
{
  "backend": "u2",
  "package": "com.blackout.word",
  "min_score": 0.90,
  "match_margin": 0.03,
  "aspect_tol": 1.45,
  "use_structure": true,
  "consensus_frames": 3,
  "ms_per_point": 55,
  "settle": 1.6,
  "poll": 0.25,
  "gap": 0.12,
  "batch": true,
  "max_attempts": 4,
  "panel": null,
  "rows": null,
  "cols": null,
  "advance_tap": null,
  "word_list": null
}
```

---

## A note on scope

This automates a single-player, offline, locally-stored game. There is no
server, no account and no leaderboard to affect, but it near-certainly breaches
the app's terms of service. Keep it to your own device.