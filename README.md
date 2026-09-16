# Riparian Encroachment Detector

Kenya's rule for how close a building can legally stand to a river isn't as settled as it
sounds. Thirty metres has been the long-enforced standard. A newer, stricter 60-metre line
is now on the table, and some of the structures already caught in enforcement were approved
by the very government now demolishing them. Legal compliance here is a moving target, set
by policy, not by physics.

Satellite imagery can't resolve that ambiguity. It was never built to. Coarse government
remote-sensing surveys have found as few as 118 structures in an area a Pamoja Trust ground
survey later confirmed holds roughly 700: dense, clustered informal homes blend into a
single pixel and simply vanish from the count. A satellite can tell you where a rooftop
probably is. It can't tell you where a wall ends, when a building went up, who owns it, or
which version of the rule applies to it.

This project sits in that gap. Not to declare what's legal, but to tell a human being where to
look first, and exactly how much to trust what they find once they get there.

## Team

Kelvin Mwaniki · Jane Murungi · Miring'u Kamau · Kimberly Wangui · Lewis Ndung'u · Christopher Bosire

*Moringa School · Data Science Capstone*

## How we built this

**Step 1: Understand what "too close" actually means**

Before writing any code, we had to understand the problem we were actually solving, not the
one that's easy to code. Nairobi's riparian reserve isn't a single fixed number, so a tool
that just draws one line and calls everything inside it "illegal" would be lying by
omission. We weren't building something to hand down a verdict. We were building something
to help a person figure out where to look first, honestly.

**Step 2: Define the regions, clean the data**
(`notebooks/01_preprocessing.ipynb`)

We picked three study areas (Kasarani, Gatharaini, Motoine), each a fixed-radius circle,
not a hand-picked bounding box. A hand-picked box is too easy to unconsciously fit around
what you already expect to find. From there: clip the OSM waterways to each region, build
the 60m riparian buffer, pull a full-band Sentinel-2 composite for the year, and label every
pixel against ESA WorldCover ground truth.

**Step 3: Detect and score buildings**
(`notebooks/02_modelling.ipynb`)

Our first instinct was to train a custom detector directly on the satellite imagery. We
checked that assumption against the real median building size in our data before committing
to it, and found most structures here are smaller than a single Sentinel-2 pixel. That
ruled out pixel-based detection outright, so we built on Google's Open Buildings dataset
instead, and trained one shared Random Forest across all three regions to score each
detection as built-up or not. The custom detector isn't deleted, just stubbed behind the
same interface. The resolution problem, not the idea, is what's currently blocking it.

**Step 4: Build something people can actually use**
(`app.py`)

A results CSV isn't useful to anyone who isn't already a data scientist. We built an
interactive Streamlit dashboard instead: pick a region, adjust the screening thresholds
live, see detections on a real map, compare areas side by side. Every number the dashboard
shows carries its own honesty label: Kasarani's flagging distance is calibrated against a
real field count; Gatharaini and Motoine reuse it untested, and the dashboard says so
everywhere those numbers appear, not just in a footnote somewhere.

**Step 5: Calibrate against reality, and catch our own mistakes**
(`notebooks/03_fusion_and_report.ipynb`)

We measured the straight-line distance from every detected building to the nearest river,
then swept that distance until Kasarani's flagged count matched an independent Pamoja Trust
field survey of roughly 700 structures counted by hand. Sixteen metres got us closest.

We also caught a mistake in our own evaluation, and we'd rather tell you about it than bury
it. Early on, our model's accuracy panel reported 96%, but it was being scored on data it
had already seen during training, which flatters any model. Once we tested it properly, on a
held-out split it had never touched, the honest number was 84%. We think telling you that is
more useful than the impressive-sounding one.

### Summary results

| Region | Total screened | Passed filters | Encroaching (≤ 16m) | Ground-truth calibrated |
|---|---|---|---|---|
| **Kasarani** | 56,678 | 56,415 | **731** | Yes, matched against ~700 ground truth |
| **Gatharaini** | 27,123 | 27,046 | **87** | No, 16m cutoff applied untested |
| **Motoine** | 28,554 | 28,104 | **348** | No, 16m cutoff applied untested |

*Gatharaini and Motoine reuse the 16m threshold as an untested extrapolation until field
validation data becomes available for either.*

### Primary output artifacts

- `{region}_encroaching_buildings.csv`: tabular triage data, including coordinates,
  confidence scores, and metric river distances.
- `{region}_riparian_buffer.geojson` / `{region}_rivers.geojson`: spatial vector layers for
  GIS analysis and mapping.
- `pipeline_summary.json`: consolidated execution stats and calibration metadata.
- `rf_baseline.joblib`: the trained, shared Random Forest.

## Getting started

Full setup instructions, including how to run just the dashboard versus regenerating the
data end to end, are in [SETUP.md](SETUP.md).
