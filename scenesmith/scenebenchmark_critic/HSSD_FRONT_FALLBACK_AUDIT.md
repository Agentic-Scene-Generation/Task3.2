# HSSD Semantic-Front Fallback Audit

## Conclusion

The wardrobe issue is part of a larger unfinished front-review layer.
The current full lookup contains:

```text
total assets                                      10,963
semantic-front assets                              1,047
fallback assets                                    9,916
fallback with week27 access_sides=["front"]       3,899
fallback without that policy                      6,017
high-priority semantic-front candidates            2,931
other front-policy candidates                        968
```

All 3,899 policy candidates have complete `front.png`, `back.png`, `left.png`,
`right.png`, `iso.png`, and `top.png` render evidence under the shared
`hssd_rendered_assets` root. The candidate audit therefore identifies work that
can be reviewed, rather than treating missing renders as the reason for a
fallback.

## High-priority candidates

These categories have a functional face that normally has semantic orientation:

```text
wall art              1,091
wall mirror             255
chest of drawers        229
desk                    170
kitchen cabinet         151
bookcase                135
cabinet                 128
armoire                 118
wall lamp               118
wall clock               87
hanging cabinet          79
sink cabinet             68
refrigerator             54
china cabinet            43
fireplace                43
toilet                   34
wall shelf               33
sink                     22
television receiver      21
washer                   20
microwave                 9
stove                     7
dishwasher                6
monitor                   6
oven                      2
dryer                     1
toaster oven              1
```

The high-priority tier is a review queue, not an automatic promotion rule.
Different assets in the same category can have mirrored geometry, multiple
access faces, or no strict positive front. Each promoted record must retain a
view index, evidence image, confidence, and separate
`is_strict_positive_front` label, as the wardrobe correction now does.

## Other policy candidates

The remaining 968 records have `access_sides=["front"]` but their category name
does not match the strong functional-face vocabulary. They include object
classes such as stools, curtains, mirrors, decor, and small appliances. Some
have a meaningful viewing/access face; others have only a weak or
context-dependent orientation. They require the same asset-level review before
promotion.

## Machine-readable outputs

- `data/FRONT_FALLBACK_AUDIT.json`: full category-level counts, render coverage,
  sample IDs, and tier definitions.
- `data/FRONT_FALLBACK_CANDIDATES.csv`: all 3,899 candidate IDs with category,
  tier, current front fields, render status, and render directory.
- `scripts/audit_front_fallbacks.py`: reproducible audit; it does not mutate
  annotation records.

Example:

```bash
python scripts/audit_front_fallbacks.py \
  --lookup data/hssd_annotation_lookup.json.gz \
  --render-root /path/to/hssd_rendered_assets \
  --output-json data/FRONT_FALLBACK_AUDIT.json \
  --output-csv data/FRONT_FALLBACK_CANDIDATES.csv
```

## Current boundary

This audit deliberately does not promote the 3,899 candidates automatically.
The wardrobe patch was evidence-backed by reviewing all 17 assets and selecting
the correct view per asset. A full promotion requires the same review protocol,
including coordinate mapping and strict-positive labeling, for each candidate.
