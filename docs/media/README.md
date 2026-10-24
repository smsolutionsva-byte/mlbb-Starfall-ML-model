# Demo media and privacy edit

`weekly-rank-282793-redacted.png` is the supplied ranking screenshot with only the two username fields redacted. The final image retains the source's 1600 × 717 resolution. The built-in image-editing tool produced the redactions; only those two strips were composited into the original decoded screenshot. A pixel comparison checked **zero changes outside the redaction rectangles**, preserving the rank, scores, other entries, and game artwork. Coordinates and checksums are recorded in `provenance.json`.

`gameplay-preview.jpg` is a frame extracted at 45 seconds from the annotated replay. Its boxes and suggested path are post-capture analysis. The full MP4 is a release asset, rather than a Git-tracked video.

## Image-editing prompt

> Edit target: the attached 1600 x 717 pixel MLBB ranking screenshot. Privacy redaction only. Apply a strong Gaussian blur/pixelation to BOTH occurrences of the rank-1 player's username: first in the blue rank-1 row approximately x=1165..1370 y=215..258, second in the white highlighted self row approximately x=1165..1370 y=595..638. Make the entire username unreadable in both places. Do not obscure or change the avatar, rank 1 medals/numbers, Points column, or the exact score 282793, which occurs twice. Keep all other pixels, framing, resolution, artwork, other players' names and scores, headings, and UI unchanged. No beautification, repainting, annotations, extra labels, altered numbers or invented details. Output the same screenshot with only those two localized name strips redacted.

See [the benchmark page](../RESULTS.md) for the maintainer-reported 282,793-point result from the private, unreleased model, the earlier public detector used for visualization, and the media provenance.
