# Implementation amendments, before adaptation

## A1. Correct the expected clean source-data count

The pinned `PolyAI/banking77` revision contains 10,003 training rows and
3,080 test rows. All texts within each split are distinct. Literal UTF-8
text comparison finds **zero** train/test overlaps, so the registered exact
decontamination rule leaves **3,080**, not the anticipated 3,076.

The splitter initially rejected this discrepancy. Aggregate-only diagnostics
found six overlaps after `.strip()` and seven after NFKC/casefold/whitespace
normalization. Neither produces the anticipated count. We will not invent a
filter or delete four arbitrary examples to manufacture it.

The base author's original preparation code uses the different `mteb/banking77`
mirror. Its README's 3,076-row result is therefore not a count established for
our pinned PolyAI source. The original code was inspected at:
`https://raw.githubusercontent.com/Johnson-f/tinytriage/b63f579376d68b524b19b0b2c605bf8cdcfc028c/data/prepare.py`.

**Correction:** retain the specified pinned PolyAI revision, preserve exact
decontamination, and assert 3,080 clean rows. The same label-stratified
50/25/25 rule, seed, locked-exam protections, and measured-base bar remain.
The resulting expected sizes are 1,540 exam, 770 working, and 770 feedback
rows, subject to the generated manifest's actual counts.

This correction happened during one-shot preparation, before any split was
sealed, mask search, vocabulary construction, or working-set evaluation.
No exam messages or model outcomes were printed or used to make the choice.
Include this amendment in the public preregistration before adaptation.

## A2. Baseline comparator remains the requested literal protocol

The author's evaluation script uses 40 output tokens and answer normalization.
Our requested protocol specifies 20-token greedy literal exact match, plus
valid-label-constrained greedy, taking the higher measured score. We preserve
that protocol and disclose that it is not a byte-for-byte reproduction of
the author's original evaluation script. The card's 93.4% is never substituted
for the final measured bar.
