#!/usr/bin/env bash
# Generate all example outputs from sample structures.
# Run from the repo root: bash examples/generate.sh

set -euo pipefail

DIR=examples/structures
OUT=examples
mkdir -p "$OUT"

echo "=== Presets ==="
xyzrender "$DIR/caffeine.xyz" -o "$OUT/caffeine_default.svg"
xyzrender "$DIR/caffeine.xyz" -o "$OUT/caffeine_default.png"
xyzrender "$DIR/caffeine.xyz" --config flat -o "$OUT/caffeine_flat.svg"
xyzrender "$DIR/caffeine.xyz" --config paton -o "$OUT/caffeine_paton.svg"

echo "=== Display options ==="
xyzrender "$DIR/ethanol.xyz" --hy -o "$OUT/ethanol_all_h.svg"           # all H
xyzrender "$DIR/ethanol.xyz" --hy 7 8 9 -o "$OUT/ethanol_some_h.svg"   # specific H atoms
xyzrender "$DIR/ethanol.xyz" --no-hy -o "$OUT/ethanol_no_h.svg"        # no H
xyzrender "$DIR/benzene.xyz" --hy -o "$OUT/benzene.svg"                 # aromatic

echo "=== VdW spheres ==="
xyzrender "$DIR/asparagine.xyz" --hy --vdw -o "$OUT/asparagine_vdw.svg"              # all atoms

echo "=== QM output files ==="
xyzrender "$DIR/bimp.out" -o "$OUT/bimp_qm.svg" 
xyzrender "$DIR/mn-h2.log" -o "$OUT/mn-h2_qm.svg" --ts

echo "=== GIF animations ==="
xyzrender "$DIR/caffeine.xyz" -o "$OUT/caffeine_gif.svg" --gif-rot -go "$OUT/caffeine.gif"
xyzrender "$DIR/caffeine.xyz" -o "$OUT/caffeine_xy.svg" --gif-rot xy -go "$OUT/caffeine_xy.gif"
xyzrender "$DIR/bimp.out" -o "$OUT/bimp_rot.svg" --gif-rot --gif-ts --vdw 84-169 -go "$OUT/bimp.gif"
xyzrender "$DIR/bimp.out" -o "$OUT/bimp_trj.svg" --gif-trj --ts -go "$OUT/bimp_trj.gif"
xyzrender "$DIR/mn-h2.log" -o "$OUT/mn-h2_gif.svg" --gif-ts -go "$OUT/mn-h2.gif"

echo "Done! Outputs written to $OUT/"
