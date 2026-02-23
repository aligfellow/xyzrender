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
xyzrender "$DIR/caffeine.xyz" --bo -k -o "$OUT/caffeine_kekule.svg"    # Kekule bond orders

echo "=== VdW spheres ==="
xyzrender "$DIR/asparagine.xyz" --hy --vdw -o "$OUT/asparagine_vdw.svg"  # all atoms
xyzrender "$DIR/asparagine.xyz" --hy --vdw "1-6" -o "$OUT/asparagine_vdw_partial.svg"  # some atoms
xyzrender "$DIR/asparagine.xyz" --hy --vdw --config paton -o "$OUT/asparagine_vdw_paton.svg"  # all atoms

echo "=== QM output files ==="
xyzrender "$DIR/bimp.out" -o "$OUT/bimp_qm.svg" 
xyzrender "$DIR/mn-h2.log" -o "$OUT/mn-h2_qm.svg" --ts

echo "=== TS and NCI options ==="
xyzrender "$DIR/sn2.out" --ts-bond "1-2" -o "$OUT/sn2_ts_man.svg" 
xyzrender "$DIR/sn2.out" --ts --hy -o "$OUT/sn2_ts.svg" 
xyzrender "$DIR/Hbond.xyz" --hy --nci-bond "8-9" -o "$OUT/nci_man.svg"  # specific NCI bond only
xyzrender "$DIR/Hbond.xyz" --hy --nci -o "$OUT/nci.svg"  # specific NCI bond only
xyzrender "$DIR/bimp.out" --nci -o "$OUT/bimp_nci.svg"  # all NCI bonds

echo "=== GIF animations ==="
xyzrender "$DIR/caffeine.xyz" -o "$OUT/caffeine_gif.svg" --gif-rot -go "$OUT/caffeine.gif"
xyzrender "$DIR/caffeine.xyz" -o "$OUT/caffeine_xy.svg" --gif-rot xy -go "$OUT/caffeine_xy.gif"
xyzrender "$DIR/bimp.out" -o "$OUT/bimp_rot.svg" --gif-rot --gif-ts --vdw 84-169 -go "$OUT/bimp.gif"
xyzrender "$DIR/bimp.out" -o "$OUT/bimp_trj.svg" --gif-trj --ts -go "$OUT/bimp_trj.gif"
xyzrender "$DIR/mn-h2.log" -o "$OUT/mn-h2_gif.svg" --gif-ts -go "$OUT/mn-h2.gif"
xyzrender "$DIR/bimp.out" -o "$OUT/bimp_nci.svg" --ts --gif-trj --vdw 84-169 --nci -go "$OUT/bimp_nci_trj.gif"
xyzrender "$DIR/bimp.out" -o "$OUT/bimp_nci.svg" --gif-ts --gif-rot --vdw 84-169 --nci -go "$OUT/bimp_nci_ts.gif"

echo "Done! Outputs written to $OUT/"
