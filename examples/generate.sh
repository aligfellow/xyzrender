#!/usr/bin/env bash
# Generate all example outputs from sample structures.

set -euo pipefail

OUT=$(realpath ${BASH_SOURCE[0]%/*})
DIR=$OUT/structures
IMG=$OUT/images
mkdir -p "$IMG"

echo "=== Presets ==="
xyzrender "$DIR/caffeine.xyz" -o "$IMG/caffeine_default.svg"
xyzrender "$DIR/caffeine.xyz" -o "$IMG/caffeine_default.png"
xyzrender "$DIR/caffeine.xyz" --config flat -o "$IMG/caffeine_flat.svg"
xyzrender "$DIR/caffeine.xyz" --config paton -o "$IMG/caffeine_paton.svg"

echo "=== Display options ==="
xyzrender "$DIR/ethanol.xyz" --hy -o "$IMG/ethanol_all_h.svg"           # all H
xyzrender "$DIR/ethanol.xyz" --hy 7 8 9 -o "$IMG/ethanol_some_h.svg"   # specific H atoms
xyzrender "$DIR/ethanol.xyz" --no-hy -o "$IMG/ethanol_no_h.svg"        # no H
xyzrender "$DIR/benzene.xyz" --hy -o "$IMG/benzene.svg"                 # aromatic
xyzrender "$DIR/caffeine.xyz" --bo -k -o "$IMG/caffeine_kekule.svg"    # Kekule bond orders

echo "=== VdW spheres ==="
xyzrender "$DIR/asparagine.xyz" --hy --vdw -o "$IMG/asparagine_vdw.svg"  # all atoms
xyzrender "$DIR/asparagine.xyz" --hy --vdw "1-6" -o "$IMG/asparagine_vdw_partial.svg"  # some atoms
xyzrender "$DIR/asparagine.xyz" --hy --vdw --config paton -o "$IMG/asparagine_vdw_paton.svg"  # all atoms

echo "=== QM output files ==="
xyzrender "$DIR/bimp.out" -o "$IMG/bimp_qm.svg" 
xyzrender "$DIR/mn-h2.log" -o "$IMG/mn-h2_qm.svg" --ts

echo "=== Input files ==="
xyzrender "$DIR/ala_phe_ala.pdb" -o "$IMG/ala_phe_ala.svg"
xyzrender --smi "C1CCCCC1" --hy -o "$IMG/cyclohexane_smi.svg"

echo "=== TS and NCI options ==="
xyzrender "$DIR/sn2.out" --ts-bond "1-2" -o "$IMG/sn2_ts_man.svg"
xyzrender "$DIR/sn2.out" --ts --hy -o "$IMG/sn2_ts.svg"
xyzrender "$DIR/bimp.out" --nci -o "$IMG/bimp_nci.svg"
xyzrender "$DIR/Hbond.xyz" --hy --nci-bond "8-9" -o "$IMG/nci_man.svg"  # specific NCI bond only
xyzrender "$DIR/Hbond.xyz" --hy --nci -o "$IMG/nci.svg"  # specific NCI bond only

echo "=== Annotations & measurements ==="
xyzrender "$DIR/caffeine.xyz" --idx -o "$IMG/caffeine_idx.svg" 
xyzrender "$DIR/caffeine.xyz" --idx n --hy --label-size 25 -o "$IMG/caffeine_idx_n.svg" 
xyzrender "$DIR/caffeine.xyz" --hy --cmap "$DIR/caffeine_charges.txt" -o "$IMG/caffeine_cmap.svg" --gif-rot -go "$IMG/caffeine_cmap.gif"
xyzrender "$DIR/caffeine.xyz" --hy --cmap "$DIR/caffeine_charges.txt" -o "$IMG/caffeine_cmap.svg" --cmap-range -0.5 0.5
xyzrender "$DIR/caffeine.xyz" -l 13 6 9 4 t -l 1 a -l 14 d -l 7 12 8 a -l 11 d -o "$IMG/caffeine_dihedral.svg"
xyzrender "$DIR/caffeine.xyz" -l 1 best -l 2 "NBO: 0.4" -o "$IMG/caffeine_labels.svg"
xyzrender "$DIR/sn2.out" --ts --label "$DIR/sn2_label.txt" -o "$IMG/sn2_ts_label.svg" --label-size 40

echo "=== Molecular orbitals ==="
xyzrender "$DIR/caffeine_lumo.cube" --mo --mo-colors maroon teal -o "$IMG/caffeine_lumo.svg"
xyzrender "$DIR/caffeine_homo.cube" --mo --hy --iso 0.03 -o "$IMG/caffeine_homo_iso_hy.svg"
xyzrender "$DIR/caffeine_homo.cube" --mo -o "$IMG/caffeine_homo.svg" --gif-rot -go "$IMG/caffeine_homo.gif"

echo "=== Density surface ==="
xyzrender "$DIR/caffeine_dens.cube" --dens --iso 0.01 -o "$IMG/caffeine_dens_iso.svg"
xyzrender "$DIR/caffeine_dens.cube" --dens --dens-color teal --opacity 0.75 -o "$IMG/caffeine_dens_custom.svg"
xyzrender "$DIR/caffeine_dens.cube" --dens -o "$IMG/caffeine_dens.svg" --gif-rot -go "$IMG/caffeine_dens.gif"

echo "=== ESP surface ==="
xyzrender "$DIR/caffeine_dens.cube" --esp "$DIR/caffeine_esp.cube" -o "$IMG/caffeine_esp.svg"
xyzrender "$DIR/caffeine_dens.cube" --esp "$DIR/caffeine_esp.cube" --iso 0.005 --opacity 0.75 -o "$IMG/caffeine_esp_custom.svg"

echo "=== GIF animations ==="
xyzrender "$DIR/caffeine.xyz" -o "$IMG/caffeine_gif.svg" --gif-rot -go "$IMG/caffeine.gif"
xyzrender "$DIR/caffeine.xyz" -o "$IMG/caffeine_xy.svg" --gif-rot xy -go "$IMG/caffeine_xy.gif"
xyzrender "$DIR/bimp.out" -o "$IMG/bimp_rot.svg" --gif-rot --gif-ts --vdw 84-169 -go "$IMG/bimp.gif"
xyzrender "$DIR/bimp.out" -o "$IMG/bimp_trj.svg" --gif-trj --ts -go "$IMG/bimp_trj.gif"
xyzrender "$DIR/mn-h2.log" -o "$IMG/mn-h2_gif.svg" --gif-ts -go "$IMG/mn-h2.gif"
xyzrender "$DIR/bimp.out" -o "$IMG/bimp_ts_nci.svg" --ts --gif-trj --vdw 84-169 --nci -go "$IMG/bimp_nci_trj.gif"
xyzrender "$DIR/bimp.out" -o "$IMG/bimp_ts_nci.svg" --gif-ts --gif-rot --vdw 84-169 --nci -go "$IMG/bimp_nci_ts.gif"

echo "=== Vector arrows ==="
xyzrender "$DIR/ethanol.xyz" --vectors "$DIR/ethanol_dip.json" -o "$IMG/ethanol_dip.svg" --gif-rot -go "$IMG/ethanol_dip.gif"           # dipole at center of mass, with rotation
xyzrender "$DIR/ethanol.xyz" --hy --vectors "$DIR/ethanol_forces_efield.json" --vector-scale 1.5 -o "$IMG/ethanol_forces_efield.svg" -go "$IMG/ethanol_forces_efield.gif" --gif-rot  # per-atom forces, with rotation

echo "=== Crystal / unit cell ==="
xyzrender "$DIR/caffeine_cell.xyz" --cell -o "$IMG/caffeine_cell.svg" --no-orient --gif-rot -go "$IMG/caffeine_cell.gif" 
xyzrender "$DIR/caffeine_cell.xyz" --cell-color maroon -o "$IMG/caffeine_cell_custom.svg" # custom edge color

echo "=== Crystal / periodic structures ==="
xyzrender "$DIR/NV63.vasp" --crystal vasp -o "$IMG/NV63_vasp.svg" --gif-rot -go "$IMG/NV63_vasp.gif"  # auto-detected as VASP
xyzrender "$DIR/NV63.in" --crystal qe --no-axes -o "$IMG/NV63_qe.svg"          # explicit QE mode
xyzrender "$DIR/NV63_cell.xyz" -o "$IMG/NV63_cell.svg"      
xyzrender "$DIR/NV63_cell.xyz" --no-ghosts --no-axes -o "$IMG/NV63_cell_no_ghosts.svg"       
xyzrender "$DIR/NV63_cell.xyz" --no-cell -o "$IMG/NV63_cell_no_cell.svg"       
xyzrender "$DIR/NV63_cell.xyz" --axis 001 -o "$IMG/NV63_001.svg"   # looking down [001]
xyzrender "$DIR/NV63_cell.xyz" --axis 111 --gif-rot 111 -o "$IMG/NV63_111.svg" -go "$IMG/NV63_111.gif"  # look down [111], rotate around [111]

echo "=== Overlays ==="
xyzrender "$DIR/isothio_xtb.xyz" --overlay "$DIR/isothio_uma.xyz" -c 1 --hy -o "$IMG/isothio_overlay.svg" --gif-rot -go "$IMG/isothio_overlay.gif"
xyzrender "$DIR/isothio_xtb.xyz" --overlay "$DIR/isothio_uma.xyz" -c 1 -o "$IMG/isothio_overlay_custom.svg" --no-orient --overlay-color green -a 2

echo "=== NCI surfaces ==="
xyzrender "$DIR/base-pair-dens.cube" --nci-surf "$DIR/base-pair-grad.cube" -o "$IMG/base-pair-nci_surf.svg"
xyzrender "$DIR/phenol_di-dens.cube" --nci-surf "$DIR/phenol_di-grad.cube" -o "$IMG/phenol_di-nci_surf.svg"

echo "Done! Outputs written to $IMG/"
