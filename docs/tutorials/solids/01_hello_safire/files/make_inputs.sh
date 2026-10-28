#!/bin/bash
# hamiltonian.h5 and wavefunction.h5: primitive-cell Si at the Gamma point with
# 20 bands, from a Quantum ESPRESSO calculation converted by CoQui. The QE and
# CoQui inputs are those of the solids/03_Si_writing_H_and_wfn_coqui tutorial.
#
# Needs pw.x, pw2coqui.x and coqui on the PATH.
set -euo pipefail

files=$(cd "$(dirname "$0")" && pwd)
src=$files/../../03_Si_writing_H_and_wfn_coqui/files
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

# same layout as the tutorial: run in data/, inputs in ../files/
mkdir -p "$work/files" "$work/data"
cp -r "$src/qe" "$src/coqui" "$work/files"

cd "$work/data"
pw.x -inp ../files/qe/scf.in
pw.x -inp ../files/qe/nscf.in
pw2coqui.x < ../files/qe/pw2coqui.in

coqui ../files/coqui/hamil.toml
coqui ../files/coqui/wfn.toml

cp hamil.h5 "$files/hamiltonian.h5"
cp wfn.h5 "$files/wavefunction.h5"
