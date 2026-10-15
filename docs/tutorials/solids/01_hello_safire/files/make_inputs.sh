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

cp -r "$src/qe" "$src/coqui" "$work"

cd "$work/qe"
pw.x -inp scf.in
pw.x -inp nscf.in
pw2coqui.x < pw2coqui.in

cd "$work/coqui"
coqui hamil.toml
coqui wfn.toml

cp hamil.h5 "$files/hamiltonian.h5"
cp wfn.h5 "$files/wavefunction.h5"
