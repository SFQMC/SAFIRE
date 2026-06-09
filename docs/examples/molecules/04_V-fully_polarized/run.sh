#!/bin/bash

cwd=`pwd`

cd scf
python3 scf.py > scf.out
cd $cwd

cd ham
python3 ham.py > ham.out
cd $cwd

if [ ! -d afqmc ]; then
  mkdir afqmc
fi
cd afqmc
write_afqmc_json -i ../ham/afqmc.h5 -b 400
cd $cwd
