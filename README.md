blastwave 
============
[![blastwave](https://github.com/jhmatthews/blastwave/actions/workflows/test_figures.yml/badge.svg)](https://github.com/jhmatthews/blastwave/actions/workflows/test_figures.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15011341.svg)](https://doi.org/10.5281/zenodo.15011341)

This repository contains scripts and data for the paper 

**"Blast waves and reverse shocks: from ultra-relativistic GRBs to moderately relativistic X-ray binaries"**, _James H. Matthews, Alex J. Cooper, Lauren Rhodes, Katherine Savard, Rob Fender, Francesco Carotenuto, Fraser J. Cowie, Emma L. Elley, Joe Bright, Andrew K. Hughes and Sara E. Motta_, Submitted to MNRAS. 

Usage 
=========
To set up your environment, use `pip -r requirements.txt`, ideally within a virtual environment. 

To make the figures, run 
```
python scripts/MakeAllFigures.py
```

This will make the majority, although not quite all, of the figures from the paper. Fig. 1 is not included at this stage because the data will be published as part of Fender & Motta (submitted). 
The schematic figures are also not included but can be shared on request. 

The script will put the figures in the figures/ directory. 

Directory structure
=====================

* figures/ -- folder for figures
* scripts/ -- plotting scripts
* data/ -- majority of data for the paper
* simulations/ -- outputs from relativistic hydro simulations
