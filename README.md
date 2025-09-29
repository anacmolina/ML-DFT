# ab-flowMC

**ab-flowMC** is a flowMC-based method that accelerates the extraction of thermodynamic observables between free energy minima in molecular systems. By integrating Monte Carlo simulations and Density Functional Theory with the active learning normalizing flows and machine learning potentials, this method achieves efficient and accurate exploration of non-local minima structures with quantum-level precision.

## Installation

```bash
git clone https://github.com/anacmolina/ML-DFT.git
cd ML-DFT/packages/flonacomldft
pip install -e .
```

## Suggested Environment Setup

To create a clean environment and install the required dependencies, run the following commands in your terminal:

```
conda create -n ml-dft python
conda activate ml-dft

python -m pip install --upgrade pip setuptools wheel

#INSTALLING ase
conda install scipy
conda install matplotlib
conda install pandas
conda install tk
conda install Flask
conda install pytest
conda install pytest-mock
conda install pytest-xdist
python -m pip install spglib

python -m pip install ase

#INSTALLING gpaw
conda install -c conda-forge gpaw -y
gpaw install-data ./ --basis --version=gpaw-basis-pvalence-0.9.20000.tar.gz

#INSTALLING chemcoord
python -m pip install chemcoord

#INSTALLING pythorch and scikit-learn
conda install pytorch torchvision torchaudio cpuonly -c pytorch
conda install -c conda-forge scikit-learn
```

The code held in this repository was used for the following research:

**Active Learning of Boltzmann Samplers and Potential Energies with Quantum Mechanical Accuracy**  
*Ana Molina-Taborda, Pilar Cossio, Olga Lopez-Acevedo and Marylou Gabrié*  
*J. Chem. Theory Comput*. 2024, 20, 20, 8833–8843.  
DOI: https://doi.org/10.1021/acs.jctc.4c00506

There is also a preprint available at ArXiv.
DOI: https://doi.org/10.48550/arXiv.2401.16487