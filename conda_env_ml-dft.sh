
#Installing CONDA
wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh
sha256sum Anaconda3-2022.05-Linux-x86_64.sh
bash Anaconda3-2022.05-Linux-x86_64.sh
~/anaconda3/bin/conda init
source ~/.bashrc
conda update -n base -c defaults conda

#CONDA environment: ml-dft
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
#Sometimes you need to install gcc previously
python -m pip install chemcoord

#INSTALLING pythorch and scikit-learn
conda install pytorch torchvision torchaudio cpuonly -c pytorch
conda install -c conda-forge scikit-learn
