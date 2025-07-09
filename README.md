### Data Mining Final Project

- Uses Sentinel-2 Satellite Imagery
- `all.qgz` is a QGIS project containing the masked raw imagery and truth labels.

#### Raw Data source:

- https://browser.dataspace.copernicus.eu/

#### Labeled Data source:

- https://livingatlas.arcgis.com/landcoverexplorer

#### Use WSL! Highly Recommended :D!

- Required Directories
```bash
mkdir stacked
mkdir models
mkdir raw
cd raw
mkdir 2019
mkdir 2020
```

- Environment Setup
```bash
conda create -n landcover-unet
conda activate landcover-unet
conda install -c conda-forge --file conda-requirements.txt
pip install -r requirements.txt # Global install :D
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 # Install CUDA pytorch for faster training! (If you have NVIDIA GPU)
```

- Execution Sequence
```bash
# Read the inline comments!
cd src
# Merge the raw imagery first
python merge.py
# Run model training scripts
python train-resnet34.py
python train-effnetb4.py
# Evaluate the winning model :D
python eval.py
```
