# Monodepth-CIL

## Environment Setup
It is expected to have `test` and `train` 
data in their respective directories inside `data/`.

Load environment and install required packages
```shell
conda env create -f environment.yml
conda activate depth-world
pip install -r requirements.txt
```


To use V-Jepa, clone the official repository and correct the base URL
```shell
mkdir -p external
git clone https://github.com/facebookresearch/vjepa2.git external/vjepa2
sed -i 's|VJEPA_BASE_URL = "http://localhost:8300"|VJEPA_BASE_URL = "https://dl.fbaipublicfiles.com/vjepa2"|' \
  external/vjepa2/src/hub/backbones.py
  ```

Clone Depth-Anything-V2 and load checkpoint
```shell
cd external
git clone https://github.com/DepthAnything/Depth-Anything-V2.git external/Depth-Anything-V2
cd Depth-Anything-V2
pip install -r requirements.txt
cd ../..
curl -L \
  "https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth?download=true" \
  -o checkpoints/depth_anything_v2_vitb.pth
```

## Development

To update environment save files, run
```shell
conda env export --from-history > environment.yml
sed -i '/^prefix:/d' environment.yml 
pip freeze > requirements.txt
```