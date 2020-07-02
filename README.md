
## sdf-research
- heavily based on [DeepSDF](https://github.com/facebookresearch/DeepSDF)

### Installation
```
git clone
cd sdf-research

python3 -m venv venv
source venv/bin/activate
pip install -U wheel pip setuptools

pip install -r requirements.txt

export GOOGLE_APPLICATION_CREDENTIALS=$PWD/credentials/gs_viewer.json
dvc pull

# Loads third-party repositories
sudo apt install libeigen3-dev libgl1-mesa-dev libglew-dev
./build_deep_sdf.sh
```
#### Common issues
- [mpark/variant](https://github.com/facebookresearch/DeepSDF/issues/29#issuecomment-534558824)

- **nanoflann** includes
```
cd src/Utils.h
#include <naoflann/nanoflann.hpp>
->
#include <nanoflann.hpp>
```
- **headless** mode
```
PANGOLIN_WINDOW_URI=headless://
```

### Guide

#### 1. Download data
- **.obj** format

#### 2. Process Meshes
-  **normalize to a unit sphere** (in practice fit to sphere radius of 1/1.03)
- sample N (ex: 500 000) spatial points (more aggresive **near surface**, to capture details)

```
DEEP_SDF_BIN=/deep_sdf/bin ./preprocess_mesh.sh /path/to/meshes /path/to/out
```

- **TODO**: allow pass num_of_samples (now only 250 000, 500 000)

![K3D Visualization](https://github.com/jackersson/sdf-research/blob/master/docs/images/k3d_demo.png)

#### 3. Split to Train/Test
- **TODO**: make a proper script
```
from deep_sdf.utils.data import split_train_test, write_split_to_file

train_mesh, test_mesh = split_train_test(MESH_DIR)
write_split_to_file(DIR, train_mesh, CLASS_NAME, is_train=True)
write_split_to_file(DIR, test_mesh , CLASS_NAME, is_train=False)
```

#### 4. Define model config
- **TODO**: move to yaml config
```
NUM_LAYERS = 8
NUM_NEURONS = 512

CONFIG = {
    "two_d": False,
    "latent_size": 0,
    "dims": [512] * 8,
    "dropout": [0, 1, 2, 3, 4, 5, 6, 7],
    "dropout_prob": 0.2,
    "norm_layers": [0, 1, 2, 3, 4, 5, 6, 7],
    "latent_in": [],
    "xyz_in_all": False,
    "use_tanh": False,
    "latent_dropout": False,
    "weight_norm": True
}
```

#### 5. Run training
- **TODO**: convert to script
- **TODO**: use TFboard instead of livelossplot
- **TODO**: chamber distance for accuracy plot
- notebooks/sdf_basics_3d.ipynb


#### 6. Reconstruct


#### 7. Evaluation
##### 7.1 Sample visible mesh surface
```
DEEP_SDF_BIN=/deep_sdf/bin ./sample_visible_mesh_surface.sh /path/to/meshes /path/to/out
```
- Outputs *.npz with normalization params ['offset', 'scale']
- Outputs *.ply with point cloud near surface

##### 7.2 Calculate Chamfer Distance



##### Visualization
- [K3D](https://github.com/K3D-tools/K3D-jupyter/tree/master/examples)
- [trimesh](https://github.com/mikedh/trimesh)
- [pyrender](https://pyrender.readthedocs.io/en/latest/)

