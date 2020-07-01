```
git clone
cd sdf-research

python3 -m venv venv
source venv/bin/activate
pip install -U wheel pip setuptools

pip install -r requirements.txt
```

```

git clone https://github.com/facebookresearch/DeepSDF.git
cd DeepSDF

git submodule update --init

# Eigen3
sudo apt install libeigen3-dev

cd third-party

# CLI11
git clone https://github.com/CLIUtils/CLI11.git
cd CLI11
git submodule update --init

mkdir build
cd build
cmake ..
make -j
sudo make install
cd ../..

# nanoflann
git clone https://github.com/jlblancoc/nanoflann.git
cd nanoflann/

mkdir build
cd build
cmake ..
make -j
sudo make install
cd ../..

# Pangolin
git clone https://github.com/stevenlovegrove/Pangolin.git
cd Pangolin

mkdir build
cd build
cmake ..
make -j
sudo make install
cd ../..


# DeepSDF
cd ..
mkdir build
cd build
# https://github.com/facebookresearch/DeepSDF/issues/29#issuecomment-534558824
cmake .. -DCMAKE_CXX_STANDARD=17
make -j
```

```
cd src/Utils.h
#include <naoflann/nanoflann.hpp>
->
#include <nanoflann.hpp>
```


**TODO** allow set number of points with command line
- generates train points (~500 000)

- generates test points (250 000)
```
./bin/PreprocessMesh -m /home/taras/coder/projects/sdf-research/data/planes/0_simplified.obj -o 0_simplified_test.npy -t
```