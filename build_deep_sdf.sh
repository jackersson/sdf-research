# Eigen3
apt-get -y --no-install-recommends install \
    libeigen3-dev \
    libglew-dev \

ROOT=$PWD

cd deep_sdf
mkdir third-party
cd third-party

# cnpy
git clone https://github.com/rogersce/cnpy.git

# CLI11
git clone https://github.com/CLIUtils/CLI11.git
cd CLI11
git submodule update --init

mkdir build
cd build
cmake ..
make -j $(nproc)
make install
cd $ROOT/deep_sdf/third-party

# nanoflann
git clone https://github.com/jlblancoc/nanoflann.git
cd nanoflann/

mkdir build
cd build
cmake ..
make -j $(nproc)
make install
cd $ROOT/deep_sdf/third-party

# Pangolin
git clone https://github.com/stevenlovegrove/Pangolin.git
cd Pangolin

mkdir build
cd build
cmake .. -DBUILD_TOOLS=OFF -DBUILD_EXAMPLES=OFF
make -j $(nproc)
make install
cd $ROOT/deep_sdf/third-party


# DeepSDF
cd ..
mkdir build
cd build
# https://github.com/facebookresearch/DeepSDF/issues/29#issuecomment-534558824
cmake .. -DCMAKE_CXX_STANDARD=17
make -j $(nproc)