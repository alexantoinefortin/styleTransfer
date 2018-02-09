# Author: Alex-Antoine Fortin
# Date: Monday, March 21st 2017
# Description
# Command line to install TF with GPU on Ubuntu 16.04

# ssh into your instance
ssh -C aws-cdiscount
#update list of available pkgs thru apt-get
sudo apt-get update
sudo apt-get upgrade
sudo apt-get dist-upgrade
sudo apt autoremove

# build-essential
sudo apt-get -y install htop
sudo apt-get -y install build-essential
sudo apt-get -y install cmake curl git unzip pkg-config

# mathematical efficiency modules
sudo apt-get -y install libatlas-dev
sudo apt-get -y install libatlas-doc
sudo apt-get -y install libblas-dev
sudo apt-get -y install libblas-doc
sudo apt-get -y install liblapack-dev
sudo apt-get -y install liblapack-doc

#gcc
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt-get update
sudo apt-get -y install gcc-5 g++-5
sudo apt-get update
sudo apt-get upgrade
sudo apt-get dist-upgrade
sudo apt autoremove
# fortran
sudo apt-get -y install gfortran
#sudo apt-get -y install texlive

# Python dependencies
sudo apt-get -y install libreadline-gplv2-dev
sudo apt-get -y install libncursesw5-dev
sudo apt-get -y install libssl-dev
sudo apt-get -y install libsqlite3-dev
sudo apt-get -y install tk-dev
sudo apt-get -y install libgdbm-dev
sudo apt-get -y install libc6-dev
sudo apt-get -y install libbz2-dev

# to make checkinstall work
#sudo apt-get install auto-apt
# Python
#mkdir ~/Downloads
cd ~/Downloads
wget https://www.python.org/ftp/python/3.6.4/Python-3.6.4.tgz
# Extract and go to directory
tar -xvf Python-3.6.4.tgz
cd Python-3.6.4
./configure --enable-unicode=ucs4 --enable-optimizations
make
sudo make install

#check
gcc -v

# Python package
cd ~/Downloads
pip3 --version
sudo apt-get -y install python3-dev
sudo pip3 install cython
sudo pip3 install numpy
sudo pip3 install scipy --no-cache-dir
sudo pip3 install pandas
sudo pip3 install scikit-learn
sudo pip3 install statsmodels --no-cache-dir
sudo pip3 install tqdm
sudo pip3 install pillow
sudo pip3 install protobuf
sudo pip3 install wheel
sudo pip3 install six
sudo pip3 install h5py
sudo pip3 install pymongo
sudo pip3 install numba
# kaggle-specific
sudo apt-get -y install python-lxml
sudo pip3 install kaggle-cli

# Java
java -version
sudo apt-get -y install default-jdk

# Bazel (used to install TF)
echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -
sudo apt-get update && sudo apt-get -y install bazel
# Upgrade to newer version of bazel
sudo apt-get upgrade bazel

 ###  ### ##       ### ###
##########################
# 0. GPU cuda install
# 0.1 CUDA Pre-installation actions
# http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#pre-installation-actions
# Verify that we have a GPU-capable instance
lspci | grep -i nvidia ## Run the next 2 lines if it does not print anything
#cd /sbin
#update-pciids
uname -m && cat /etc/*release ## Should print something
gcc --version  ## Should not return an error message
uname -r ## Version of the kernel header that we should choose for the CUDA install
# Installs kernel headers and development pacakges
sudo apt-get -y install linux-headers-$(uname -r)
sudo apt-get -y install --upgrade linux-headers-$(uname -r)

# 0.2 CUDA installation actions
# Downloads the version of CUDA that you need
# See https://developer.nvidia.com/cuda-downloads for guidance
# NOTE: At this point, I was using 8/8Gig of space and needed to bump up my EBS-volume in the  aws.amazon.com (Select volume on the left-hand side menu in the DashBoard)
# NOTE (continued): I needed to "reboot" the instance once I added the extra storage space
# NOTE (continued): useful command "df -h", "du -h", "sudo du -xh / | sort -h | tail -40"
cd ~/Downloads
#wget https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64-deb
#sudo dpkg -i cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64-deb

wget https://developer.nvidia.com/compute/cuda/9.0/Prod/local_installers/cuda-repo-ubuntu1604-9-0-local_9.0.176-1_amd64-deb
sudo dpkg -i cuda-repo-ubuntu1604-9-0-local_9.0.176-1_amd64-deb
sudo apt-key add /var/cuda-repo-9-0-local/7fa2af80.pub
sudo apt-get update
sudo apt-get install cuda-toolkit-9.0

nano ~/.bashrc
# Just copy paste that in .bashrc then save & exit
export PATH=/usr/local/cuda-9.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
source ~/.bashrc

# 0.3 CUDA post-installation actions
# http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#post-installation-actions
#cuDNN
#NOTE: create account and download cuDNN from hyperSearch
#NOTE (cont.): https://developer.nvidia.com/rdp/cudnn-download
#from your aws instance
cd ~/Downloads
wget https://developer.nvidia.com/compute/machine-learning/cudnn/secure/v7.0.5/prod/9.0_20171129/cudnn-9.0-linux-x64-v7
tar -zxvf cudnn-9.0-linux-x64-v7.tgz
sudo cp cuda/include/cudnn.h /usr/local/cuda-9.0/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda-9.0/lib64
sudo chmod a+r /usr/local/cuda-9.0/include/cudnn.h /usr/local/cuda-9.0/lib64/libcudnn*


# Testing out cuDNN
cd /usr/local/cuda/samples/5_Simulations/nbody
sudo make
./nbody -benchmark -numbodies=2560000 -device=0

cd /usr/local/cuda/samples/1_Utilities/deviceQuery
sudo make
./deviceQuery

# One extra tool requiered for TF
sudo apt-get -y install libcupti-dev
#sudo pip install tensorflow-gpu
#sudo pip install keras
# Get TF from source (could be replaced by sudo pip install tensorflow-gpu)
cd ~/Downloads
git clone https://github.com/tensorflow/tensorflow
cd tensorflow
git checkout r1.5
./configure ## Used cuDNN version 5.1.10
#bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package
bazel build --config=mkl -c opt --copt=-march=native --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-msse4.2 --copt=-mfpmath=both --config=cuda --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" //tensorflow/tools/pip_package:build_pip_package
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
#mv tensorflow-1.2.0rc1-cp27-cp27mu-linux_x86_64.whl /tmp/tensorflow_pkg/tensorflow-1.2.0rc1-cp27-cp27mu-linux_x86_64.whl
# wheel file name's depend on your OS and other stuff, double check.
#sudo pip install /tmp/tensorflow_pkg/tensorflow-1.2.1-cp27-cp27m-macosx_10_6_x86_64.whl
cd /tmp/tensorflow_pkg
sudo pip3 install tensorflow-1.5.0-cp36-cp36m-linux_x86_64.whl
sudo pip3 install keras
#No gpu
#sudo pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.0.1-cp27-none-linux_x86_64.whl
