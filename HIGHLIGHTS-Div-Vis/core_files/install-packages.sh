#!/bin/sh

#  install-packages.sh
#  
#
#  Created by Britt D on 8/17/20.
#

# Bash "strict mode", to help catch problems and bugs in the shell
# script. Every bash script you write should include this. See
# http://redsymbol.net/articles/unofficial-bash-strict-mode/ for
# details.
# set -euo pipefail

# Tell apt-get we're never going to be able to give manual
# feedback:
export DEBIAN_FRONTEND=noninteractive

# Update the package listing, so we know what package exist:
apt-get update

# Install security updates:
apt-get -y upgrade

# Install a new package, without unnecessary recommended packages:
pip3 install syslog-ng
pip3 install pandas==0.25.3
pip3 install numpy==1.17.4
pip3 install tensorwatch
pip3 install scikit-learn
pip3 install seaborn==0.10.0
pip3 install chainer
pip3 install h5py
pip3 install matplotlib==3.1.1
pip3 install scikit-image==0.16.2
pip3 install gym==0.15.4
pip3 install keras==2.2.4
pip3 install tensorflow==1.15.2
pip3 install innvestigate==1.0.8
pip3 install opencv-python==4.1.1.26
pip3 install scipy==1.3.3
pip3 install joblib==0.14.1
pip3 install coloredlogs==14.0
pip3 install torch==1.6.0
pip3 install python-varname==0.4.0
pip3 install gym[atari]

# Delete cached files we don't need anymore:
apt-get clean
rm -rf /var/lib/apt/lists/*
