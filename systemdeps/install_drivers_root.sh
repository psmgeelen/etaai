#!/bin/bash
apt update
apt install gpg -y
echo 'install drivers for USB dongle'
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -
apt-get update
apt-get install libedgetpu1-std -y
# switch with libedgetpu1-std for higher performance, but can run really hot!
# sudo apt-get install libedgetpu1-max