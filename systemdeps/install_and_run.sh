#!/bin/bash
echo "~~installing system dependencies~~"
sudo apt update
sudo apt install gpg curl git gcc make build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev wget llvm libncursesw5-dev \
xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev -y

echo 'install system dependencies for window and webcam'
sudo apt install libgirepository1.0-dev gcc libcairo2-dev pkg-config \
python3-dev gir1.2-gtk-4.0 -y

echo 'install drivers for USB dongle'
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" |
  sudo tee /etc/apt/sources.list.d/coral-edgetpu.list

curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -

sudo apt-get update
sudo apt-get install libedgetpu1-std -y

# switch with libedgetpu1-std for higher performance, but can run really hot!
# sudo apt-get install libedgetpu1-max

echo 'Please plug in (or remove and plug in) that adapter after installation'

echo "~~installing pyenv~~"
curl https://pyenv.run | bash

echo "~~adding pyenv to bashrc~~"
echo 'export PATH="$HOME/.pyenv/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc

echo "~~reloading bashrc~~"
source ~/.bashrc

echo "~~installing poetry~~"
curl -sSL https://install.python-poetry.org | python3 -
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc

echo "~~reloading bashrc~~"
source ~/.bashrc

# Validate that everything is working
cd ../api

echo "~~Install Python dependencies~~"
poetry install --no-root

echo "~~Run PyTest~~"
poetry run pytest

echo "~~Run API~~"
poetry run uvicorn main:app --reload