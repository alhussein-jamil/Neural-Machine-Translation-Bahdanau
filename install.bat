@echo off

conda create -n projetmla python=3.10 -y

conda activate projetmla

pip install --upgrade -r requirements.txt

pip install --upgrade -e ./src

pytest
echo Environment setup completed.
