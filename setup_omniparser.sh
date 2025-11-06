#!/bin/bash

git clone https://github.com/VishnuDurairaj/omniparser-fastapi.git
cd omniparser-fastapi
python3.12 -m venv env
source env/bin/activate
pip install -r requirements.txt
python download_weights.py
python main.py
