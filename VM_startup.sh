#!/bin/sh

mkdir /home/Victor/model
wget -P /home/Victor/model https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-24_H-1024_A-16.zip && unzip /home/Victor/model/uncased_L-24_H-1024_A-16.zip -d /home/Victor/model/
sudo apt-get install python3-venv
python3 -m venv /home/Victor/Bert2
source "/home/Victor/Bert2/bin/activate"
pip install --upgrade pip
pip install tensorflow-gpu==1.15
pip install -U bert-serving-server bert-serving-client
pip install requests
pip install -U bert-serving-server[http]
bert-serving-start -model_dir=/home/Victor/model/uncased_L-24_H-1024_A-16/ -num_worker=1 -http_port 5000

