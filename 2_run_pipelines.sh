#!/bin/bash
########################################################################################################################
# 2_run_pipelines.sh - Runs the train and predict for each dataset to train and then generate predictions
########################################################################################################################

########################################################################################################################
# Data - Is Epic Intro
########################################################################################################################
python train.py -d "data/epic" -l "Labels.csv" -t "Is Epic" -o "models/epic/"
python predict.py -d "predict/epic" -l "predict.txt" -t "Is Epic" -m "models/epic/" -o "predict/predict_epic.csv"

########################################################################################################################
# Data - Needs Respray
########################################################################################################################
python train.py -d "data/respray" -l "Labels.csv" -t "Needs Respray" -o "models/respray/"
python predict.py -d "predict/respray" -l "predict.txt" -t "Needs Respray" -m "models/respray/" -o "predict/predict_respray.csv"

########################################################################################################################
# Data - Is GenAI
########################################################################################################################
python train.py -d "data/ai" -l "Labels.csv" -t "Is GenAI" -o "models/ai/"
python predict.py -d "predict/ai" -l "predict.txt" -t "Is GenAI" -m "models/ai/" -o "predict/predict_ai.csv"
