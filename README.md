# Convolutional Neural Networks for Scene Recognition

## Environment Setup

Clone this repository
```bash
git clone https://github.com/reddy-s/scene-recognition.git
```

Create conda virtual environment
```bash
conda create --name scene-recognition  python=3.11
```
Note: you can alternatively use virtualenv python module to create a virtual environment

Activate the environment by using 
```bash
conda activate scene-recognition
```

## Training and Testing

### Notebook
[01-train-scene-recognition.ipynb](./notebooks/01-train-scene-recognition.ipynb)

### Script for cloud training
```bash
python train.py
```

## Evaluation

### Notebook
[02-evaluate-scene-recognition.ipynb](./notebooks/02-evaluate-scene-recognition.ipynb)

## Clean up

Deactivate virtual environment 
```bash
conda deactivate
```

## Hyper-parameters to be explored
- Batch size
- Learning rate
- Number of epochs
- Optimizer
- Weight decay
- Scheduler

> Note: Microsoft Excel for us to collaborate: [Spreadsheet](https://surreyac-my.sharepoint.com/:x:/r/personal/sd01356_surrey_ac_uk/Documents/Scene%20Recognition%20HyperParamter%20Tuning%20Grid.xlsx?d=w4fdd462bf0214de7894352e7eb62b499&csf=1&web=1&e=o3ZNG7)

## Report Document for collaboration
[Report](https://docs.google.com/document/d/1SShA5zXqZq67t0BjLlNOIw_V5XNcfctQ6Bi9StusVZo/)

## Test data
[Scene Recognition Test Data](https://drive.google.com/drive/folders/1hYwXeMN9Q7ug_1M-KHtX6bfwcch8ckMK)