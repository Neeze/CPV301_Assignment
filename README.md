# Emotion Recognition

## Getting Started

### Prerequisites

- Python 3.10.13
- Libraries: opencv-python, scikit-learn, scikit-image, matplotlib, seaborn, numpy, pandas, tqdm, gdown


### Installation

1. Clone the repo

2. Install Dependency
```bash
pip install opencv-python scikit-learn scikit-image matplotlib seaborn numpy pandas tqdm gdown --quiet
```

### Usage
Data structure
``` bash
├── data
│   ├── test
│   │   ├── angry
│   │   ├── disgust
│   │   ├── fear
│   │   ├── happy
│   │   ├── neutral
│   │   ├── sad
│   │   └── surprise
│   └── train
│       ├── angry
│       ├── disgust
│       ├── fear
│       ├── happy
│       ├── neutral
│       ├── sad
│       └── surprise

```

For training purpose please follow steps in the notebook `notebooks\emotion-recognition.ipynb`

I have provided the pretrained model and you can download it from [Releases](https://github.com/Neeze/CPV301_Assignment/releases) and put it the `pretrained` folder

Use the `main.py` script. 
```bash
python main.py --model pretrained\emotion_recognition_mlp_model.pkl
```
press `q` to quit

