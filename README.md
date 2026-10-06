# Credit Card Fraud Detection with a PyOD AutoEncoder

Unsupervised fraud detection for MSCS-633-A01 (Advanced Artificial Intelligence),
Hands-On Assignment 4.

An AutoEncoder from [PyOD](https://github.com/yzhao062/pyod) is trained on
anonymized credit card transactions without using the fraud labels. It learns to
rebuild ordinary transactions, so transactions it rebuilds badly (high
reconstruction error) are flagged as suspicious. The labels are only used
afterward to measure how well the flags match real fraud.

## Setup

1. Download the dataset from Kaggle (a free account is required):
   https://www.kaggle.com/datasets/whenamancodes/fraud-detection
   and save the file as `creditcard.csv` in this folder. It is not committed to
   the repository because of its size.

2. Create a virtual environment and install dependencies:
   ```
   python3 -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Run it:
   ```
   python fraud_detection.py
   ```

Options:
```
--data PATH         path to the CSV (default: creditcard.csv)
--epochs N          training epochs (default: 20)
--output-dir PATH   where the charts are saved (default: current folder)
```

## Output

The script prints the dataset size, ROC-AUC, a confusion matrix, and a
precision/recall report, and saves two charts:

- `score_distribution.png`: anomaly scores for legitimate vs. fraud transactions
- `precision_recall_curve.png`: precision against recall across thresholds

## Model

PyOD's `hidden_neuron_list` describes only the encoder half and the decoder is
mirrored automatically, so `[16, 8, 4]` builds this network:

```
30 inputs -> 16 -> 8 -> 4 (bottleneck) -> 8 -> 16 -> 30 outputs
```

## Notes

- The dataset is highly imbalanced (about 0.17% fraud), so accuracy is not
  reported. Precision, recall, and the precision-recall curve are more meaningful.
- `contamination` is set to the real fraud rate, which is a simplification for
  this experiment. A production system would pick the alert threshold based on
  the cost of a missed fraud versus a false alarm.
