# News Edits Update Type Prediction 

Used Python 3.10 

## Set environment 

```bash
conda create -n newsedits_prediction python=3.10
conda activate newsedits_prediction
pip install -r requirements.txt
```

### Format data 

```bash
python format_data.py
``` 

### Zero-shot prediction 

```bash
python predict_label_type.py
```

### Fine-tuned prediction 

Use Steeve's codebase as starting point. Inside `newsedit_pp` directory, run the following command: 

```bash
source run_led.sh
```



### GPT-3.5 Results 

```
2024-02-06 16:04:49.986 | INFO     | __main__:main:127 - Overall accuracy: 62.30%
2024-02-06 16:04:50.010 | INFO     | __main__:main:131 - Overall F1 score: 0.61
2024-02-06 16:04:50.012 | INFO     | __main__:main:137 - Accuracy for fact: 78.06%
2024-02-06 16:04:50.013 | INFO     | __main__:main:137 - Accuracy for style: 25.17%
2024-02-06 16:04:50.021 | INFO     | __main__:main:144 - Accuracy for Update Background: 78.70%
2024-02-06 16:04:50.024 | INFO     | __main__:main:144 - Accuracy for Tonal Edits: 24.91%
2024-02-06 16:04:50.026 | INFO     | __main__:main:144 - Accuracy for Delete Background: 84.40%
2024-02-06 16:04:50.028 | INFO     | __main__:main:144 - Accuracy for Event Update: 79.17%
2024-02-06 16:04:50.029 | INFO     | __main__:main:144 - Accuracy for Quote Update: 5.88%
2024-02-06 16:04:50.030 | INFO     | __main__:main:144 - Accuracy for Syntax Correction: 40.00%
2024-02-06 16:04:50.031 | INFO     | __main__:main:144 - Accuracy for Delete Quote: 42.86%
2024-02-06 16:04:50.032 | INFO     | __main__:main:144 - Accuracy for Quote Addition: 0.00%
2024-02-06 16:04:50.032 | INFO     | __main__:main:144 - Accuracy for Emphasize a Point: 0.00%
2024-02-06 16:04:50.033 | INFO     | __main__:main:144 - Accuracy for Add Analysis: 0.00%
2024-02-06 16:04:50.034 | INFO     | __main__:main:144 - Accuracy for Style-Guide Edits: 0.00%
2024-02-06 16:04:50.035 | INFO     | __main__:main:144 - Accuracy for Update Analysis: 0.00%
```

### GPT-4 Results: 
```
2024-02-06 16:31:49.020 | INFO     | __main__:main:127 - Overall accuracy: 61.30%
2024-02-06 16:31:49.055 | INFO     | __main__:main:131 - Overall F1 score: 0.59
2024-02-06 16:31:49.057 | INFO     | __main__:main:137 - Accuracy for fact: 78.77%
2024-02-06 16:31:49.058 | INFO     | __main__:main:137 - Accuracy for style: 20.13%
2024-02-06 16:31:49.067 | INFO     | __main__:main:144 - Accuracy for Update Background: 80.70%
2024-02-06 16:31:49.071 | INFO     | __main__:main:144 - Accuracy for Tonal Edits: 19.65%
2024-02-06 16:31:49.073 | INFO     | __main__:main:144 - Accuracy for Delete Background: 82.80%
2024-02-06 16:31:49.074 | INFO     | __main__:main:144 - Accuracy for Event Update: 87.50%
2024-02-06 16:31:49.075 | INFO     | __main__:main:144 - Accuracy for Quote Update: 11.76%
2024-02-06 16:31:49.076 | INFO     | __main__:main:144 - Accuracy for Syntax Correction: 40.00%
2024-02-06 16:31:49.077 | INFO     | __main__:main:144 - Accuracy for Delete Quote: 14.29%
2024-02-06 16:31:49.078 | INFO     | __main__:main:144 - Accuracy for Quote Addition: 0.00%
2024-02-06 16:31:49.078 | INFO     | __main__:main:144 - Accuracy for Emphasize a Point: 0.00%
2024-02-06 16:31:49.079 | INFO     | __main__:main:144 - Accuracy for Add Analysis: 0.00%
2024-02-06 16:31:49.080 | INFO     | __main__:main:144 - Accuracy for Style-Guide Edits: 0.00%
2024-02-06 16:31:49.080 | INFO     | __main__:main:144 - Accuracy for Update Analysis: 0.00%
```
