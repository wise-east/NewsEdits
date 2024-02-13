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

### Zero-shot & Fine-tuned OpenAI prediction 

```bash
python predict_label_type.py --model_name <model> --prompt_type <prompt_type> 
```

### Fine-tuned prediction 

Use Steeve's codebase as starting point. Inside `newsedit_pp` directory, run the following command: 

```bash
source run_led.sh
```



```
Test set info: Counter({'none': 1654, 'fact': 211, 'style': 135})
```

### GPT-3.5 Results 

```
Sentence only: 
2024-02-11 18:16:24.035 | INFO     | __main__:main:145 - Overall accuracy: 48.45\%
2024-02-11 18:16:24.048 | INFO     | __main__:main:149 - Overall F1 score: 0.56
2024-02-11 18:16:24.053 | INFO     | __main__:main:156 - F1 score for fact: 0.63
2024-02-11 18:16:24.055 | INFO     | __main__:main:156 - F1 score for style: 0.24
2024-02-11 18:16:24.064 | INFO     | __main__:main:156 - F1 score for none: 0.68

2024-02-11 22:59:55.681 | INFO     | __main__:main:147 - Overall accuracy: 45.45\%
2024-02-11 22:59:55.701 | INFO     | __main__:main:151 - Overall F1 score: 0.53
2024-02-11 22:59:55.704 | INFO     | __main__:main:158 - F1 score for style: 0.19
2024-02-11 22:59:55.706 | INFO     | __main__:main:158 - F1 score for fact: 0.53
2024-02-11 22:59:55.713 | INFO     | __main__:main:158 - F1 score for none: 0.66

Direct context: 
2024-02-11 18:31:01.978 | INFO     | __main__:main:145 - Overall accuracy: 66.75\%
2024-02-11 18:31:01.991 | INFO     | __main__:main:149 - Overall F1 score: 0.68
2024-02-11 18:31:01.994 | INFO     | __main__:main:156 - F1 score for fact: 0.22
2024-02-11 18:31:01.996 | INFO     | __main__:main:156 - F1 score for style: 0.07
2024-02-11 18:31:02.002 | INFO     | __main__:main:156 - F1 score for none: 0.88

2024-02-11 23:18:50.596 | INFO     | __main__:main:147 - Overall accuracy: 66.85\%
2024-02-11 23:18:50.604 | INFO     | __main__:main:151 - Overall F1 score: 0.68
2024-02-11 23:18:50.606 | INFO     | __main__:main:158 - F1 score for fact: 0.23
2024-02-11 23:18:50.612 | INFO     | __main__:main:158 - F1 score for none: 0.88
2024-02-11 23:18:50.614 | INFO     | __main__:main:158 - F1 score for style: 0.10

Full article: 
2024-02-11 18:51:40.048 | INFO     | __main__:main:147 - Overall accuracy: 66.85\%
2024-02-11 18:51:40.076 | INFO     | __main__:main:151 - Overall F1 score: 0.69
2024-02-11 18:51:40.083 | INFO     | __main__:main:158 - F1 score for fact: 0.22
2024-02-11 18:51:40.080 | INFO     | __main__:main:158 - F1 score for style: 0.19
2024-02-11 18:51:40.100 | INFO     | __main__:main:158 - F1 score for none: 0.88
```

### Fine-tuned GPT-3.5
```
Sentence only: 
2024-02-11 20:56:33.654 | INFO     | __main__:main:147 - Overall accuracy: 35.85\%
2024-02-11 20:56:33.666 | INFO     | __main__:main:151 - Overall F1 score: 0.44
2024-02-11 20:56:33.670 | INFO     | __main__:main:158 - F1 score for fact: 0.52
2024-02-11 20:56:33.679 | INFO     | __main__:main:158 - F1 score for none: 0.52
2024-02-11 20:56:33.681 | INFO     | __main__:main:158 - F1 score for style: 0.62

Direct context: 
2024-02-11 20:48:04.393 | INFO     | __main__:main:147 - Overall accuracy: 36.30\%
2024-02-11 20:48:04.404 | INFO     | __main__:main:151 - Overall F1 score: 0.45
2024-02-11 20:48:04.407 | INFO     | __main__:main:158 - F1 score for style: 0.61
2024-02-11 20:48:04.411 | INFO     | __main__:main:158 - F1 score for fact: 0.50
2024-02-11 20:48:04.420 | INFO     | __main__:main:158 - F1 score for none: 0.53

Full article: 
2024-02-11 22:19:09.682 | INFO     | __main__:main:147 - Overall accuracy: 52.90\%
2024-02-11 22:19:09.694 | INFO     | __main__:main:151 - Overall F1 score: 0.60
2024-02-11 22:19:09.698 | INFO     | __main__:main:158 - F1 score for style: 0.33
2024-02-11 22:19:09.702 | INFO     | __main__:main:158 - F1 score for fact: 0.57
2024-02-11 22:19:09.710 | INFO     | __main__:main:158 - F1 score for none: 0.73
```

### GPT-4 Results: 
```
Sentence only: 
2024-02-11 19:12:25.392 | INFO     | __main__:main:147 - Overall accuracy: 14.65\%
2024-02-11 19:12:25.403 | INFO     | __main__:main:151 - Overall F1 score: 0.12
2024-02-11 19:12:25.406 | INFO     | __main__:main:158 - F1 score for fact: 0.84
2024-02-11 19:12:25.412 | INFO     | __main__:main:158 - F1 score for none: 0.11
2024-02-11 19:12:25.414 | INFO     | __main__:main:158 - F1 score for style: 0.47

Direct context: 
2024-02-11 20:53:47.388 | INFO     | __main__:main:147 - Overall accuracy: 59.10\%
2024-02-11 20:53:47.402 | INFO     | __main__:main:151 - Overall F1 score: 0.64
2024-02-11 20:53:47.406 | INFO     | __main__:main:158 - F1 score for fact: 0.27
2024-02-11 20:53:47.412 | INFO     | __main__:main:158 - F1 score for none: 0.81
2024-02-11 20:53:47.414 | INFO     | __main__:main:158 - F1 score for style: 0.28

Full article: 
2024-02-11 22:47:18.482 | INFO     | __main__:main:147 - Overall accuracy: 63.80\%
2024-02-11 22:47:18.495 | INFO     | __main__:main:151 - Overall F1 score: 0.66
2024-02-11 22:47:18.507 | INFO     | __main__:main:158 - F1 score for none: 0.85
2024-02-11 22:47:18.509 | INFO     | __main__:main:158 - F1 score for style: 0.25
2024-02-11 22:47:18.510 | INFO     | __main__:main:158 - F1 score for fact: 0.19
```

### Fine-tuned longformer results 

```
Sentence only:
Overall accuracy: 55.05%
Overall F1 score: 0.60
F1 score for fact: 0.69
F1 score for style: 0.89
F1 score for none: 0.63

Direct context:
Overall accuracy: 52.75%
Overall F1 score: 0.59
F1 score for fact: 0.67
F1 score for style: 0.89
F1 score for none: 0.60

Full article:
Overall accuracy: 54.00%
Overall F1 score: 0.59
F1 score for fact: 0.68
F1 score for style: 0.89
F1 score for none: 0.61

```