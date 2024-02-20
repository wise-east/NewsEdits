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
# format data for longformer training and preparing data that will be formatted for openAI prediction (gpt_finetuning.py --format_data)
# strict_fact is the stricter set of labels that correspond to fact
python format_data.py --label_type [all, fact, strict_fact] --version [2, 3] --home_dir <path to repo>
```

### Longformer Fine-tuning

Used Steeve's [codebase](https://github.com/khuangaf/newsedit_pp) as starting point. Inside `newsedit_pp/modeling` directory, run the following command:

```bash
source run_led.sh
```

### Zero-shot & Fine-tuned OpenAI prediction

```bash
# predicting with OpenAI models
python predict_label_type.py --model_name [gpt-3.5-turbo, gpt-4-0125-preview, <finetuned model name>] --prompt_type [sentence_only, direct_context, full_article]

# format data for gpt_finetuning, saves all prompt type formats. Requires file from format_data.py
python gpt_finetuning.py --format_data

# upload file and submit finetuning job to OpenAI. set fp to the file path of the training data from the previous step
python gpt_finetuning.py --prompt_type [sentence_only, direct_context, full_article] --fp <training file fp> --model [gpt-3.5-turbo]
```


# Get prediction results
```bash
# compute detailed metrics for predictions.
# can handle both test_generation.txt files from longformer training results and predictions from OpenAI models
python compute_f1.py --predictions_fp <fp>
```


### Strict fact label results with Longformer:

```
Sentence only:
              precision    recall  f1-score   support

        fact       0.10      0.28      0.15        32
        none       0.93      0.80      0.86       391

    accuracy                           0.76       423
   macro avg       0.52      0.54      0.51       423
weighted avg       0.87      0.76      0.81       423


Direct context:
--- not trained fully ---

Full article:
              precision    recall  f1-score   support

        fact       0.13      0.53      0.21        32
        none       0.95      0.72      0.82       391

    accuracy                           0.70       423
   macro avg       0.54      0.62      0.51       423
weighted avg       0.89      0.70      0.77       423

```

### fact (fact + background) results with longformer

```
Sentence only:

              precision    recall  f1-score   support

        fact       0.26      0.17      0.20        53
        none       0.89      0.93      0.91       370

    accuracy                           0.83       423
   macro avg       0.57      0.55      0.56       423
weighted avg       0.81      0.83      0.82       423

Direct context:


        fact       0.18      0.28      0.22        53
        none       0.89      0.81      0.85       370

    accuracy                           0.74       423
   macro avg       0.53      0.55      0.53       423
weighted avg       0.80      0.74      0.77       423


Full article:
              precision    recall  f1-score   support

        fact       0.23      0.57      0.33        53
        none       0.92      0.74      0.82       370

    accuracy                           0.71       423
   macro avg       0.58      0.65      0.57       423
weighted avg       0.84      0.71      0.76       423

```

## Strict fact label results with GPT-3.5 zero-shot

```
Sentence only:
        fact       0.07      0.28      0.11        32
        none       0.92      0.69      0.79       391

    accuracy                           0.66       423
   macro avg       0.33      0.32      0.30       423
weighted avg       0.86      0.66      0.74       423


Direct context:
                precision    recall  f1-score   support

          fact       0.03      0.03      0.03        32
factual update       0.00      0.00      0.00         0
          none       0.92      0.92      0.92       391

      accuracy                           0.85       423
     macro avg       0.32      0.32      0.32       423
  weighted avg       0.85      0.85      0.85       423

Full article:
             precision    recall  f1-score   support

        fact       0.07      0.09      0.08        32
        none       0.92      0.91      0.91       391

    accuracy                           0.84       423
   macro avg       0.50      0.50      0.50       423
weighted avg       0.86      0.84      0.85       423
```

## Strict fact label results with GPT-4 zero-shot
```
Sentence only:
              precision    recall  f1-score   support

        fact       0.06      0.41      0.11        32
        none       0.91      0.52      0.66       391

    accuracy                           0.51       423
   macro avg       0.49      0.46      0.39       423
weighted avg       0.85      0.51      0.62       423

Direct context:
              precision    recall  f1-score   support

        fact       0.11      0.22      0.15        32
        none       0.93      0.86      0.89       391

    accuracy                           0.81       423
   macro avg       0.52      0.54      0.52       423
weighted avg       0.87      0.81      0.84       423

Full article:

              precision    recall  f1-score   support

        fact       0.12      0.19      0.15        32
        none       0.93      0.89      0.91       391

    accuracy                           0.84       423
   macro avg       0.53      0.54      0.53       423
weighted avg       0.87      0.84      0.85       423
```

### v5 results with longformer

```
# strict fact, sentence only
              precision    recall  f1-score   support

        fact       0.19      0.25      0.21        32
        none       0.94      0.91      0.92       391

    accuracy                           0.86       423
   macro avg       0.56      0.58      0.57       423
weighted avg       0.88      0.86      0.87       423

# strict fact, direct context
              precision    recall  f1-score   support

        fact       0.20      0.25      0.22        32
        none       0.94      0.92      0.93       391

    accuracy                           0.87       423
   macro avg       0.57      0.58      0.57       423
weighted avg       0.88      0.87      0.87       423


# strict fact, full article
              precision    recall  f1-score   support

        fact       0.20      0.34      0.25        32
        none       0.94      0.89      0.91       391

    accuracy                           0.85       423
   macro avg       0.57      0.62      0.58       423
weighted avg       0.89      0.85      0.86       423

```

```
# with style
              precision    recall  f1-score   support

        fact       0.23      0.13      0.17        53
        none       0.72      0.25      0.38       306
       style       0.18      0.78      0.29        64

    accuracy                           0.32       423
   macro avg       0.37      0.39      0.28       423
weighted avg       0.58      0.32      0.34       423

              precision    recall  f1-score   support

        fact       0.21      0.19      0.20        53
        none       0.71      0.29      0.41       306
       style       0.16      0.64      0.26        64

    accuracy                           0.33       423
   macro avg       0.36      0.37      0.29       423
weighted avg       0.56      0.33      0.36       423


              precision    recall  f1-score   support

        fact       0.26      0.17      0.20        53
        none       0.67      0.21      0.32       306
       style       0.17      0.77      0.27        64

    accuracy                           0.29       423
   macro avg       0.37      0.38      0.27       423
weighted avg       0.54      0.29      0.30       423

```

## TODO

<!-- checklist -->
- [] combine predict_label_type and gpt_finetuning into one script
