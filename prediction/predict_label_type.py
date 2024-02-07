from openai import OpenAI
from utils import load_csv_as_df
import numpy as np 
from loguru import logger 
from definitions import DETAILED_LABEL_DEFINITIONS, HIGH_LEVEL_LABEL_DEFINITIONS
from argparse import ArgumentParser
from pprint import pprint 
from tqdm import tqdm 
from sklearn.metrics import f1_score
from pathlib import Path


class OpenAIClassifier: 

    def __init__(self, model_name:str, consider_nones:bool=True):
        
        self.client = OpenAI() 
        self.model_name = model_name
        self.consider_nones = consider_nones
        
        self.detailed_label_definitions = DETAILED_LABEL_DEFINITIONS
        self.high_level_label_definitions = HIGH_LEVEL_LABEL_DEFINITIONS
        
        self._format_label_descriptions()
        
    def predict_label_with_target_sentence_only(self, target_sentence:str, show_input:bool=False)->str:
            
        prompt = self._form_target_sentence_only_prompt(target_sentence)
        messages = self._form_messages(prompt)
        
        if show_input: 
            logger.info(f"Messages:")
            pprint(messages)
        
        return self.predict(messages)
    
    def predict(self, messages)->str:
        
        chat_completion = self.client.chat.completions.create(
            messages= messages, 
            model=self.model_name,
        )
        
        return chat_completion.choices[0].message.content
    
    def _format_label_descriptions(self)->None: 
        
        self.high_level_label_description = ""
        for label, definition in self.high_level_label_definitions.items():
            if not self.consider_nones and label.lower() == "none":
                continue
            self.high_level_label_description += f"\t{label.lower()}: {definition}\n\n"
        
        self.detailed_label_description = "" 
        for label, definition in self.detailed_label_definitions.items(): 
            if not self.consider_nones and label.lower() == "none":
                continue
            self.detailed_label_description += f"\t{label.lower()}: {definition}\n\n"
        
    
    def _form_target_sentence_only_prompt(self, target_sentence:str) -> str: 
        
        prompt = f"Predict which type of update that the given target sentence will have given these label descriptions:\n{self.high_level_label_description}Target sentence: {target_sentence}\nLabel:"
        
        return prompt 
    
    def _form_messages(self, prompt): 
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
        return messages
        
        
        
def main(): 
    
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, help="Name of the model to use for prediction. Default: gpt-3.5-turbo", default="gpt-3.5-turbo")
    parser.add_argument("--prediction_method", type=str, help="One of [target_sentence_only, target_sentence_and_context]. Default: target_sentence_only", default="target_sentence_only")
    parser.add_argument("--test", action="store_true", help="Use small sample of test data for testing the script")
    parser.add_argument("--random_seed", type=int, help="Random seed to use for sampling test data. Default: 42", default=42)
    parser.add_argument("--testset_size", type=int, help="Size of test set to use for testing the script. Default: 1000", default=1000)
    parser.add_argument("--prediction_save_path", type=str, help="Path to save predictions. Default: predictions.csv", default="predictions.csv")
    parser.add_argument("--consider_nones", action="store_true", help="Consider None as a label. Default: True", default=True)
    args = parser.parse_args()
    
    # load test set data 
    data = load_csv_as_df("/Users/jcho/projects/NewsEdits/data/prediction_data/v2-silver-labeled-data-for-prediction-with-date-and-split.csv")
    logger.info(data.label.value_counts())

    # any task specific data processing 
    if args.prediction_method == "target_sentence_only": 
        # remove cases that have no target sentence
        data = data[~data["sentence"].isna()]
    
    # keep only the test set 
    test_set = data[data["split"] == "test"]

    # set random seed to get consistent results
    np.random.seed(42)
    
    # sample to only use args.testset_size
    if args.test:
        args.testset_size = 5 
    logger.info(f"Sampling {args.testset_size} from test set")
    test_set = test_set.sample(args.testset_size)   
    
    # get label distribution 
    label_distribution = test_set.label.value_counts()
    logger.info(label_distribution)
    
    # initialize classifier 
    openai_classifier = OpenAIClassifier(model_name=args.model_name, consider_nones=args.consider_nones)
    
    # predict label for test set 
    predictions =[] 
    first = True
    for idx, row in tqdm(test_set.iterrows(), total = len(test_set)):         

        target_sentence = row["sentence"]
        label = row["update_category"]
        prediction = openai_classifier.predict_label_with_target_sentence_only(target_sentence, show_input = first)
        logger.info(f"Target sentence: {target_sentence}\nLabel: {label}\nPrediction: {prediction}")
        predictions.append(prediction.lower())
        
        first = False 

    test_set["prediction"] = predictions
        
    category_labels = [i.lower() for i in test_set.update_category[:len(predictions)]]
    accuracy = sum([1 for i, j in zip(category_labels, predictions) if i == j])/len(category_labels) * 100 
    logger.info(f"Overall accuracy: {accuracy:.2f}\%")
    
    # get f1 score 
    f1 = f1_score(category_labels, predictions, average="weighted")
    logger.info(f"Overall F1 score: {f1:.2f}")
    
    # accuracy for each category (fact / style)
    category_distribution = test_set.update_category.value_counts()
    for category in category_distribution.index:
        category_accuracy = sum([1 for i, j in zip(category_labels, predictions) if i == j and i == category])/category_distribution[category] * 100
        logger.info(f"Accuracy for {category}: {category_accuracy:.2f}\%")
    
    # accuracy for each specific label
    for label in label_distribution.index: 
        # if prediction = update_category column, then it's correct for that specific label 
        test_set["correct"] = test_set["update_category"] == test_set["prediction"]
        label_accuracy = test_set[test_set["label"] == label]["correct"].sum() / label_distribution[label] * 100
        logger.info(f"Accuracy for {label}: {label_accuracy:.2f}\%") 
    
    # rename prediction filepath to include model name, random seed, and testset size
    prediction_save_path = Path(args.prediction_save_path)
    prediction_save_path = prediction_save_path.parent / f"{args.model_name}_seed{args.random_seed}_testset{args.testset_size}_{prediction_save_path.name}"
    
    #save predictions
    # if file path already exists, add the date to the end of the file name
    if prediction_save_path.exists():
        prediction_save_path = prediction_save_path.parent / f"{prediction_save_path.stem}_{datetime.now().strftime('%Y-%m-%d')}{prediction_save_path.suffix}"
        # if file path already exists, add number to the end of the file name
        if prediction_save_path.exists():
            i = 1
            while prediction_save_path.exists():
                prediction_save_path = prediction_save_path.parent / f"{prediction_save_path.stem}_{i}{prediction_save_path.suffix}"
                i += 1
    test_set.to_csv(prediction_save_path, index=False)
    
    return 

if __name__ == "__main__":
    main()  
    
    