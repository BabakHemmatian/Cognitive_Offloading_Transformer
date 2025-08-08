## import needed packages
import argparse
import os 
import torch
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification
import csv

## set path variable to the current folder
dir_path = os.path.dirname(os.path.realpath(__file__))

## Create an argument parser for feeding in data file names from the command line
argparser = argparse.ArgumentParser(description="Applies the trained offloading model to any dataset.")
argparser.add_argument("-f", "--file", type=str, required=True, help="The name of the data file. Path is assumed to be the same as the folder in which this script is located.")

## determine the model to load
model_name = "offloading_roberta-base_final"
model_path = os.path.join(dir_path, model_name)
tokenizer = RobertaTokenizerFast.from_pretrained(model_name,do_lower_case=True)
max_length = 512

## Load the texts to be classified from the file indicated in the command line argument
texts = []
with open(argparser.parse_args().file, "r", encoding='utf-8', errors='ignore') as f:
    if ".txt" in argparser.parse_args().file:
        # Read non-empty lines from a text file
        for line in f:
            if line.strip():
                texts.append(line.strip())
    elif ".csv" in argparser.parse_args().file:
        # Read non-empty lines from a CSV file
        reader = csv.reader(f)
        for row in reader:
            if row:
                if row[0].strip():
                    texts.append(row[0].strip())
    else:
        raise ValueError("Unsupported data file format. Please provide a .txt or .csv file.")

## extract the embeddings for the tokenized input
encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)

## determine if GPU acceleration is available and set the processing device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Performing computations on {device}") 

## transfer the model to the processing device
model = RobertaForSequenceClassification.from_pretrained(model_path).to(device)

## determine label mapping
label_title = {0:"Not Offloading", 1:"Light Offloading", 2:"Heavy Offloading", 3:"Other (write comment)"}

## Define a function to get the prediction for a single text input. Input should be a string. Returns a tuple containing the predicted class and the probabilities for all classes.
def get_prediction(text, max_length=max_length):
    # Tokenize the text
    # Assuming 'tokenizer', 'max_length', and 'device' are defined globally
    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    ).to(device)

    # Model inference
    # Assuming 'model' is defined globally
    outputs = model(**inputs)
    probs = outputs[0].softmax(1)[0]  # Take the first row (batch size = 1)
    return probs.argmax().item(), [probs[i].item() for i in range(len(probs))]

## Get model predictions for all input texts
predictions = [get_prediction(text) for text in texts]

## save the text, the assigned label and the probabilities for all labels to disk
with open(os.path.join(dir_path, "predicted_labels.csv"), "w", encoding='utf-8', errors='ignore',newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["text", "predicted_label", "No offloading prob", "Light Offloading prob", "Heavy Offloading prob"])
    for text, (predicted_label, probs) in zip(texts, predictions):
        writer.writerow([text, label_title[predicted_label]] + probs)
