# Cognitive Offloading Identification in Educational Chats with Generative AI

This repository contains the training and evaluation data, as well as test results for a transformer-based classifier that identifies light or heavy cognitive offloading to AI chatbots in educational conversations. The model was developed by Babak Hemmatian at the University of Nebraska-Lincoln, in collaboration with the Forward College in Berlin.

*Note:* Given our research question, this model was only trained on the first requests in conversations with generative AI, and would require further training for generalization to entire discussions.

## Data
The data is provided by researchers at Forward College, Berlin, Germany. For more information about the theories, procedures and findings, please reach out to [Liudmila Piatnitckaia](mailto:liudmila.piatnitckaia@forward-college.eu).
The training, evaluation and test data splits, including the original texts and the associated labels, can be found in the ```train_test_classifier_data_split``` folder. 

## Performance on a held-out 10% test set

### Macro Average:
precision: 0.80, recall: 0.81, f1: 0.80

### Per Class: 
No offloading: 
precision: 0.90, recall: 0.86, f1: 0.88
Light offloading: 
precision: 0.83, 'recall': 0.75, f1: 0.78
Heavy offloading:
precision: 0.68, recall: 0.81, f1: 0.74

## Usage

### Requirements
First ensure that the required packages listed in the requirements file are installed and, if included as part of a virtual environment, make sure that the enviroment is activated.

### Training and Model Evaluation
You can use the ```train_test_classifier.py``` script to train new models on the provided data or evaluate the existing models. Follow the detailed comments within the script for its proper use.

### Inference
The final trained model can be found [here](https://drive.google.com/drive/u/0/folders/11ljkn5eeM3fBzwiAS1IDLkPUmUp40oXx). Copy the entire folder into the path where this repository is located. Then run the code below in the command line where the ```-f``` argument shows the path to your input file. This file should be in ```.txt``` or ```.csv``` format, with each text for which you would like to get labels placed on a separate line. If using the ```.csv``` format, place the texts in the first column. The results will be saved to a file titled ```predicted_labels.csv```.
```
python -m infer.py -f [your_input_file_path]
```
