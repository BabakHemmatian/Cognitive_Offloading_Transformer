## import the needed packages
import csv
import random
import math

## custom function for doing a training/validation/test split. Relevant for train_relevance
def dataset_split(texts,labels,proportion):
    training_id = random.sample(range(len(texts)),math.floor(proportion*len(texts)))
    test_id = [i for i in range(len(texts)) if i not in training_id]
    training_texts = []
    training_labels = []
    test_texts = []
    test_labels = []
    
    for idx,i in enumerate(texts):
        if idx in training_id:
            training_texts.append(i)
            training_labels.append(labels[idx])
        elif idx in test_id:
            test_texts.append(i)
            test_labels.append(labels[idx])
        else:
            raise Exception
    return training_texts,test_texts,training_labels,test_labels

## writes training, evaluation and test data splits to csv files for reproducible training results. 
def split_dataset_to_file(file,list):
    with open(file,"w",encoding='utf-8',errors='ignore',newline="") as f:
        if "text" in file:
            writer = csv.writer(f)
            for i in list:
                writer.writerow([i])
        elif "label" in file:
            for i in list:
                print(i,file=f)

## reads training, evaluation and test data splits from csv files for reproducible training results.
def split_dataset_from_file(file):
    list_ = []
    with open(file,"r",encoding='utf-8',errors='ignore') as f:
        if "text" in file:
            reader = csv.reader(f)
            for i in reader:
                list_.append(i[0])
        elif "label" in file:
            for i in f:
                list_.append(int(i.strip()))
    return list_

## calculates and returns precision, recall and F1 for train_test_classifier.py models
def f1_calculator(labels, predictions):

    # check that the length of labels and predictions matches
    assert len(labels) == len(predictions), "Labels and predictions must be same length"

    # Ensure inputs are Python ints (not tensors or numpy types)
    labels = [int(l) for l in labels]
    predictions = [int(p) for p in predictions]

    # Unique class labels
    classes = sorted(set(labels) | set(predictions))

    # Initialize counters
    tp = {cls: 0 for cls in classes}
    fp = {cls: 0 for cls in classes}
    fn = {cls: 0 for cls in classes}

    # determine true positive, false positive and false negative counts
    for true, pred in zip(labels, predictions):
        if true == pred:
            tp[true] += 1
        else:
            fp[pred] += 1
            fn[true] += 1

    # Compute per-class metrics
    per_class = {}
    precisions, recalls, f1s = [], [], []

    for cls in classes:
        cls_tp = tp[cls]
        cls_fp = fp[cls]
        cls_fn = fn[cls]

        precision = cls_tp / (cls_tp + cls_fp) if (cls_tp + cls_fp) > 0 else 0.0
        recall = cls_tp / (cls_tp + cls_fn) if (cls_tp + cls_fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class[cls] = {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    # the overall performance across the test set
    macro_avg = {
        'precision': sum(precisions) / len(classes),
        'recall': sum(recalls) / len(classes),
        'f1': sum(f1s) / len(classes)
    }

    return {
        'per_class': per_class,
        'macro_avg': macro_avg
    }
