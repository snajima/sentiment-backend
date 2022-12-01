import torch
import transformers
import pandas as pd
import numpy as np
from pathlib import Path
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, 
                          PreTrainedModel, DistilBertModel, DistilBertForSequenceClassification,
                          TrainingArguments, Trainer)
from transformers.modeling_outputs import SequenceClassifierOutput

from datasets import load_dataset
emotions = load_dataset("go_emotions", "raw")

df = emotions['train'].to_pandas()
label_cols = ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral']

id2label = {str(i):label for i, label in enumerate(label_cols)}
label2id = {label:str(i) for i, label in enumerate(label_cols)}

## Preprocessing
df["labels"] = df[label_cols].values.tolist()

## Adding user data
df.loc[len(df.index)] = ['I hate my mom', 'eda8ds4', 'Stephen', 'confessions', 't3_ajki2b', 't1_eda65q2	', 1.548167e+09, 15, False, 0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]] 


df_sample = df.sample(n=1000)
mask = np.random.rand(len(df)) < 0.8
df_train = df[mask]
df_test = df[~mask]

## Tokenize and code
model_ckpt = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
train_encodings = tokenizer(df_train["text"].values.tolist(), truncation=True)
test_encodings = tokenizer(df_test["text"].values.tolist(), truncation=True)
train_labels = df_train["labels"].values.tolist()
test_labels = df_test["labels"].values.tolist()

class GoEmotionDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = GoEmotionDataset(train_encodings, train_labels)
test_dataset = GoEmotionDataset(test_encodings, test_labels)

## Fine-tuning
class DistilBertForMultilabelSequenceClassification(DistilBertForSequenceClassification):
    def __init__(self, config):
      super().__init__(config)

    def forward(self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.distilbert(input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict)

        hidden_state = outputs[0]
        pooled_output = hidden_state[:, 0]  
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = torch.nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), 
                            labels.float().view(-1, self.num_labels))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions)

num_labels=28
model = DistilBertForMultilabelSequenceClassification.from_pretrained(model_ckpt, num_labels=num_labels).to()

model.config.id2label = {
    "0": "admiration",
    "1": "amusement",
    "2": "anger",
    "3": "annoyance",
    "4": "approval",
    "5": "caring",
    "6": "confusion",
    "7": "curiosity",
    "8": "desire",
    "9": "disappointment",
    "10": "disapproval",
    "11": "disgust",
    "12": "embarrassment",
    "13": "excitement",
    "14": "fear",
    "15": "gratitude",
    "16": "grief",
    "17": "joy",
    "18": "love",
    "19": "nervousness",
    "20": "optimism",
    "21": "pride",
    "22": "realization",
    "23": "relief",
    "24": "remorse",
    "25": "sadness",
    "26": "surprise",
    "27": "neutral"
  },
model.config.label2id ={
    "admiration": 0,
    "amusement": 1,
    "anger": 2,
    "annoyance": 3,
    "approval": 4,
    "caring": 5,
    "confusion": 6,
    "curiosity": 7,
    "desire": 8,
    "disappointment": 9,
    "disapproval": 10,
    "disgust": 11,
    "embarrassment": 12,
    "excitement": 13,
    "fear": 14,
    "gratitude": 15,
    "grief": 16,
    "joy": 17,
    "love": 18,
    "nervousness": 19,
    "neutral": 27,
    "optimism": 20,
    "pride": 21,
    "realization": 22,
    "relief": 23,
    "remorse": 24,
    "sadness": 25,
    "surprise": 26
  }

def accuracy_thresh(y_pred, y_true, thresh=0.5, sigmoid=True): 
    y_pred = torch.from_numpy(y_pred)
    y_true = torch.from_numpy(y_true)
    if sigmoid: 
      y_pred = y_pred.sigmoid()
    return ((y_pred>thresh)==y_true.bool()).float().mean().item()

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    return {'accuracy_thresh': accuracy_thresh(predictions, labels)}

batch_size = 32
# configure logging so we see training loss
logging_steps = len(train_dataset) // batch_size

args = TrainingArguments(
    output_dir="emotion",
    evaluation_strategy = "epoch",
    save_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=logging_steps
)

trainer = Trainer(
    model,
    args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer)

trainer.evaluate()
trainer.train()






############################################################################################
# Choosing Weights for the Three Algorithms

# Initialize weights of all three algorithms to 1/3
# Run all three algorithms
# Multiply the results of each algorithm by the weights and add everything together
# Truncate to the top three results

# Store User's sentence and output as (x,y)

# After user decides between the top three, change the weights of the algorithm
# Initialize hyperparameters x, y with x > y that represents the weight of top two selected emotions (both x and y are less than 1, x + y = 1)
# For the algorithm that had the highest proportion of the top emotion, increase weight of the algorithm by its (weight of the top "emotion"
# - average of the emotion of the three algorithms) multiplied by x and (1-weight) 
# (i.e. if 0.5 of happy came from this algorithm, after first iteration, with total happy of 0.9, 
# increase weight by ((0.5-0.3) * 2/3 * x)
# if the difference between the average and the value is negative, then instead multiply by weight of model rather than 1 - weight
# Do the same for the two other algorithms
# Reweigh all algorithms on a scale of 0 to 1 by dividing by sum and multiplying by 3

# Do the same for B
# Create variables representing the proportion of weight relative to each 
# For each of the three emotions, multiply the previous hyperparameters (a, b, c) by 


# If frequent miscategorization, potentially switch to scheme where weights of algorithms are dependent on previous predictions of top emotions
# (i.e. if one category usually predicts sad, and the user constantly says we are predicting wrong, then we switch to a higher weight of sad predictor)
# Potentially decrease x, y once there is more information?


