import torch
import transformers
import pandas as pd
import numpy as np
from pathlib import Path
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, 
                          PreTrainedModel, DistilBertModel, 
                          DistilBertForSequenceClassification,
                          TrainingArguments, Trainer)
from transformers.modeling_outputs import SequenceClassifierOutput
from datasets import load_dataset
from keras import utils
from keras import models
from keras import preprocessing
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, GRU

############################### TABLE OF CONTENTS ##############################
# 1/ Global variables and classes
# 2/ Intializing dataframe
# 3/ Training various algorithms
# 4/ Predicting using various algorithms
# 5/ Adding user data to dataframe

######################## 1/ GLOBAL VARIABLES AND CLASSES #######################
## Class to be used for algorithm 1
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

## Class to be used for algorithm 1
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

## Accuracy metric used for training algorithm 1
def accuracy_thresh(y_pred, y_true, thresh=0.5, sigmoid=True): 
    y_pred = torch.from_numpy(y_pred)
    y_true = torch.from_numpy(y_true)
    if sigmoid: 
      y_pred = y_pred.sigmoid()
    return ((y_pred>thresh)==y_true.bool()).float().mean().item()

## Accuracy calculation used for training algorithm 1
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    return {'accuracy_thresh': accuracy_thresh(predictions, labels)}

## Global variables
label_cols = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief', 
    'remorse', 'sadness', 'surprise', 'neutral'
    ]
id2label = {str(i):label for i, label in enumerate(label_cols)}
label2id = {label:str(i) for i, label in enumerate(label_cols)}
num_labels = 28
sample_amount = 100000
num_words = 10000
input_length = 200
batch_size = 128
model_1 = "distilbert-base-uncased"
tokenizer_1 = AutoTokenizer.from_pretrained(model_1)
tokenizer_2 = preprocessing.text.Tokenizer(num_words=num_words)
top_guess_count = 0
# Initialize weights of all three algorithms to 1/3
alg1_w = alg2_w = alg3_w = 1/3
model_path_1 = "./models/algo_1"
model_path_2 = "./models/algo_2"
model_path_3 = "./models/algo_3"

########################### 2/ INTIALIZING DATAFRAME ###########################
def intialization():
    emotions = load_dataset("go_emotions", "raw")
    df = emotions['train'].to_pandas()
    df["labels"] = df[label_cols].values.tolist()
    return df

######################## 3/ TRAINING VARIOUS ALGORITHMS ########################
############################# Training Algorithm 1 #############################
## df_global is full dataset
## sample_size is size of sample to train on
## model is string name of model used
def training_algorithm_1(df, sample_size, model):
    
    # Preprocessing
    df["labels"] = df[label_cols].values.tolist()
    df = df.sample(n=sample_size)
    mask = np.random.rand(len(df)) < 0.8
    df_global_train = df[mask]
    df_global_test = df[~mask]

    # Creating model
    model = DistilBertForMultilabelSequenceClassification.from_pretrained(
        model, num_labels=num_labels, id2label=id2label, 
        label2id=label2id).to()

    # Tokenize
    train_global_encodings = tokenizer_1(
        df_global_train["text"].values.tolist(), truncation=True)
    test_global_encodings = tokenizer_1(
        df_global_test["text"].values.tolist(), truncation=True)
    train_global_labels = df_global_train["labels"].values.tolist()
    test_global_labels = df_global_test["labels"].values.tolist()
    
    train_global_dataset = GoEmotionDataset(
        train_global_encodings, train_global_labels)
    test_global_dataset = GoEmotionDataset(
        test_global_encodings, test_global_labels)

    # Setting parameter values
    batch_size = 32
    logging_steps = len(train_global_dataset) // batch_size

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

    # Training and saving model
    global_trainer = Trainer(
        model,
        args,
        train_dataset=train_global_dataset,
        eval_dataset=test_global_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer_1)

    global_trainer.train()
    # global_trainer.evaluate()
    global_trainer.save_model("./models/algo_1")

############################# Training Algorithm 2 #############################
## df_global is full dataset
## sample_size is size of sample to train on
## num_words is max number of words in vocabulary
## input_length is max number of words in one input
## batch_size is batch size
def training_algorithm_2(
    df_global, sample_size, num_words, input_length, batch_size):
    
    # Preprocessing
    df_global["labels"] = df_global[label_cols].values.tolist()
    df_global = df_global.sample(n=sample_size)
    mask = np.random.rand(len(df_global)) < 0.8
    df_global_train = df_global[mask]
    df_global_test = df_global[~mask]
    train_text = df_global_train["text"].values.tolist()
    test_text = df_global_test["text"].values.tolist()

    # Tokenize
    tokenizer_2.fit_on_texts(train_text)
    train_encodings = tokenizer_2.texts_to_sequences(train_text)
    train_x = utils.pad_sequences(train_encodings, maxlen=input_length)
    
    tokenizer_2.fit_on_texts(test_text)
    test_encodings = tokenizer_2.texts_to_sequences(test_text)
    test_x = utils.pad_sequences(test_encodings, maxlen=input_length)
    
    train_y = np.array(df_global_train["labels"].values.tolist())
    test_y = np.array(df_global_test["labels"].values.tolist())

    # Creating neural network
    model = Sequential()
    model.add(
        Embedding(input_dim=num_words, output_dim=128, input_length=input_length))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(0.5))
    model.add(Dense(28, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 

    # Training and saving model
    model.fit(
        train_x, train_y, batch_size=batch_size, epochs=12,
        validation_data=[test_x, test_y])
    model.save("./models/algo_2")

############################# Training Algorithm 3 #############################
## df_global is full dataset
## sample_size is size of sample to train on
## num_words is max number of words in vocabulary
## input_length is max number of words in one input
## batch_size is batch size
def training_algorithm_3(
    df_global, sample_size, num_words, input_length, batch_size):
    
    # Preprocessing
    df_global["labels"] = df_global[label_cols].values.tolist()
    df_global = df_global.sample(n=sample_size)
    mask = np.random.rand(len(df_global)) < 0.8
    df_global_train = df_global[mask]
    df_global_test = df_global[~mask]
    train_text = df_global_train["text"].values.tolist()
    test_text = df_global_test["text"].values.tolist()

    # Tokenize
    tokenizer_2.fit_on_texts(train_text)
    train_encodings = tokenizer_2.texts_to_sequences(train_text)
    train_x = utils.pad_sequences(train_encodings, maxlen=input_length)
    
    tokenizer_2.fit_on_texts(test_text)
    test_encodings = tokenizer_2.texts_to_sequences(test_text)
    test_x = utils.pad_sequences(test_encodings, maxlen=input_length)
    
    train_y = np.array(df_global_train["labels"].values.tolist())
    test_y = np.array(df_global_test["labels"].values.tolist())

    # Creating neural network
    model = Sequential()
    model.add(
        Embedding(input_dim=num_words, output_dim=128, input_length=input_length))
    model.add(Bidirectional(GRU(64)))
    model.add(Dropout(0.8))
    model.add(Dense(28, activation='relu'))
    model.add(Dense(28, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 

    # Training and saving model
    model.fit(
        train_x, train_y, batch_size=batch_size, epochs=12,
        validation_data=[test_x, test_y])
    model.save("./models/algo_3")

#################### 4/ PREDICTING USING VARIOUS ALGORITHMS ####################
############################ Prediction Algorithm 1 ############################
## model_path is string name of where model is saved
## txt is string to be predicted
def run_algorithm_1(model_path, txt):
    
    # Tokenize and loading model
    encoding = tokenizer_1(txt, return_tensors="pt")
    model = DistilBertForMultilabelSequenceClassification.from_pretrained(model_path)
    trainer = Trainer(model)
    encoding = {k: v.to(trainer.model.device) for k,v in encoding.items()}

    # Returning list of emotions ranked from most to least probable emotion
    outputs = trainer.model(**encoding)
    logits = outputs.logits
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(logits.squeeze().cpu())
    probs_lst = probs.detach().numpy()
    # sort_index = np.argsort(-probs_lst)
    # predicted_labels = [id2label[str(idx)] for idx in sort_index]
    return probs_lst

########################### Prediction Algorithm 2/3 ###########################
## model_path is string name of where model is saved
## txt is string to be predicted 
## num_words is max number of words in vocabulary
## input_length is max number of words in one input
def run_algorithm_2_3(model_path, txt, num_words, input_length):
    
    # Tokenize and load model
    txt_lst = []
    txt_lst.append(txt)
    tokenizer_2.fit_on_texts(txt_lst)
    tok_txt = tokenizer_2.texts_to_sequences(txt_lst)
    pad_txt = utils.pad_sequences(tok_txt, maxlen=input_length)
    model = models.load_model(model_path)

    # Returning list of emotions ranked from most to least probable emotion
    probs_lst = model.predict(pad_txt)
    # sort_index = np.argsort(-probs_lst[0])
    # list(filter(None, sort_index))
    # predicted_labels = [id2label[str(idx)] for idx in sort_index]
    return probs_lst

####################### 5/ ADDING USER DATA TO DATAFRAME #######################
# df is dataframe
# sentence is a sentence/paragraph in string format
# label is an emotion in string format
def append_df(df, sentence, label):
    idx_num = int(label2id[label])
    label_list = [0] * 28
    label_list[idx_num] = 1
    df.loc[len(df.index)] = [
        sentence, 'eda8ds4', 'Stephen', 'confessions', 't3_ajki2b', 
        't1_eda65q2', 2, 15, False, 0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0, label_list]
    return df

#################################### TESTING ###################################
if __name__ == "__main__":
    
    df = intialization()

    # Training algorithms
    # training_algorithm_1(df, sample_amount, model_1)
    # training_algorithm_2(
    #     df, sample_amount, num_words, input_length, batch_size)
    training_algorithm_3(
        df, sample_amount, num_words, input_length, batch_size)

    # Adding input sentence to training set 
    label = "anger"
    sentence = "I hate going to school"
    df = append_df(df, sentence, label)

    # Making a prediction on an input sentence
    model_path_1 = "./models/algo_1"
    model_path_2 = "./models/algo_2"
    model_path_3 = "./models/algo_3"
    txt = "I hate it here."
    pred_1 = run_algorithm_1(model_path_1, txt)
    pred_2 = run_algorithm_2_3(
        model_path_2, txt, num_words, input_length)
    pred_3 = run_algorithm_2_3(
        model_path_3, txt, num_words, input_length)

    print("Sentence predicting:", txt)
    print("Algorithm 1 predicted emotions:", pred_1[0], pred_1[1], pred_1[2])
    print("Algorithm 2 predicted emotions:", pred_2[0], pred_2[1], pred_2[2])
    print("Algorithm 3 predicted emotions:", pred_3[0], pred_3[1], pred_3[2])

############ MODEL REWEIGHING ALGORITHM ######################################
# Choosing Weights for the Three Algorithms

# Labels to output for user
 
def top_labels(txt):
     # Run all three algorithms
    alg1_res = run_algorithm_1(model_path_1, txt)
    alg2_res = run_algorithm_2_3(
        model_path_2, txt, num_words, input_length)
    alg3_res = run_algorithm_2_3(
        model_path_3, txt, num_words, input_length)
    # Multiply the results of each algorithm by the weights and add everything together
    # Truncate to the top three results
    alg_cum = alg1_w * alg1_res + alg2_w * alg2_res + alg3_w * alg3_res
    top_ordered = sorted(alg_cum, key= lambda i: i[1], reverse=True)
    # Store User's sentence and output as (x,y)
    # Need help here from @Shungo
    top_label = label_cols[top_ordered[0][0]]
    return top_label

def update_weights(emotion):
    # Multiply the results of each algorithm by the weights and add everything together
    # Truncate to the top three results
    global alg1_w
    global alg2_w
    global alg3_w

    alg1_res = run_algorithm_1(model_path_1, txt)
    alg2_res = run_algorithm_2_3(model_path_2, txt, num_words, input_length)
    alg3_res = run_algorithm_2_3(model_path_3, txt, num_words, input_length)
    # Store User's sentence and output as (x,y)
    top_w = 1
    # For the algorithm that had the highest amount of the top emotion, increase weight of the algorithm by its (weight of the top "emotion"
    # - average of the emotion of the three algorithms) multiplied by x and (1-weight)
    # (i.e. if 0.5 of happy came from this algorithm, after first iteration, with total happy of 0.9,
    # increase weight by ((0.5-0.3) * 2/3 * x)
    if emotion in label_cols:
        choice_index = label_cols.index(emotion)
        choice_alg1 = alg1_res[choice_index]
        choice_alg2 = alg2_res[choice_index]
        choice_alg3 = alg3_res[choice_index]
        choice_avg = sum(choice_alg1, choice_alg2, choice_alg3)/3
        # Observed through random assignment, there was eventually convergence to one value despite none technically being best
        # To combat this, set max between 0.05 (obtained through testing on Excel) and calculation
        # Need to set caps for algorithms such that no algorithm will remain dead despite being top choice
        # Through testing, observed that 0.05 was a good limit such that a new algorithm may become top within a week given right conditions
        choice_val1 = max(0.05, (choice_alg1 - choice_avg) * top_w * alg1_w * (1 - alg1_w))
        choice_val2 = max(0.05, (choice_alg2 - choice_avg) * top_w * alg2_w * (1 - alg2_w))
        choice_val3 = max(0.05, (choice_alg3 - choice_avg) * top_w * alg3_w * (1 - alg3_w))
        choice_val_sum = choice_val1 + choice_val2 + choice_val3
        # Performing a hard reset of values is not optimal, and using momentum could lead to massive overshooting
        # Do the same for the two other algorithms
        # Reweigh all algorithms on a scale of 0 to 1 by dividing by sum and multiplying by 3
        alg1_w = choice_val1 / choice_val_sum
        alg2_w = choice_val2 / choice_val_sum
        alg3_w = choice_val3  / choice_val_sum
        choice_val1, choice_val2, choice_val3
    return

    # Function that returns top three emotions
    # Function that reweighs based on chosen emotion