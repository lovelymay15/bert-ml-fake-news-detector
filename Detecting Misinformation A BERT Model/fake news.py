# MODULES
import os
import time
import datetime
import random
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from statistics import mean
from collections import Counter
from langdetect import detect, DetectorFactory
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler, Subset
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup

# CONSTANTS
DF_ORIGINAL = r"WELFake_Dataset.csv"
DF_FINAL = r"final_data.csv"
MODEL_FOLDER = r"/model/"

device = torch.device("cpu")
os.environ['CURL_CA_BUNDLE'] = '' 

# SEED
my_seed = 288933

random.seed(my_seed)
np.random.seed(my_seed)
torch.manual_seed(my_seed)
torch.cuda.manual_seed_all(my_seed)

# IMPORTING DATASET
data = pd.read_csv(DF_ORIGINAL, index_col=0)

# Check amount of fake and real news
print(f'-------------------------------------------------------------------------------------------------')
print(data.groupby('label').size())

# Filtering 'title' and 'label'
data = data[['title', 'label']]

n_row_before = len(data)
data = data[~data['title'].isnull()]
data = data[~data['label'].isnull()]

n_row_after = len(data)
print(f'''------------------------------------------------------------------------------------------------- 
      Number of observations removed for null values in title or rating columns: {n_row_before- n_row_after}''')

n_row_before = len(data)
data.drop_duplicates(inplace=True)
data.drop_duplicates(subset='title', keep=False, inplace=True, ignore_index=True)  # Values removed here are titles that have both labels

n_row_after = len(data)
print(f'''-------------------------------------------------------------------------------------------------
       Number of observations removed for duplicate values: {n_row_before- n_row_after}''')

DetectorFactory.seed = 0

language = []
for title in data['title']:
    try:
        language.append(detect(title))
    except:
        language.append('error')
        pass

print(Counter(language))

# Filtering titles in other language
n_row_before = len(data)
data.drop([idx for idx, l in enumerate(language) if l in ['ru', 'ar', 'lt', 'tr', 'pl', 'bg', 'el', 'zh-cn', 'error']]
        ,axis=0  # rows
        ,inplace=True)
data.reset_index(drop=True, inplace=True)

n_row_after = len(data)
print(f'''-------------------------------------------------------------------------------------------------
      Number of titles removed for not being in English: {n_row_before- n_row_after}''')

# BERT Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Token vector size calculate
data['n_token'] = [len(tokenizer.encode(title, add_special_tokens=True)) for title in data['title']]


print('-------------------------------------------------------------------------------------------------')
print('Minimum tokens: ', min(data['n_token']))
print('Maximum tokens: ', max(data['n_token']))
print('Average of tokens: ', round(mean(data['n_token']), 4))
print('Number of tokens per percentile: ', np.quantile(data['n_token'], q=[.1, .2, .3, .4, .5, .6, .7, .8, .9]))
print('Number of titles with more than 128 tokens: ', sum([i > 128 for i in data['n_token']]))
print('Number of titles with more than 64 tokens: ', sum([i > 64 for i in data['n_token']]))
print('Number of titles with more than 32 tokens: ', sum([i > 32 for i in data['n_token']]))


# Check amount of fake and real news after pre-processing
print(data.groupby('label').size())

# Saving pre-process database to DF_FINAL
data.to_csv(DF_FINAL, index=False)
data = pd.read_csv(DF_FINAL)

# TUNING
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', 
    num_labels = 2, output_attentions = False, output_hidden_states = False)

# Tokenizes all sentences
input_ids = []
attention_masks = []

for sentence in data['title'].values:
    encoded_dict = tokenizer.encode_plus(
                        sentence,
                        add_special_tokens = True,
                        max_length = 64,  
                        truncation=True, 
                        padding='max_length',  
                        return_attention_mask = True,  
                        return_tensors = 'pt'
                   )

    # Save into a list
    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])

# Transform the list of tensors into a tensor with multiple lists
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(data['label'].values)

# Example of result
print('Original sentence: ', data['title'].values[0])
print('Tokenized sentence: ', input_ids[0])
print('Attention: ', attention_masks[0])
print('Label: ', labels[0])

# Creating a TensorDataset object
dataset = TensorDataset(input_ids, attention_masks, labels)

train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

print('{:>5,} sentences for training'.format(train_size))
print('{:>5,} sentences for validation'.format(val_size))
print('{:>5,} sentences for test'.format(test_size))

# Hyperparameters
batch_size = 32
learning_rate = 3e-5
periods = 3

train_dataloader = DataLoader(train_dataset, 
    sampler = RandomSampler(train_dataset), batch_size = batch_size)
validation_dataloader = DataLoader(val_dataset, 
    sampler = SequentialSampler(val_dataset), batch_size = batch_size)
prediction_dataloader = DataLoader(test_dataset, 
    sampler = SequentialSampler(test_dataset), batch_size = batch_size)

optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)

scheduler = get_linear_schedule_with_warmup(optimizer, 
    num_warmup_steps = 0, num_training_steps = len(train_dataloader) * periods)

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))

    return str(datetime.timedelta(seconds=elapsed_rounded))

# Model Evaluation Statistics
training_stats = [] 

initial_time = time.time()

# For each period...
for period in range(0, periods):
    print('\n======== Period {:} / {:} ========'.format(period + 1, periods))
    print('Training...')
    
    t0 = time.time()  
    total_train_accuracy = 0 
    total_train_loss = 0 

    model.train()

    for step, batch in enumerate(train_dataloader):
        if step % 40 == 0 and not step == 0:
            print('Batch {:>5,} of {:>5,}. Time elapsed: {:}.'.format(step, len(train_dataloader), 
                format_time(time.time() - t0)))
            
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        
        model.zero_grad()
        
        result = model(b_input_ids,
                       token_type_ids=None,
                       attention_mask=b_input_mask,
                       labels=b_labels,
                       return_dict=True)
        loss = result.loss
        logits = result.logits
        total_train_loss += loss.item()
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        total_train_accuracy += flat_accuracy(logits, label_ids) 
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) 
        optimizer.step()
        scheduler.step()

    # Calculate average training accuracy
    avg_train_accuracy = total_train_accuracy / len(train_dataloader) 
    print("  Accuracy: {0:.2f}".format(avg_train_accuracy))
    # Calculate average loss
    avg_train_loss = total_train_loss / len(train_dataloader) 
    training_time = format_time(time.time() - t0)

    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training period took: {:}".format(training_time))

    print("\nValidating...")

    t0 = time.time()

    model.eval()  # Validation mode

    total_eval_accuracy = 0
    total_eval_loss = 0

    for batch in validation_dataloader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        with torch.no_grad():
            result = model(b_input_ids,
                           token_type_ids=None,
                           attention_mask=b_input_mask,
                           labels=b_labels,
                           return_dict=True)

        loss = result.loss
        logits = result.logits

        total_eval_loss += loss.item()  

        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        total_eval_accuracy += flat_accuracy(logits, label_ids)

    avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
    print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

    avg_val_loss = total_eval_loss / len(validation_dataloader) 

    validation_time = format_time(time.time() - t0)

    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    print("  Validation took: {:}".format(validation_time))

    # Storing statistics from this period
    training_stats.append(
        {
            'period': period + 1,
            'Training loss': avg_train_loss,
            'Validation loss': avg_val_loss,
            'Training accuracy': avg_train_accuracy,
            'Validation accuracy': avg_val_accuracy,
            'Training time': training_time,
            'Validation time': validation_time
        }
    )

print("\nTraining complete!")
print("Total training time: {:} (hh:mm:ss)".format(format_time(time.time()-initial_time)))

# SAVING TRAINED MODEL AND TOKENIZER
if not os.path.exists(MODEL_FOLDER):
    os.makedirs(MODEL_FOLDER)

model_to_save = model.module if hasattr(model, 'module') else model
model_to_save.save_pretrained(MODEL_FOLDER)
tokenizer.save_pretrained(MODEL_FOLDER)

# Saving model statistics
data_stats = pd.DataFrame(data=training_stats).set_index('period')
data_stats.to_csv(os.path.join(MODEL_FOLDER, 'training_stats.csv'))

data_indexes = pd.DataFrame({'set': ['train' for _ in range(len(train_dataset))] + ['validation' for _ in range(len(val_dataset))] + ['test' for _ in range(len(test_dataset))]
                          ,'indice': train_dataset.indices + val_dataset.indices + test_dataset.indices})
data_indexes.to_csv(os.path.join(MODEL_FOLDER, 'indices_set.csv'), index=False)

# Analyzing training statistics
data_stats = pd.read_csv(os.path.join(MODEL_FOLDER, 'training_stats.csv'), index_col=0)

# Using the style from the seaborn package
sns.set(style='darkgrid')

# Increase the plot size and font size
sns.set(font_scale=1.5)
plt.rcParams["figure.figsize"] = (12,6)

# Plot the learning curve
plt.plot(data_stats['Training Loss'], 'b-o', label="Training")
plt.plot(data_stats['Validation Loss'], 'g-o', label="Validation")

# Label the plot
plt.title("Training and Validation Loss")
plt.xlabel("Period")
plt.ylabel("Loss")
plt.legend()
plt.xticks([1, 2, 3])

plt.show()

tokenizer = BertTokenizer.from_pretrained(MODEL_FOLDER)
model = BertForSequenceClassification.from_pretrained(MODEL_FOLDER, num_labels = 2, output_attentions = False, output_hidden_states = False)

model.to(device)

data_indexes = pd.read_csv(os.path.join(MODEL_FOLDER, 'indices_set.csv'))
test_indexes = data_indexes.loc[data_indexes['set'] == 'test', 'indice'].tolist()

test_dataset = Subset(dataset, test_indexes)
prediction_dataloader = DataLoader(test_dataset, sampler = SequentialSampler(test_dataset), batch_size = batch_size)

model.eval()

logits, true_labels = [], []

for batch in prediction_dataloader:
    batch = tuple(t.to(device) for t in batch)
    b_input_ids, b_input_mask, b_labels = batch

    with torch.no_grad():
        result = model(b_input_ids,
                        token_type_ids=None,
                        attention_mask=b_input_mask,
                        return_dict=True)

    logits_aux = result.logits

    logits_aux = logits_aux.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()
    logits.append(logits_aux)
    true_labels.append(label_ids)



# Transforming the list of arrays to arrays
logits = np.concatenate(logits, axis=0)
true_labels = np.concatenate(true_labels, axis=0)

# Creating folders to store the logits and true_labels arrays
EVAL_FOLDER = os.path.join(MODEL_FOLDER, 'evaluation')

if not os.path.exists(EVAL_FOLDER): os.makedirs(EVAL_FOLDER)

# Saving all arrays
np.save(os.path.join(EVAL_FOLDER, f'logits.npy'), logits)
np.save(os.path.join(EVAL_FOLDER, f'true_labels.npy'), true_labels)

logits = np.load(os.path.join(EVAL_FOLDER, f'logits.npy'))
true_labels = np.load(os.path.join(EVAL_FOLDER, f'true_labels.npy'))

predictions = np.argmax(logits, axis=1).flatten()

print(data.loc[test_indexes].groupby('label').size())

tp = sum(predictions[true_labels == 1] == 1)  # true positive
tn = sum(predictions[true_labels == 0] == 0)  # true negative
fp = sum(predictions[true_labels == 0] == 1)  # fake positive
fn = sum(predictions[true_labels == 1] == 0)  # fake negative

tp_perc = '{0:.4%}'.format(tp/(tp + tn + fp + fn))
tn_perc = '{0:.4%}'.format(tn/(tp + tn + fp + fn))
fp_perc = '{0:.4%}'.format(fp/(tp + tn + fp + fn))
fn_perc = '{0:.4%}'.format(fn/(tp + tn + fp + fn))

print(pd.DataFrame({'Positive predicted': [tp, fp], 'Negative predicted': [fn, tn]}, index=['True positive', 'True negative']))

accuracy = (tp + tn)/(tp + tn + fp + fn)
precision = tp/(tp + fp)
sensitivity = tp/(tp + fn)
specificity = tn/(tn + fp)
f1_score = 2*(precision * sensitivity)/(precision + sensitivity)

print(f"{'Accuracy in the test set: ':<35}{accuracy:>.4%}")
print(f"{'Precision in the test set: ':<35}{precision:>.4%}")
print(f"{'F1 score in the test set: ':<35}{f1_score:>.4%}")
print(f"{'Sensitivity in the test set: ':<35}{sensitivity:>.4%}")
print(f"{'Specificity in the test set: ':<35}{specificity:>.4%}")

wrong_classification = []
for a, b, c in zip(predictions.tolist(), true_labels.tolist(), test_indexes):
    if a != b:
        wrong_classification.append(c)

print(data.loc[wrong_classification].sample(10))
