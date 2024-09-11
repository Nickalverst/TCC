import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, EarlyStoppingCallback, TrainingArguments
import torch
from torch.utils.data import Dataset, DataLoader
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import csv

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    acc = accuracy_score(labels, predictions)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}


def preprocess_text(text):
    # Tokenize text and remove stopwords
    tokens = [word for word in text.split() if word.lower() not in stop_words]
    # Lemmatize each word
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    # Rejoin tokens into a cleaned sentence
    return ' '.join(tokens)


# Download necessary resources
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Load the CSV file
df = pd.read_csv('reviews_for_training_1-100.csv') # 19.658
df2 = pd.read_csv('reviews_for_training_100-200.csv') # 68.832 / Total: 88.490
df3 = pd.read_csv('reviews_for_training_200-300.csv') # 31.623 / Total: 120.113
df4 = pd.read_csv('reviews_for_training_300-500.csv') # 148.935 / Total: 269.044
df5 = pd.read_csv('reviews_1-500_upto100k.csv') # 943.683 / Total: 1.212.727

df = pd.concat([df, df2, df3, df4, df5]).reset_index(drop=True)

# Apply preprocessing to the reviews column
df['review'] = df['review'].apply(preprocess_text)

# Count the number of negative reviews
negative_reviews_count = len(df[df['recommend'] == 0])
print(negative_reviews_count)

# Sample the same number of positive reviews as there are negative reviews
positive_reviews_sample = df[df['recommend'] == 1].sample(negative_reviews_count, random_state=42)

# Get all the negative reviews
negative_reviews = df[df['recommend'] == 0]

# Concatenate the positive and negative reviews to create a balanced dataset
balanced_df = pd.concat([positive_reviews_sample, negative_reviews])

# Shuffle the dataframe to mix the positive and negative reviews
df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Check the class distribution
print(df['recommend'].value_counts())

# Randomly sample 269,044 reviews from the balanced dataframe
df = df.sample(n=269044, random_state=42)
df = df.reset_index(drop=True)

# Check the distribution again to ensure it's still balanced
print(df['recommend'].value_counts())

df.to_csv('preprocessed_input.csv', index=False)

# Ensure the columns are correctly loaded
print(df.head())

# Ensure I've loaded the correct files
print(df.shape[0])

train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForSequenceClassification.from_pretrained('roberta-base')

train_encodings = tokenizer(train_data['review'].tolist(), truncation=True, padding=True)
test_encodings = tokenizer(test_data['review'].tolist(), truncation=True, padding=True)


class ReviewsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = ReviewsDataset(train_encodings, train_data['recommend'].tolist())
test_dataset = ReviewsDataset(test_encodings, test_data['recommend'].tolist())

# Create DataLoader with multiple workers (half the # of cores in my CPU)
train_loader = DataLoader(train_dataset, shuffle=True, num_workers=6)
test_loader = DataLoader(test_dataset, shuffle=False, num_workers=6)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    fp16=True,  # Enabled mixed precision training since model_3
    gradient_accumulation_steps=2,  # Simulates a batch size of 16 since model_3
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=50,
    eval_strategy="steps",  # Ensure evaluation happens during training
    save_total_limit=3,  # Keep only the 3 most recent checkpoints
    eval_steps=1000,
    save_steps=1000,
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]  # Stop if no improvement after 3 eval steps
)

trainer.train()

results = trainer.evaluate()
print(results)

# Append the new metrics to your CSV file
with open('model_comparison.csv', mode='a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([
        'model_9',
        results['eval_loss'],
        results['eval_runtime'],
        results['eval_samples_per_second'],
        results['eval_steps_per_second'],
        results['eval_accuracy'],
        results['eval_f1'],
        results['eval_precision'],
        results['eval_recall'],
        results['epoch']
    ])
