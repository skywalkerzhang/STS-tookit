import stst
from transformers import DebertaV2ForSequenceClassification, AutoModelForSequenceClassification, AutoConfig, set_seed, AutoTokenizer, Trainer, TrainingArguments, default_data_collator, EvalPrediction
import numpy as np
from datasets import load_dataset, load_metric
import torch
from torch import nn
from sklearn.datasets import load_svmlight_file
import pandas as pd

# Define Model
gb = stst.Classifier(stst.GradientBoostingRegression())
model = stst.Model('S1-gb', gb)

# Add features to the Model
model.add(stst.WeightednGramOverlapFeature(type='lemma'))
model.add(stst.BOWFeature(stopwords=False))
model.add(stst.AlignmentFeature())
model.add(stst.IdfAlignmentFeature())
model.add(stst.NegativeFeature())

# train and test
train_file = './data/stsbenchmark/sts-train.csv'
dev_file  = './data/stsbenchmark/sts-dev.csv'
test_file = './data/stsbenchmark/sts-test.csv'

# init the server and input the address
parser = stst.StanfordNLP('http://localhost:9000')

# parse data
train_instances = stst.load_parse_data(train_file, parser)
dev_instances = stst.load_parse_data(dev_file, parser)
test_instances= stst.load_parse_data(test_file, parser)

# train and test
model.make_feature_file(train_instances, train_file, 'train')
model.make_feature_file(dev_instances, dev_file, 'dev')
model.make_feature_file(test_instances, test_file, 'test')


class BertEnsembleForNextSentencePrediction(DebertaV2ForSequenceClassification):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config)

        self.deberta_model = AutoModelForSequenceClassification.from_pretrained(
            'microsoft/deberta-v3-large',
            from_tf=False,
            config=config,
            cache_dir='./cache',
            revision='main',
            use_auth_token=None,
        )

        self.mlp = nn.Sequential(
            nn.Linear(16, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )
        # combine the 2 models into 1
        self.cls = nn.Linear(769, 1)
        # self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            labels=None,
            traditional_features=None
    ):

        deberta_res = self.deberta_model(input_ids,
                                          attention_mask=attention_mask)

        mlp_res = self.mlp(traditional_features)

        # just get the [CLS] embeddings
        logits = self.cls(torch.cat([deberta_res.logits, mlp_res], dim=1))

        if labels is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct(logits.squeeze(), labels.squeeze())
            return loss, logits
        else:
            return logits


task_name = 'stsb'
model_name_or_path = 'microsoft/deberta-v3-large'
cache_dir = './cache'
output_dir = './output'
model_revision = 'main'
use_fast_tokenizer = True
seed = 1
set_seed(seed)
data = load_dataset('glue', task_name)
num_labels = 768

config = AutoConfig.from_pretrained(
    model_name_or_path,
    num_labels=num_labels,
    finetuning_task=task_name,
    cache_dir=cache_dir,
    revision=model_revision,
    use_auth_token=None,
)

model = BertEnsembleForNextSentencePrediction(config)

tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path,
    cache_dir=cache_dir,
    use_fast=use_fast_tokenizer,
    revision=model_revision,
    use_auth_token=None,
)

padding = "max_length"
max_seq_length = 256

# read features
X_train, _ = load_svmlight_file("./generate/models/S1-gb.feature.train.txt")
X_dev, _ = load_svmlight_file("./generate/models/S1-gb.feature.dev.txt")
X_test, _ = load_svmlight_file("./generate/models/S1-gb.feature.test.txt")

X_train = pd.DataFrame.sparse.from_spmatrix(X_train).drop(16, axis=1).values.tolist()
X_dev = pd.DataFrame.sparse.from_spmatrix(X_dev).drop(16, axis=1).values.tolist()
X_test = pd.DataFrame.sparse.from_spmatrix(X_test).drop(16, axis=1).values.tolist()


def preprocess_function(examples):
    # Tokenize the texts
    return tokenizer(examples['sentence1'], examples['sentence2'], padding=padding,
                     max_length=max_seq_length, truncation=True)


data_collator = default_data_collator
data = data.map(
    preprocess_function,
    batched=True,
    load_from_cache_file=True,
    desc="Running tokenizer on dataset",
)
train_dataset = data['train']
train_dataset = train_dataset.map(lambda x: {"traditional_features": X_train[x['idx']]})
eval_dataset = data['validation']
eval_dataset = eval_dataset.map(lambda x: {"traditional_features": X_dev[x['idx']]})
predict_dataset = data['test']
predict_dataset = predict_dataset.map(lambda x: {"traditional_features": X_test[x['idx']]})
metric = load_metric("glue", task_name)


def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.squeeze(preds)
    result = metric.compute(predictions=preds, references=p.label_ids)
    if len(result) > 1:
        result["combined_score"] = np.mean(list(result.values())).item()
    return result


training_args = TrainingArguments(
    output_dir="./debertl3",
    learning_rate=5.2e-6,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=5,
    weight_decay=0.01,
    save_steps=14000,
    seed=1,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    data_collator=data_collator,
)
trainer.train()
trainer.save_model()
predictions = trainer.predict(predict_dataset, metric_key_prefix="predict").predictions
predictions = np.squeeze(predictions)

df = pd.Series(predictions)
df.to_csv("preds.csv", index=False, header=False)


