import random

import numpy as np
import pandas as pd
import sklearn
import torch
from simpletransformers.classification import ClassificationModel
from sklearn.model_selection import train_test_split

# noinspection PyUnresolvedReferences
import monkey_patches
from prepare_data import PREFIX

seed = 123456

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

train_features = pd.read_csv(
    PREFIX / 'data/email_best_send_time_train_features.csv',
    keep_default_na=False,
    index_col=0
)[['text', 'labels']]
train_df, eval_df = train_test_split(train_features, shuffle=True, test_size=0.2)
test_df = pd.read_csv(
    PREFIX / 'data/email_best_send_time_test_features.csv',
    keep_default_na=False,
    index_col=0,
)

batch_size = 40
model = ClassificationModel(
    'bert', str(PREFIX / 'outputs/best_model_o-3'),
    use_cuda=True,
    args={
        'num_train_epochs': 20,

        'overwrite_output_dir': True,
        'output_dir': str(PREFIX / 'outputs'),
        'best_model_dir': str(PREFIX / 'outputs/best_model'),

        'save_eval_checkpoints': False,
        'evaluate_during_training_verbose': True,
        'evaluate_during_training_silent': False,

        # 'early_stopping_patience': 3,
        # 'early_stopping_delta': 0,
        # 'early_stopping_metric': 'eval_loss',
        # 'early_stopping_metric_minimize': True,
        # 'use_early_stopping': True,

        'save_steps': -1,
        'save_model_every_epoch': True,

        # 'evaluate_during_training': True,
        # 'evaluate_during_training_steps': len(train_df) // (batch_size * 4),

        'evaluate_each_epoch': True,
        'evaluate_during_training': True,
        'evaluate_during_training_steps': -1,

        'train_batch_size': batch_size,
        'eval_batch_size': batch_size,

        'learning_rate': 3e-5,
        'use_multiprocessing': False,
        'use_multiprocessing_for_evaluation': False,
    },
)


model.train_model(
    train_df=train_df, eval_df=eval_df,
    f1=lambda a, b: sklearn.metrics.f1_score(a, b)
)


# preds = model.predict(test_df['text'].to_list())[0]
# submission = pd.Series(preds, name='Opened', index=test_df.index)
# submission.to_csv(PREFIX / 'data/email_best_send_time_submission.csv')
