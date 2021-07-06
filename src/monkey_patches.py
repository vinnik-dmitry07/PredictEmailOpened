# A monkey patches adding dense features
import simpletransformers.classification.classification_utils as classification_utils
import simpletransformers.classification.transformer_models.bert_model as bert_model
import torch

from prepare_data import PREFIX

DENSE_SIZE = 15


def ClassificationDataset_init(self, data, tokenizer, args, mode, multi_label, output_mode, no_cache):
    import pandas as pd
    from simpletransformers.classification.classification_utils import build_classification_dataset

    self.examples, self.labels = build_classification_dataset(
        data, tokenizer, args, mode, multi_label, output_mode, no_cache
    )

    # new
    if mode == 'train':
        self.examples['dense'] = torch.tensor(
            pd.read_csv(
                PREFIX / 'data/email_best_send_time_train_features.csv',
                index_col=0,
            ).drop(columns=['text', 'labels']).values,
            dtype=torch.half,
        )
    else:
        self.examples['dense'] = torch.tensor(
            pd.read_csv(
                PREFIX / 'data/email_best_send_time_test_features.csv',
                index_col=0,
            ).drop(columns='text').values,
            dtype=torch.half,
        )


def BertForSequenceClassification_init(self, config, weight=None):
    import torch.nn as nn
    from transformers.models.bert.modeling_bert import BertModel

    super(bert_model.BertForSequenceClassification, self).__init__(config)  # new
    self.num_labels = config.num_labels

    self.bert = BertModel(config)
    self.dropout = nn.Dropout(config.hidden_dropout_prob)
    self.classifier = nn.Linear(config.hidden_size + DENSE_SIZE, self.config.num_labels)  # new
    self.weight = weight

    self.init_weights()


def BertForSequenceClassification_forward(
        self,
        dense=None,  # new
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
):
    from torch.nn import MSELoss, CrossEntropyLoss

    outputs = self.bert(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
    )
    # Complains if input_embeds is kept

    pooled_output = outputs[1]
    pooled_output = self.dropout(pooled_output)
    pooled_output = torch.cat([pooled_output, dense], dim=-1)  # new

    logits = self.classifier(pooled_output)

    outputs = (logits,) + outputs[
        2:
    ]  # add hidden states and attention if they are here

    if labels is not None:
        if self.num_labels == 1:
            #  We are doing regression
            loss_fct = MSELoss()
            loss = loss_fct(logits.view(-1), labels.view(-1))
        else:
            if self.weight is not None:
                weight = self.weight.to(labels.device)
            else:
                weight = None
            loss_fct = CrossEntropyLoss(weight=weight)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        outputs = (loss,) + outputs

    return outputs  # (loss), logits, (hidden_states), (attentions)


classification_utils.ClassificationDataset.__init__ = ClassificationDataset_init
bert_model.BertForSequenceClassification.__init__ = BertForSequenceClassification_init
bert_model.BertForSequenceClassification.forward = BertForSequenceClassification_forward
