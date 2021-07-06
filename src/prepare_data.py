from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd
from simpletransformers.classification import ClassificationModel

# PREFIX = Path('/content/drive/MyDrive/Colab Notebooks/CSC21')
PREFIX = Path('.')


def process(df: pd.DataFrame, stats, test=False):
    df = df.copy()

    inverted_tz = df.TimeZone.map(
        lambda t:
        (('+' if t[5] == '-' else '-') + t[6:11])
        if isinstance(t, str)
        else '+00:00'
    )

    symbols = sorted(list(set(''.join(df.Subject))))
    to_delete = [s for s, e in zip(symbols, map(tokenizer.tokenize, symbols)) if e and s != e[0]]
    delete_replaces = zip(to_delete, [''] * len(to_delete))
    df['Subject'] = df.Subject.map(
        lambda s: ' '.join(
            reduce(lambda si, r: si.replace(*r), delete_replaces, s[1:-1]).split()
        )
    )

    # TODO: pd.to_datetime(df.SentOn + inverted_tz) -- Pandas 1.3.0 bug in pandas.core.tools.datetimes._maybe_cache
    # https://github.com/pandas-dev/pandas/pull/42261
    df['ReceiveOn'] = (df.SentOn + inverted_tz).map(lambda s: pd.to_datetime(s, utc=True))
    total_minutes = 60 * df.ReceiveOn.dt.hour + df.ReceiveOn.dt.minute
    total_minutes_norm = total_minutes / (24 * 60)

    df['ReceiveOnSin'] = np.sin(2 * np.pi * total_minutes_norm)
    df['ReceiveOnCos'] = np.cos(2 * np.pi * total_minutes_norm)
    df['ReceiveDay'] = df.ReceiveOn.dt.dayofweek  # 0-6
    df['ReceiveSeason'] = df.ReceiveOn.dt.month % 12 // 3  # 0-3

    df = df.rename(columns={'Subject': 'text'})
    if not test:
        df = df.rename(columns={'Opened': 'labels'})

    onehot_cols = ['ReceiveDay', 'ReceiveSeason']
    dense_cols = ['ReceiveOnSin', 'ReceiveOnCos']

    for on, value in stats.items():
        df = df.merge(value, on=on, how='left')
        df[value.name].fillna(0, inplace=True)  # test use train
        dense_cols.append(value.name)

    df = df[
        ['MailID', 'text'] +
        (['labels'] if not test else []) +
        [*onehot_cols, *dense_cols]
    ]

    for col in onehot_cols:
        df = df.join(pd.get_dummies(df[col]).add_prefix(col))
    df = df.drop(columns=onehot_cols)

    df[dense_cols] -= df[dense_cols].mean()
    df[dense_cols] /= df[dense_cols].std()

    return df


if __name__ == '__main__':
    tokenizer = ClassificationModel('bert', 'bert-base-cased').tokenizer
    train = pd.read_csv(PREFIX / 'data/email_best_send_time_train.csv')
    test = pd.read_csv(PREFIX / 'data/email_best_send_time_test.csv')

    stats = {
        'MailBoxID': train.groupby('MailBoxID')['Opened'].mean().rename('SenderRate'),
        'ContactID': train.groupby('ContactID')['Opened'].mean().rename('RecipientRate'),
    }

    process(train, stats).to_csv(PREFIX / 'data/email_best_send_time_train_features.csv', index=False)
    process(test, stats, test=True).to_csv(PREFIX / 'data/email_best_send_time_test_features.csv', index=False)
