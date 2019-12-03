import re
from pathlib import Path
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from datamaestro import prepare_dataset

def get_glove_embeddings():
    word2id, embeddings = prepare_dataset('edu.standford.glove.6b.50').load()

    # add a null embedding for pad: set it to id zero, so it works when zero-padding
    # In order to set 0 to PAD_ID, we need to shift all the ids for the embeddings
    # by adding one (because the id 0 is already used)
    for word in word2id:
        word2id[word] += 1
    word2id['<pad>'] = 0
    embeddings = np.insert(embeddings, 0, values=0, axis=0)

    # add an OOV embedding: use the mean of all embeddings
    OOV_ID = len(word2id)
    word2id['<oov>'] = OOV_ID
    embeddings = np.insert(embeddings, OOV_ID, embeddings.mean(0), axis=0)

    return word2id, embeddings


def clean(text: str):
    """Normalizes text."""

    # convert to lower case and clean HTML tags
    RE_TAGS = r'<.*?>'
    return re.sub(RE_TAGS, '', text.lower())

def tokenize(text: str):
    """Tokenizes text by seperating on spaces."""
    RE_WORDS = r'\S+'
    return list([x for x in re.findall(RE_WORDS, text.lower())])

def numericalize(tokens: list, word2id: dict):
    """Encodes tokens as a tensor of integers."""

    OOV_ID = word2id['<oov>']
    ids = list()
    for word in tokens:
        try:
            i = word2id[word]
        except KeyError:
            i = OOV_ID
        ids.append(i)
    return torch.LongTensor(ids)


class FolderText(Dataset):
    """Dataset for IMDB."""
    def __init__(self, classes, word2id, load):
        self.files = []
        self.filelabels = []
        self.label2id = dict(zip(classes.keys(), [0, 1]))

        for label, folder in classes.items():
            for file in folder.glob("*.txt"):
                self.files.append(file)
                self.filelabels.append(label)

        self.word2id = word2id

    def __len__(self):
        return len(self.filelabels)

    def __getitem__(self, idx):
        text = self.files[idx].read_text()
        label = self.filelabels[idx]

        # clean text, tokenize and numericalize
        tokenized = tokenize(clean(text))
        # get label id
        return numericalize(tokenized, self.word2id), self.label2id[label]

    def get_raw(self, idx):
        return self.files[idx].read_text(), self.filelabels[idx]

    @staticmethod
    def collate(batch):
        """Collate function (pass to DataLoader's collate_fn arg).
        Args:
            batch (list): list of examples returned by __getitem__
        Returns:
            tuple: Three tensors: batch of padded documents, lengths of documents,
                and target classes.
        """
        data, target = list(zip(*batch))
        lengths = torch.LongTensor([len(s) for s in data])
        # pad sequences with 0
        return (pad_sequence(data, padding_value=0), lengths, torch.LongTensor(target))


def get_dataloaders(batch_size, word2id):

    # load the IMDB dataset
    ds = prepare_dataset("edu.standford.aclimdb")

    # ds.train.classes and ds.test.classes are dict ('class', 'path-to-files')
    dev_ds = FolderText(ds.train.classes, word2id, load=False)
    test_ds = FolderText(ds.test.classes, word2id, load=False)

    # partition development set as train and validation set
    train_len = int(len(dev_ds)*0.9)
    val_len = len(dev_ds) - train_len
    train_ds, val_ds = torch.utils.data.random_split(
        dev_ds, [train_len, val_len])

    kwargs = dict(collate_fn=FolderText.collate,
                  pin_memory=(torch.cuda.is_available()),
                  num_workers=torch.multiprocessing.cpu_count())

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **kwargs)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True, **kwargs)

    return train_loader, val_loader, test_loader


if __name__ == '__main__':

    EMBEDDING_SIZE = 50  # size of GloVe vectors

    # load the IMDB dataset
    print('Loading IMDB dataset..')
    ds = prepare_dataset("edu.standford.aclimdb")
    # load pretrained GloVe word embeddings (400k trained vectors)
    print('Loading GloVe embeddings..')
    word2id, embeddings = get_glove_embeddings()

    print('Vocab from GloVe: size {}, head: {}'.format(
        len(word2id), [(i, w) for w, i in word2id.items()][:40]))
    print('Embeddings matrix size:', type(embeddings), embeddings.shape)


    # ds.train.classes and ds.test.classes are dict ('class', 'path-to-files')
    train_dataset = FolderText(ds.train.classes, word2id, load=False)
    test_dataset = FolderText(ds.test.classes, word2id, load=False)

    print('IMDB dataset:')
    print('Number of training samples:', len(train_dataset))
    print('Number of testing samples:', len(test_dataset), '\n')
    print('Train samples:\n')
    for idx in random.sample(range(len(train_dataset)), 4):
        data, target = train_dataset[idx]
        text = train_dataset.get_raw(idx)
        print('Input:', text)
        print('Target:', target, '\n\n')

    train_loader, val_loader, test_loader = get_dataloaders(16, word2id)
    # Test batch
    print('Train batch:')
    data, lengths, target = next(iter(train_loader))
    print(f"Input {tuple(data.size())}:\n{data}")
    print(f"Lenghts:\n{lengths}")
    print(f"Target {tuple(target.size())}:\n{target}")
