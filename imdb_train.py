"""Trains a POS-tagger on the French GSD dataset."""

from tqdm import tqdm
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *


def train(checkpoint, criterion, train_loader, val_loader, epochs, patience=None,
          writer=None, device=get_device()):
    """Full training loop"""

    print("Training on", 'GPU' if device.type == 'cuda' else 'CPU', '\n')
    net, optimizer = checkpoint.model, checkpoint.optimizer
    if patience is not None:
        early_stopper = EarlyStopper(patience)
    min_loss = float('inf')
    iteration = 1

    def train_epoch():
        """An epoch of training."""
        nonlocal iteration
        epoch_loss = 0.
        pbar = tqdm(train_loader, desc=f'TRAIN Epoch {epoch}/{epochs}', dynamic_ncols=True)  # progress bar
        net.train()
        correct = 0
        for data, lengths, target in pbar:

            data, lengths, target = data.to(device), lengths.to(device), \
                target.to(device)
            # reset gradients
            optimizer.zero_grad()
            output = net(data, lengths)
            # compute loss
            loss = criterion(output, target)
            epoch_loss += loss.item()
            pbar.set_postfix(loss=f'{loss.item():.4e}')
            if writer:
                writer.add_scalar('Iteration-loss', loss.item(), iteration)
            # compute correct predictions
            pred = output.argmax(1)
            correct += pred.eq(target).sum().item()
            # compute gradients, update parameters
            loss.backward()
            optimizer.step()
            iteration += 1

        acc = correct / len(train_loader.dataset)
        epoch_loss /= len(train_loader)
        if writer:
            writer.add_scalar(f"Train-epoch-loss", epoch_loss, epoch)
            writer.add_scalar(f"Train-epoch-accuracy", acc, epoch)
        print(f"TRAIN Epoch {epoch}/{epochs}; Mean Loss: {epoch_loss:.4e}, "
              f"Accuracy: {acc:.4f}")
        return epoch_loss,

    def evaluate_epoch():
        """An evaluation epoch."""
        net.eval()
        correct = 0
        epoch_loss = 0.
        pbar = tqdm(val_loader, desc=f'TRAIN Epoch {epoch}/{epochs}', dynamic_ncols=True)
        with torch.no_grad():
            for data, lengths, target in pbar:
                data, lengths, target = data.to(device), lengths.to(device), \
                    target.to(device)
                output = net(data, lengths)
                # compute loss
                loss = criterion(output, target)
                epoch_loss += loss.item()
                # compute correct predictions
                pred = output.argmax(1)
                correct += pred.eq(target).sum().item()

        epoch_loss /= len(val_loader)
        acc = correct / len(val_loader.dataset)

        print(f"VALIDATION Epoch {epoch}/{epochs}; Mean Loss: {epoch_loss:.4e}, "
              f"Accuracy: {acc:.4f}")
        if writer:
            writer.add_scalar(f"Validation-epoch-loss", epoch_loss, epoch)
            writer.add_scalar(f"Validation-epoch-accuracy", acc, epoch)
        return epoch_loss, acc

    # Begin training
    begin_epoch = checkpoint.epoch
    for epoch in range(begin_epoch, epochs+1):
        train_epoch()
        loss, acc = evaluate_epoch()
        checkpoint.epoch += 1
        if loss < min_loss:
            min_loss = loss
            best_acc = acc
            best_epoch = epoch
            checkpoint.save('_best')
        checkpoint.save()
        if patience is not None:
            early_stopper.add(loss, epoch)
            if early_stopper.stop():
                print(f"No improvement in {patience} epochs, early stopping.")
                break

    print("\nFinished.")
    print(f"Best validation loss: {min_loss:.4e}")
    print(f"With accuracy: {best_acc}")
    print(f"Best epoch: {best_epoch}")
    return best_acc


if __name__ == '__main__':

    def parse_args():
        """Parse command line arguments."""
        parser = argparse.ArgumentParser(
            description="Trains baseline linear model on GloVe embeddings on IMDB "
                        "dataset.")

        parser.add_argument('--no-tensorboard', action='store_true')
        parser.add_argument('--model', default='basemodel', choices=['basemodel'])
        parser.add_argument('--batch-size', default=128, type=int)
        parser.add_argument('--lr', default=0.001, type=float)
        parser.add_argument('--epochs', default=40, type=int)
        parser.add_argument('--patience', default=None, type=int)
        return parser.parse_args()

    device = get_device()
    torch.manual_seed(42)
    args = parse_args()

    import torch.optim as optim
    from torch.utils.tensorboard import SummaryWriter

    from models import BaseModel
    from imdb_data import *

    word2id, embeddings = get_glove_embeddings(embedding_size=50)
    train_loader, val_loader, test_loader = get_dataloaders(args.batch_size, word2id)

    print(args.model)
    if args.model == 'basemodel':
        net = BaseModel(embeddings, num_classes=2)

    net = net.to(device)

    # In order to exclude losses computed on null entries (zero),
    # set ignore_index=0 for the loss criterion
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    hparams = get_hparams(args, ignore_keys={'no_tensorboard'})
    expe_name = get_expe_name(prefix='__IMDB__', hparams=hparams)

    # path where to save the model
    savepath = Path('./checkpoints/checkpt.pt')
    # Tensorboard summary writer
    if args.no_tensorboard:
        writer = None
    else:
        writer = SummaryWriter(comment=expe_name, flush_secs=10)
        # log sample data and net graph in tensorboard
        data, lengths, target = next(iter(train_loader))
        writer.add_graph(net, (data, lengths))

    checkpoint = CheckpointState(net, optimizer, savepath=savepath)

    best_acc = train(checkpoint, criterion, train_loader, val_loader, args.epochs,
          patience=args.patience, writer=writer, device=device)
    # log accuracy for hyperparameters
    if writer:
        writer.add_hparams(hparams, {'accuracy': best_acc})

    writer.close()
