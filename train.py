'''
This script handling the training process.
'''

import argparse
import math
import time

import numpy as np

from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import transformer.Constants as Constants
from dataset import TranslationDataset, paired_collate_fn
from transformer.Models import Transformer
from transformer.Optim import ScheduledOptim
from FocalLoss import *

import matplotlib.pyplot as plt

from torch.utils.data.sampler import SubsetRandomSampler

def plot(train_loss, val_loss):
    epoch_count = range(1, len(train_loss)+1)
    plt.plot(epoch_count, train_loss, 'r-')
    plt.plot(epoch_count, val_loss, 'b-')
    plt.legend(['Training Loss', 'Validation Loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    return plt

# calculating accuracy 
def get_acc(gt, pred):
    assert len(gt) == len(pred)
    correct = 0
    for i in range(len(gt)):
        if gt[i] == pred[i]:
            correct += 1
                
    return (1.0 * correct)/len(gt)

def cal_performance(pred, gold, smoothing=False):
    ''' Apply label smoothing if needed '''
    loss = cal_loss(pred, gold, smoothing)
    pred = pred.max(1)[1]

    gold = gold.contiguous().view(-1)

    non_pad_mask = gold.ne(Constants.PAD)
    n_correct = pred.eq(gold)
    n_correct = n_correct.masked_select(non_pad_mask).sum().item()

    test1 = pred.masked_select(pred.ne(Constants.PAD)).tolist()
    test2 = gold.masked_select(non_pad_mask).tolist()

    # TODO: Fixing here
    list_of_lists1 = []
    acc = []
    for i in test1:
        acc.append(i)
        if (i == Constants.EOS):
            # print(acc)
            list_of_lists1.append(acc)
            acc = []

    list_of_lists2 = []
    acc = []
    for i in test2:
        acc.append(i)
        if (i == Constants.EOS):
            # print(acc)
            list_of_lists2.append(acc)
            acc = []

    print(len(list_of_lists1))
    print(len(list_of_lists2))
    
    accuracies = []
    for test1, test2 in zip(list_of_lists1, list_of_lists2):
        accuracies.append(get_acc(test1, test2))
    print(accuracies)
    return loss, n_correct, np.mean(accuracies)


def cal_loss(pred, gold, smoothing):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)
    # return FocalLoss(gamma=2, alpha=4)(pred, gold)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(Constants.PAD)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
        loss = F.cross_entropy(
            pred, gold, ignore_index=Constants.PAD, reduction='sum')

    return loss


def train_epoch(model, training_data, optimizer, device, smoothing):
    ''' Epoch operation in training phase'''

    model.train()

    total_loss = 0
    n_word_batch_mean = 0
    n_batch = 0

    accu = []
    for batch in tqdm(
            training_data, mininterval=2,
            desc='  - (Training)   ', leave=False):

        # prepare data

        src_seq, src_sp, src_pos, tgt_seq, tgt_pos = map(
            lambda x: x.to(device), batch)
        gold = tgt_seq[:, 1:]

        # forward
        optimizer.zero_grad()
        pred = model(src_seq, src_sp, src_pos, tgt_seq, tgt_pos)

        # backward
        loss, n_correct, accuracy2 = cal_performance(pred, gold, smoothing=smoothing)
        loss.backward()

        accu.append(accuracy2)
        # update parameters
        optimizer.step_and_update_lr()

        
        n_batch += 1
        non_pad_mask = gold.ne(Constants.PAD)
        n_word = non_pad_mask.sum().item()
        total_loss += loss.item()/n_word
        n_word_batch_mean += n_correct/n_word

    # loss_per_word = total_loss/n_word_total
    mean_loss = total_loss/n_batch
    accuracy = n_word_batch_mean/n_batch

    return mean_loss, accuracy, np.mean(accu)


def eval_epoch(model, validation_data, device):
    ''' Epoch operation in evaluation phase '''

    model.eval()

    total_loss = 0
    n_word_batch_mean = 0
    n_batch = 0

    accu = []
    with torch.no_grad():
        for batch in tqdm(
                validation_data, mininterval=2,
                desc='  - (Validation) ', leave=False):

            # prepare data
            src_seq, src_sp, src_pos, tgt_seq, tgt_pos = map(
                lambda x: x.to(device), batch)
            gold = tgt_seq[:, 1:]

            # forward
            pred = model(src_seq, src_sp, src_pos, tgt_seq, tgt_pos)
            loss, n_correct, accuracy2 = cal_performance(pred, gold, smoothing=False)


            n_batch += 1
            # note keeping

            non_pad_mask = gold.ne(Constants.PAD)
            n_word = non_pad_mask.sum().item()
            total_loss += loss.item()/n_word
            n_word_batch_mean += n_correct/n_word
            accu.append(accuracy2)
            

    mean_loss = total_loss/n_batch
    accuracy = n_word_batch_mean/n_batch
    return mean_loss, accuracy, np.mean(accu)


def test(model, test_data, device, opt):
    start = time.time()
    valid_loss, valid_accu, new_accu = eval_epoch(model, test_data, device)
    print('  - (Validation) ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, right_accuracy: {accu2:3.3f} % '
          'elapsed: {elapse:3.3f} min'.format(
              ppl=math.exp(min(valid_loss, 100)), accu=100*valid_accu, accu2=100*new_accu,
              elapse=(time.time()-start)/60))


def train(model, training_data, validation_data, optimizer, device, opt):
    ''' Start training '''

    log_train_file = None
    log_valid_file = None

    if opt.log:
        log_train_file = opt.log + '.train.log'
        log_valid_file = opt.log + '.valid.log'

        print('[Info] Training performance will be written to file: {} and {}'.format(
            log_train_file, log_valid_file))

        with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
            log_tf.write('epoch,loss,ppl,accuracy\n')
            log_vf.write('epoch,loss,ppl,accuracy\n')

    train_loss_all = []
    val_loss_all = []
    valid_accus = []
    for epoch_i in range(opt.epoch):
        print('[ Epoch', epoch_i, ']')

        start = time.time()
        train_loss, train_accu, train_accuracy2 = train_epoch(
            model, training_data, optimizer, device, smoothing=opt.label_smoothing)
        print('  - (Training)   loss: {ppl: 8.5f}, accuracy: {accu:3.3f} %, accuracy_right: {accu2:3.3f} % '
              'elapsed: {elapse:3.3f} min'.format(
                  ppl=train_loss, accu=100*train_accu, accu2=100*train_accuracy2,
                  elapse=(time.time()-start)/60))

        start = time.time()
        valid_loss, valid_accu, val_accuracy2 = eval_epoch(model, validation_data, device)
        print('  - (Validation) loss: {ppl: 8.5f}, accuracy: {accu:3.3f} %, accuracy_right: {accu2:3.3f} % '
              'elapsed: {elapse:3.3f} min'.format(
                  ppl=valid_loss, accu=100*valid_accu, accu2=100*val_accuracy2,
                  elapse=(time.time()-start)/60))

        valid_accus += [val_accuracy2]

        model_state_dict = model.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'settings': opt,
            'epoch': epoch_i}

        if opt.save_model:
            if opt.save_mode == 'all':
                model_name = opt.save_model + \
                    '_accu_{accu:3.3f}.chkpt'.format(accu=100*valid_accu)
                torch.save(checkpoint, model_name)
            elif opt.save_mode == 'best':
                model_name = opt.save_model + '.chkpt'
                if val_accuracy2 >= max(valid_accus):
                    torch.save(checkpoint, model_name)
                    print('    - [Info] The checkpoint file has been updated.')

        if log_train_file and log_valid_file:
            with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
                log_tf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                    epoch=epoch_i, loss=train_loss,
                    ppl=math.exp(min(train_loss, 100)), accu=100*train_accu))
                log_vf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                    epoch=epoch_i, loss=valid_loss,
                    ppl=math.exp(min(valid_loss, 100)), accu=100*valid_accu))
        train_loss_all.append(train_loss)
        val_loss_all.append(valid_loss)

    return train_loss_all, val_loss_all


def main():
    ''' Main function '''
    parser = argparse.ArgumentParser()

    parser.add_argument('-data', default="./pssp-data/data.pt")

    parser.add_argument('-epoch', type=int, default=10)
    parser.add_argument('-batch_size', type=int, default=17)

    #parser.add_argument('-d_word_vec', type=int, default=512)
    parser.add_argument('-d_model', type=int, default=256)
    parser.add_argument('-d_inner_hid', type=int, default=512)
    parser.add_argument('-d_k', type=int, default=64)
    parser.add_argument('-d_v', type=int, default=64)

    parser.add_argument('-n_head', type=int, default=8)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-n_warmup_steps', type=int, default=4000)

    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-embs_share_weight', action='store_true')
    parser.add_argument('-proj_share_weight', action='store_true')

    parser.add_argument('-log', default=None)
    parser.add_argument('-save_model', default=True)
    parser.add_argument('-save_mode', type=str,
                        choices=['all', 'best'], default='best')

    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-label_smoothing', action='store_true')

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda
    opt.d_word_vec = opt.d_model

    #========= Loading Dataset =========#
    data = torch.load(opt.data)
    opt.max_token_seq_len = data['settings'].max_token_seq_len

    training_data, validation_data, test_data = prepare_dataloaders(data, opt)

    opt.src_vocab_size = training_data.dataset.src_vocab_size
    opt.tgt_vocab_size = training_data.dataset.tgt_vocab_size

    #========= Preparing Model =========#
    if opt.embs_share_weight:
        assert training_data.dataset.src_word2idx == training_data.dataset.tgt_word2idx, \
            'The src/tgt word2idx table are different but asked to share word embedding.'

    print(opt)

    device = torch.device('cuda' if opt.cuda else 'cpu')
    transformer = Transformer(
        opt.src_vocab_size,
        opt.tgt_vocab_size,
        opt.max_token_seq_len,
        tgt_emb_prj_weight_sharing=opt.proj_share_weight,
        emb_src_tgt_weight_sharing=opt.embs_share_weight,
        d_k=opt.d_k,
        d_v=opt.d_v,
        d_model=opt.d_model,
        d_word_vec=opt.d_word_vec,
        d_inner=opt.d_inner_hid,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        dropout=opt.dropout).to(device)

    optimizer = ScheduledOptim(
        optim.Adam(
            filter(lambda x: x.requires_grad, transformer.parameters()),
            betas=(0.9, 0.98), eps=1e-09),
        opt.d_model, opt.n_warmup_steps)

    train_loss, val_loss = train(
        transformer, training_data, validation_data, optimizer, device, opt)
    print("Starting Test...")
    test(transformer, test_data, device, opt)
    print("Making loss graph...")
    plt = plot(train_loss, val_loss)
    plt.savefig('loss.png')
    print("Finished!")


def prepare_dataloaders(data, opt):

    validation_split = 0.1
    shuffle_dataset = True
    random_seed = 42

    initDataset = TranslationDataset(
        src_word2idx=data['dict']['src'],
        tgt_word2idx=data['dict']['tgt'],
        src_insts=data['train']['src'],
        tgt_insts=data['train']['tgt'],
        sp_insts=data['train']['sp'])

    # Creating data indices for training and validation splits:
    dataset_size = len(initDataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)


# ========= Preparing DataLoader =========#
    train_loader = torch.utils.data.DataLoader(
        initDataset,
        num_workers=4,
        batch_size=opt.batch_size,
        collate_fn=paired_collate_fn,
        sampler=train_sampler)

    valid_loader = torch.utils.data.DataLoader(
        initDataset,
        num_workers=4,
        batch_size=opt.batch_size,
        collate_fn=paired_collate_fn,
        sampler=valid_sampler)

    test_loader = torch.utils.data.DataLoader(
        TranslationDataset(
            src_word2idx=data['dict']['src'],
            tgt_word2idx=data['dict']['tgt'],
            src_insts=data['valid']['src'],
            tgt_insts=data['valid']['tgt'],
            sp_insts=data['valid']['sp']
        ),
        num_workers=4,
        batch_size=opt.batch_size,
        collate_fn=paired_collate_fn)

    return train_loader, valid_loader, test_loader


if __name__ == '__main__':
    main()
