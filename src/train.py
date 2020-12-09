import sys
import os.path
import json
import argparse
from random import seed
from time import time
import math

from scipy.cluster.vq import kmeans

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import torch.utils.tensorboard as tb

from qt import QuantizedTransformerModel
from utils.data import *
from utils.training import *


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description='Trains the QT model.\nFor usage example, refer to: \n' + \
                        '\thttps://github.com/stangelid/qt')

    data_arg_group = argparser.add_argument_group('Data arguments')
    data_arg_group.add_argument('--data', 
            help='training data in json format',
            type=str, default='../data/json/space_train.json')
    data_arg_group.add_argument('--sentencepiece',
            help='sentencepiece model file',
            type=str, default='../data/sentencepiece/spm_unigram_32k.model')
    data_arg_group.add_argument('--max_num_entities',
            help='maximum number of entities to load for training (default: all)',
            type=int, default=None)
    data_arg_group.add_argument('--max_rev_len',
            help='maximum number of sentences per review (default: 40)',
            type=int, default=40)
    data_arg_group.add_argument('--max_sen_len',
            help='maximum number of tokens per sentence (default: 40)',
            type=int, default=40)

    model_arg_group = argparser.add_argument_group('Model hyperparams')
    model_arg_group.add_argument('--d_model',
            help='model dimensionality (default: 320)',
            type=int, default=320)
    model_arg_group.add_argument('--codebook_size',
            help='size of quantization codebook (default: 1024)',
            type=int, default=1024)
    model_arg_group.add_argument('--output_nheads',
            help='number of output sentence heads (default: 8)',
            type=int, default=8)
    model_arg_group.add_argument('--nlayers',
            help='number of sentence-level layers (default: 3)',
            type=int, default=3)
    model_arg_group.add_argument('--internal_nheads',
            help='number of attention heads (default: 4)',
            type=int, default=4)
    model_arg_group.add_argument('--d_ff',
            help='feed-forward dimensionality (default: 512)',
            type=int, default=512)
    model_arg_group.add_argument('--in_pos',
            help='use input positional embeddings',
            action='store_true')
    model_arg_group.add_argument('--out_pos',
            help='use output positional embeddings',
            action='store_true')
    model_arg_group.add_argument('--dropout',
            help='transformer dropout probability (default: 0.0)',
            type=float, default=0.0)

    train_arg_group = argparser.add_argument_group('Basic training hyperparams')
    train_arg_group.add_argument('--batch_size',
            help='the batch size (default: 5)',
            type=int, default=5)
    train_arg_group.add_argument('--epochs',
            help='number of epochs (default: 20)',
            type=int, default=20)
    train_arg_group.add_argument('--lr',
            help='initial learning rate',
            type=float, default=0.001)
    train_arg_group.add_argument('--lr_decay',
            help='learning rate decay (default: 0.9)',
            type=float, default=0.9)
    train_arg_group.add_argument('--label_smoothing',
            help='label smoothing coeff (default: 0.1)',
            type=float, default=0.1)
    train_arg_group.add_argument('--commitment_cost',
            help='VQ-VAE commitment coefficient (default: 1.00)',
            type=float, default=1.00)


    train_arg_group = argparser.add_argument_group('Soft EMA hyperparams')
    train_arg_group.add_argument('--ema_temp',
            help='sampling temperature for Soft EMA codebook training (default: 1.0)',
            type=float, default=1.0)
    train_arg_group.add_argument('--ema_num_samples',
            help='number of samples for Soft EMA codebook training (default: 10)',
            type=int, default=10)
    train_arg_group.add_argument('--ema_decay',
            help='exponential decay for EMA (default: 0.99)',
            type=float, default=0.99)


    lr_arg_group = argparser.add_argument_group('Learning rate drop-off hyperparams',
            'Learning rate drop-off reduces the lr to 0 after some epochs ' + \
            'and slowly increases it again. May help with quantization collapse, ' + \
            'but not necessary in most cases.')
    lr_arg_group.add_argument('--lr_drop_enc',
            help='drop lr for encoder to zero and increase slowly',
            action='store_true')
    lr_arg_group.add_argument('--lr_drop_all',
            help='drop lr for all to zero and increase slowly',
            action='store_true')
    lr_arg_group.add_argument('--lr_drop_epoch',
            help='epoch to drop learning rate to zero',
            type=int, default=-1)
    lr_arg_group.add_argument('--lr_rtrn_epochs',
            help='number of epochs to increase learning rate to normal after drop',
            type=int, default=-1)

    warmup_arg_group = argparser.add_argument_group('Transformer warmup hyperparams',
            'With transformer warmup, QT is trained without quantization for ' + \
            'some epochs, and then gradually introduces quantization. Improves ' + \
            'training stability.')
    warmup_arg_group.add_argument('--no_transformer_warmup',
            help='disable transformer warmup before quantization',
            action='store_true')
    warmup_arg_group.add_argument('--warmup_epochs',
            help='don\'t quantize at all for this many epochs (default: 4)',
            type=int, default=4)
    warmup_arg_group.add_argument('--no_warmup_annealing',
            help='disable slow decrease of non-quantized residual coefficient',
            action='store_true')
    warmup_arg_group.add_argument('--warmup_annealing_min',
            help='minimum residual coefficient for non-quantized path (default: 0.0)',
            type=float, default=0.0)
    warmup_arg_group.add_argument('--warmup_annealing_epochs',
            help='non-quantized residual reduction lasts this many epochs (default: 2)',
            type=int, default=2)

    kmeans_arg_group = argparser.add_argument_group('K-means initialization hyperparams',
            'Initialize codebook with kmeans after transformer warmup')
    kmeans_arg_group.add_argument('--no_kmeans',
            help='disable kmeans codebook initialization after warmup',
            action='store_true')
    kmeans_arg_group.add_argument('--kmeans_batches',
            help='number of batches for kmeans (default: 100)', 
            type=int, default=100)
    kmeans_arg_group.add_argument('--kmeans_iter',
            help='number of iterations for kmeans (default: 50)',
            type=int, default=50)

    other_arg_group = argparser.add_argument_group('Other arguments')
    other_arg_group.add_argument('--run_id',
            help='unique run id (for logging and saved models)',
            type=str, default='run1')
    other_arg_group.add_argument('--gpu', help='gpu device to use (default: use cpu)',
            type=int, default=-1)
    other_arg_group.add_argument('--logdir',
            help='directory to put tensorboard logs (default: \'../logs\')',
            type=str, default='../logs')
    other_arg_group.add_argument('--log_every',
            help='log every n forward passes (default: 50)',
            type=int, default=50)
    other_arg_group.add_argument('--savedir',
            help='directory to put saved model snapshots (default: \'../models\')',
            type=str, default='../models')
    other_arg_group.add_argument('--save_every',
            help='save model snapshot every N epochs (default: save on every epoch)',
            type=int, default=1)
    other_arg_group.add_argument('--seed',
            help='random seed',
            type=int, default=1)
    other_arg_group.add_argument('--data_seed',
            help='random seed for dataset (only affects batching and entity subsampling)',
            type=int, default=1)

    args = argparser.parse_args()

    seed(args.data_seed)

    if args.gpu >= 0:
        device = torch.device('cuda:{0}'.format(args.gpu))
    else:
        device = torch.device('cpu')

    data_path = args.data
    spm_path = args.sentencepiece
    save_path = args.savedir
    log_path = args.logdir

    # read data from json file
    f = open(data_path, 'r')
    data = json.load(f)
    f.close()

    # initialize dataset
    dataset = ReviewDataset(data, sample_size=args.max_num_entities, spmodel=spm_path,
            max_sen_len=args.max_sen_len, max_rev_len=args.max_rev_len)
    vocab_size = dataset.vocab_size
    nclasses = dataset.nclasses

    # prepare train/dev/test splits
    dataset.split()

    # samplers for each split
    train_sampler = \
            ReviewBucketBatchSampler(dataset, args.batch_size, split='train')
    dev_sampler = \
            ReviewBucketBatchSampler(dataset, args.batch_size, split='dev')
    test_sampler = \
            ReviewBucketBatchSampler(dataset, args.batch_size, split='test')

    # wrapper for collate function
    collator = ReviewCollator(padding_idx=dataset.pad_id(), unk_idx=dataset.unk_id(),
                              bos_idx=dataset.bos_id(), eos_idx=dataset.eos_id())

    # one dataloader per split
    train_dl = DataLoader(dataset, batch_sampler=train_sampler,
            collate_fn=collator.collate_reviews_generation)
    dev_dl = DataLoader(dataset, batch_sampler=dev_sampler,
            collate_fn=collator.collate_reviews_generation)
    test_dl = DataLoader(dataset, batch_sampler=test_sampler,
            collate_fn=collator.collate_reviews_generation)
    nbatches_trn = len(train_dl)
    nbatches_dev = len(dev_dl)
    nbatches_tst = len(test_dl)

    pad_id = dataset.pad_id()
    bos_id = dataset.bos_id()
    eos_id = dataset.eos_id()
    unk_id = dataset.unk_id()

    torch.manual_seed(args.seed)

    # define model
    model = QuantizedTransformerModel(
                vocab_size,
                d_model=args.d_model,
                temp=args.ema_temp,
                num_samples=args.ema_num_samples,
                codebook_size=args.codebook_size,
                commitment_cost=args.commitment_cost,
                nlayers=args.nlayers,
                internal_nheads=args.internal_nheads,
                output_nheads=args.output_nheads,
                d_ff=args.d_ff,
                use_in_pos=args.in_pos,
                use_out_pos=args.out_pos,
                ema_decay=args.ema_decay,
                dropout=args.dropout)
    model.to(device)

    # prepare optimizer and learning rate scheduler
    if args.lr_drop_all:
        optimizer = \
            torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_calc)
    elif args.lr_drop_enc:
        param_groups = \
                [
                    {'params': model.in_emb.parameters()},
                    {'params': model.encoder.parameters()},
                    {'params': model.decoder.parameters()},
                    {'params': model.linear.parameters()}
                ]
        lambda1 = lr_calc
        lambda2 = lambda epoch: args.lr_decay ** epoch
        lr_lambdas = [lambda1, lambda1, lambda2, lambda2]
        optimizer = \
            torch.optim.Adam(param_groups, lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambdas)
    else:
        optimizer = \
            torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, args.lr_decay)


    # define losses
    if args.label_smoothing == 0.0:
        criterion = \
                nn.CrossEntropyLoss(ignore_index=pad_id, reduction='sum')
    else:
        criterion = \
                LabelSmoothingLoss(args.label_smoothing, vocab_size, ignore_index=pad_id)
    valid_criterion = nn.CrossEntropyLoss(ignore_index=pad_id, reduction='sum')

    criterion = criterion.to(device)
    valid_criterion = valid_criterion.to(device)

    # prepare transformer warmup scheduler
    if (not args.no_transformer_warmup) and (not args.no_warmup_annealing):
        warmup_scheduler = \
                ResidualCoefficientScheduler(args.warmup_epochs,
                        args.warmup_annealing_epochs, nbatches_trn,
                        min_coeff=args.warmup_annealing_min)

    if args.logdir != '':
        tb_writer = tb.SummaryWriter(os.path.join(log_path, args.run_id))

    for epoch in range(args.epochs):
        # initialize loss and counts for accuracy
        running_loss = 0.0
        running_g_loss = 0.0
        running_q_loss = 0.0
        running_ppl = 0.0
        model.train()

        # quantize or not
        quantize = (args.no_transformer_warmup or epoch >= args.warmup_epochs)

        # if warmup is over, initialize codebook with kmeans
        if (not args.no_kmeans) and epoch == args.warmup_epochs:
            with torch.no_grad():
                model.eval()
                sentence_vecs = []
                for i, batch in enumerate(train_dl):
                    if i == args.kmeans_batches:
                        break

                    src, tgt, gld = [x.to(device) for x in batch]
                    out, _, _, _ = model.encode(src, quantize=False)
                    sentence_vecs.append(out.reshape(-1, args.d_model).detach().to('cpu'))
                sentence_vecs = torch.cat(sentence_vecs, dim=0).detach().numpy()

                kmeans_codebook, _ = kmeans(sentence_vecs, args.codebook_size, iter=args.kmeans_iter)

                # in case of missing clusters, fill in random ones.
                # missing cluster may occur when there are identical
                # sentence vectors in the clustered data
                if kmeans_codebook.shape[0] < args.codebook_size:
                    num_missing_clusters = args.codebook_size - kmeans_codebook.shape[0]
                    new_clusters = np.random.randn(num_missing_clusters, args.d_model)
                    kmeans_codebook = np.concatenate((kmeans_codebook, new_clusters), axis=0)

                model.encoder.set_codebook(torch.Tensor(kmeans_codebook))
                model.train()

            # save model snapshot
            if args.save_every is not None:
                torch.save(model, '{0}/{1}_{2}pkm_model.pt'.format(save_path, args.run_id, epoch))

        for i, batch in enumerate(train_dl):
            src, tgt, gld = [x.to(device) for x in batch]
            batch_size, nsent, src_ntokens = src.size()
            
            optimizer.zero_grad()

            if not args.no_transformer_warmup:
                if not args.no_warmup_annealing:
                    residual_coeff = warmup_scheduler.get_residual_coefficient(i, epoch)
                else:
                    residual_coeff = 0.0 if quantize else 1.0
            else:
                residual_coeff = 0.0

            out, encodings, q_loss, perplexity = \
                    model(src, tgt, quantize=quantize, residual_coeff=residual_coeff)
            if args.label_smoothing > 0.0:
                out = F.log_softmax(out, dim=-1)

            g_loss = criterion(out.flatten(end_dim=-2), gld.flatten())
            non_padding_elem = (tgt != pad_id).sum().item()
            g_loss /= batch_size * nsent
            q_loss *= float(non_padding_elem) / (batch_size * nsent)

            loss = g_loss + q_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_g_loss += g_loss.item()
            if quantize:
                running_q_loss += q_loss.item()
                running_ppl += perplexity.item()

            # log average loss per batch every k passes
            if args.logdir != '' and i % args.log_every == args.log_every - 1:
                step = epoch * nbatches_trn + i
                running_uq_loss = running_q_loss / args.commitment_cost
                lrs = lr_scheduler.get_lr()
                lr_enc = lrs[0]
                if len(lrs) > 1:
                    lr_dec = lrs[2]
                else:
                    lr_dec = lr_enc
                tb_writer.add_scalar('loss/train', running_loss / args.log_every, step)
                tb_writer.add_scalar('g_loss/train', running_g_loss / args.log_every, step)
                tb_writer.add_scalar('q_loss/train', running_q_loss / args.log_every, step)
                tb_writer.add_scalar('uq_loss/train', running_uq_loss / args.log_every, step)
                tb_writer.add_scalar('perplexity/train', running_ppl / args.log_every, step)
                tb_writer.add_scalar('residual_coeff/train', residual_coeff, step)
                tb_writer.add_scalar('learning_rate/enc', lr_enc, step)
                tb_writer.add_scalar('learning_rate/dec', lr_dec, step)
                running_loss = 0.0
                running_g_loss = 0.0
                running_q_loss = 0.0
                running_ppl = 0.0

        with torch.no_grad():
            # initialize loss
            running_loss = 0.0
            running_g_loss = 0.0
            running_q_loss = 0.0
            running_ppl = 0.0
            model.eval()
            for i, batch in enumerate(dev_dl):
                src, tgt, gld = [x.to(device) for x in batch]
                batch_size, nsent, src_ntokens = src.size()

                out, encodings, q_loss, perplexity = \
                        model(src, tgt, quantize=quantize, residual_coeff=residual_coeff)
                g_loss = valid_criterion(out.flatten(end_dim=-2), gld.flatten())

                non_padding_elem = (tgt != pad_id).sum().item()
                g_loss /= batch_size * nsent
                q_loss *= float(non_padding_elem) / (batch_size * nsent)

                loss = g_loss + q_loss

                running_loss += loss.item()
                running_g_loss += g_loss.item()
                if quantize:
                    running_q_loss += q_loss.item()
                    running_ppl += perplexity.item()

                # log average loss per batch every k passes
                if args.logdir != '' and i % args.log_every == args.log_every - 1:
                    step = epoch * nbatches_dev + i
                    running_uq_loss = running_q_loss / args.commitment_cost
                    tb_writer.add_scalar('loss/dev', running_loss / args.log_every, step)
                    tb_writer.add_scalar('g_loss/dev', running_g_loss / args.log_every, step)
                    tb_writer.add_scalar('q_loss/dev', running_q_loss / args.log_every, step)
                    tb_writer.add_scalar('uq_loss/dev', running_uq_loss / args.log_every, step)
                    tb_writer.add_scalar('perplexity/dev', running_ppl / args.log_every, step)
                    tb_writer.add_scalar('residual_coeff/dev', residual_coeff, step)
                    running_loss = 0.0
                    running_g_loss = 0.0
                    running_q_loss = 0.0
                    running_ppl = 0.0

            # initialize loss
            running_loss = 0.0
            running_g_loss = 0.0
            running_q_loss = 0.0
            running_ppl = 0.0
            model.eval()
            for i, batch in enumerate(test_dl):
                src, tgt, gld = [x.to(device) for x in batch]
                batch_size, nsent, src_ntokens = src.size()

                out, encodings, q_loss, perplexity = \
                        model(src, tgt, quantize=quantize, residual_coeff=residual_coeff)
                g_loss = valid_criterion(out.flatten(end_dim=-2), gld.flatten())

                non_padding_elem = (tgt != pad_id).sum().item()
                g_loss /= batch_size * nsent
                q_loss *= float(non_padding_elem) / (batch_size * nsent)

                loss = g_loss + q_loss

                running_loss += loss.item()
                running_g_loss += g_loss.item()
                if quantize:
                    running_q_loss += q_loss.item()
                    running_ppl += perplexity.item()

                # log average loss per batch every k passes
                if args.logdir != '' and i % args.log_every == args.log_every - 1:
                    step = epoch * nbatches_dev + i
                    running_uq_loss = running_q_loss / args.commitment_cost
                    tb_writer.add_scalar('loss/test', running_loss / args.log_every, step)
                    tb_writer.add_scalar('g_loss/test', running_g_loss / args.log_every, step)
                    tb_writer.add_scalar('q_loss/test', running_q_loss / args.log_every, step)
                    tb_writer.add_scalar('uq_loss/test', running_uq_loss / args.log_every, step)
                    tb_writer.add_scalar('perplexity/test', running_ppl / args.log_every, step)
                    tb_writer.add_scalar('residual_coeff/test', residual_coeff, step)
                    running_loss = 0.0
                    running_g_loss = 0.0
                    running_q_loss = 0.0
                    running_ppl = 0.0

        # save model snapshot
        if args.save_every is not None and epoch % args.save_every == args.save_every - 1:
            torch.save(model, '{0}/{1}_{2}_model.pt'.format(save_path, args.run_id, epoch+1))

        # decay learning rate
        if args.lr_decay > 0:
            lr_scheduler.step()
