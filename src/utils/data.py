import os
import os.path
import json
from collections import defaultdict, Counter
from random import sample, choice, shuffle
from tqdm import tqdm
from time import time

import torch
from torch.utils.data import Dataset, Sampler

import numpy as np
import sentencepiece as spm


class ReviewDataset(Dataset):
    '''Stores a Review Dataset'''
    def __init__(self, data, sample_size=None, spmodel=None, 
            max_sen_len=1000, max_rev_len=1000):
        '''Parameters:
            data (dict): dataset in appropriate json format.
            sample_size (int): number of entities to load
            spmodel (str): filename of sentencepiece model
            max_sen_len (int): maximum number of tokens/pieces in sentence
            max_rev_len (int): maximum number of sentences in review
        '''
        self.ids = []
        self.entity_ids = []
        self.reviews = defaultdict(dict)
        self.labels = defaultdict(dict)
        self.lengths = {}
        self.nclasses = 0
        self.dev_entity_ids = None
        self.test_entity_ids = None
        self.has_splits = False

        self.max_sen_len = max_sen_len
        self.max_rev_len = max_rev_len

        if sample_size is not None:
            data = sample(data, sample_size)

        # stores unprocessed data
        for entity_data in data:
            entity_id = entity_data['entity_id']
            self.entity_ids.append(entity_id)

            if 'split' in entity_data:
                if self.dev_entity_ids is None:
                    self.dev_entity_ids = []
                    self.test_entity_ids = []

                if entity_data['split'] == 'dev':
                    self.dev_entity_ids.append(entity_id)
                else:
                    self.test_entity_ids.append(entity_id)

            for review_data in entity_data['reviews']:
                review_id = review_data['review_id']
                if 'rating' in review_data:
                    label = review_data['rating'] - 1
                else:
                    label = 0

                if label + 1 > self.nclasses:
                    self.nclasses = label + 1

                full_id = '__'.join([entity_id, review_id])
                sentences = review_data['sentences']
                self.ids.append(full_id)
                self.reviews[entity_id][review_id] = sentences
                self.labels[entity_id][review_id] = label

        # load sentencepiece vocabulary
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(spmodel)
        self.vocab_size = len(self.sp)

    def __getitem__(self, index):
        '''Fetches vectorized example and stores data and lengths'''
        if type(index) is int:
            full_id = self.ids[index]
        else:
            full_id = index

        entity_id, review_id = full_id.split('__')
        label = self.labels[entity_id][review_id]
        review = self.reviews[entity_id][review_id][:self.max_rev_len]
        vectorized_review = [self.vectorize(sentence) for sentence in review]

        if full_id not in self.lengths:
            sen_len = [len(sentence) for sentence in vectorized_review]
            self.lengths[full_id] = sen_len
            self.lengths_updated = True

        return vectorized_review, label, full_id

    def __len__(self):
        '''Returns number of examples'''
        return len(self.ids)

    def split(self, train=0.8, dev=0.1, test=0.1, shuf=True):
        '''Splits dataset into train/dev/test splits'''
        if self.has_splits:
            return

        if shuf:
            shuffle(self.ids)

        size = len(self.ids)
        train_size = int(np.round(size * train))
        dev_size = int(np.round(size * dev))
        test_size = int(np.round(size * test))

        self.train_ids = self.ids[:train_size]
        self.dev_ids = self.ids[train_size:train_size + dev_size]
        self.test_ids = self.ids[train_size + dev_size:train_size + dev_size + test_size]
        self.has_splits = True

    def vectorize(self, sentence):
        '''Vectorizes a sentence using sentencepiece'''
        return self.sp.EncodeAsIds(sentence)[:self.max_sen_len]

    def unvectorize(self, token_ids):
        '''Returns sentence string from vectorized sentence'''
        if type(token_ids) != 'list':
            token_ids = token_ids.tolist()
        try:
            end = token_ids.index(self.eos_id())
        except:
            end = len(token_ids) + 1
        return self.sp.DecodeIds(token_ids[:end])

    def unvectorize_review(self, review):
        '''Returns review string from matrix'''
        sentences = []
        for token_ids in review:
            sentences.append(self.unvectorize(token_ids))
        return sentences

    def pad_id(self):
        return self.sp.pad_id()

    def unk_id(self):
        return self.sp.unk_id()

    def bos_id(self):
        return self.sp.bos_id()

    def eos_id(self):
        return self.sp.eos_id()


class ReviewSummarizationDataset(ReviewDataset):
    '''Stores a review summarization benchmark dataset'''
    def __init__(self, data, sample_size=None, spmodel=None, 
            max_sen_len=1000, max_rev_len=1000):
        super(ReviewSummarizationDataset, self).__init__(data, sample_size,
                spmodel, max_sen_len, max_rev_len)
        self.has_entity_splits = False

    def entity_split(self, dev_split=0.5, split_by='alphanum'):
        '''Splits benchmark to dev and test set'''
        if split_by == 'presplit':
            assert self.dev_entity_ids is not None \
                    and self.test_entity_ids is not None, \
                    'test dataset has no presplit info. use other splitting method'
            self.has_entity_splits = True
            return

        if split_by == 'alphanum':       # split by arphanumeric order
            entity_ids = sorted(self.entity_ids)
        elif split_by == 'original':     # splits by order they appeared in file
            entity_ids = self.entity_ids
        else:                            # splits randomly
            entity_ids = sample(self.reviews.keys(), len(self.reviews))

        dev_size = int(len(entity_ids) * dev_split)
        self.dev_entity_ids = entity_ids[:dev_size]
        self.test_entity_ids = entity_ids[dev_size:]
        self.has_entity_splits = True

    def get_entity_batch_samplers(self, batch_size=5, split=None, pct=1.0):
        '''Returns one batch sampler per entity'''
        samplers = {}
        if split is None:
            entity_ids = self.reviews.keys()
        else:
            assert split == 'dev' or split == 'test', 'Unknown split (use dev/test)'
            assert self.has_entity_splits, 'No entity split found'
            if split == 'dev':
                entity_ids = self.dev_entity_ids
            else:
                entity_ids = self.test_entity_ids

        for entity_id in self.reviews:
            samplers[entity_id] = \
                    EntityReviewBucketBatchSampler(self, entity_id, batch_size, pct)
        return samplers



class ReviewBucketBatchSampler(Sampler):
    '''Batch sampler that attempts to minimize padding by only batching together
       reviews with the same number of sentences (i.e., in the same bucket). Within
       a bucket, reviews are sorted according to the length of their longest sentence
       to further minimize padding. Batches are shuffled when the DataLoader iterator
       is initialized.
    '''
    def __init__(self, dataset, batch_size, shuffle_batches=True, split=None):
        '''Initializes sampler by creating and sorting buckets'''
        if split is None:
            ids = dataset.ids
        else:
            assert dataset.has_splits, 'Dataset doesn\'t have train/dev/test splits'
            if split == 'train':
                ids = dataset.train_ids
            elif split == 'dev':
                ids = dataset.dev_ids
            else:
                ids = dataset.test_ids

        lengths = dataset.lengths
        self.batch_size = batch_size
        self.shuffle_batches = shuffle_batches

        self.buckets = defaultdict(list)
        for full_id in tqdm(ids, disable=True):
            if full_id not in lengths:
                dataset[full_id] # calls __getitem__ to load vectorized lengths
            rev_len = len(lengths[full_id])
            self.buckets[rev_len].append(full_id)

        self.batch_list = []
        for bucket_len in tqdm(sorted(self.buckets), disable=True): 
            # sort reviews within bucket
            self.buckets[bucket_len].sort(key=lambda full_id:max(lengths[full_id]))

            # batch_list holds the bucket length and starting index of the batch
            self.batch_list += [(bucket_len, start_idx)
                    for start_idx in range(0, len(self.buckets[bucket_len]), self.batch_size)]

    def __iter__(self):
        if self.shuffle_batches:
            shuffle(self.batch_list)
        for bucket_len, start_idx in self.batch_list:
            yield self.buckets[bucket_len][start_idx:start_idx + self.batch_size]

    def __len__(self):
        return len(self.batch_list)


class EntityReviewBucketBatchSampler(Sampler):
    '''Batch sampler that attempts to minimize padding by only batching together
       reviews with the same number of sentences (i.e., in the same bucket). Within
       a bucket, reviews are sorted according to the length of their longest sentence
       to further minimize padding. Batches are shuffled when the DataLoader iterator
       is initialized.
    '''
    def __init__(self, dataset, entity_id, batch_size, pct=1.0):
        '''Initializes sampler by creating and sorting buckets'''
        ids = ['__'.join([entity_id, review_id])
                for review_id in dataset.reviews[entity_id]]

        if pct < 1.0:
            num_ids = int(len(ids) * pct)
            shuffle(ids)
            ids = ids[:num_ids]

        lengths = dataset.lengths
        self.batch_size = batch_size

        self.buckets = defaultdict(list)
        for full_id in tqdm(ids, disable=True):
            if full_id not in lengths:
                dataset[full_id] # calls __getitem__ to load vectorized lengths
            rev_len = len(lengths[full_id])
            self.buckets[rev_len].append(full_id)

        self.batch_list = []
        for bucket_len in tqdm(sorted(self.buckets), disable=True): 
            # sort reviews within bucket
            self.buckets[bucket_len].sort(key=lambda full_id:max(lengths[full_id]))

            # batch_list holds the bucket length and starting index of the batch
            self.batch_list += [(bucket_len, start_idx)
                    for start_idx in range(0, len(self.buckets[bucket_len]), self.batch_size)]

    def __iter__(self):
        for bucket_len, start_idx in self.batch_list:
            yield self.buckets[bucket_len][start_idx:start_idx + self.batch_size]

    def __len__(self):
        return len(self.batch_list)


class ReviewCollator():
    '''Wrapper class around collate functions which allows for collate parameters
    to be set (e.g., bos, eos, pad ids)
    '''
    def __init__(self, padding_idx=0, unk_idx=1, bos_idx=2, eos_idx=3):
        self.padding_idx = padding_idx
        self.unk_idx = unk_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx

    def collate_reviews(self, samples):
        '''Collates samples in a batch, by adding appropriate padding'''
        reviews, labels, _ = zip(*samples)
        max_tokens = max([len(sentence) for review in reviews for sentence in review])
        max_sentences = max([len(review) for review in reviews])

        for review in reviews:
            for sentence in review:
                sentence.extend([self.padding_idx] * (max_tokens - len(sentence)))
            review.extend([[self.padding_idx] * max_tokens] * (max_sentences - len(review)))

        return torch.tensor(reviews), torch.tensor(labels)

    def collate_reviews_with_ids(self, samples):
        '''Collates samples in a batch, by adding appropriate padding'''
        reviews, labels, full_ids = zip(*samples)
        max_tokens = max([len(sentence) for review in reviews for sentence in review])
        max_sentences = max([len(review) for review in reviews])

        for review in reviews:
            for sentence in review:
                sentence.extend([self.padding_idx] * (max_tokens - len(sentence)))
            review.extend([[self.padding_idx] * max_tokens] * (max_sentences - len(review)))

        return torch.tensor(reviews), torch.tensor(labels), full_ids

    def collate_reviews_generation(self, samples):
        '''Collates samples in a batch, by adding appropriate padding and bos/eos tokens'''
        src_reviews, labels, _ = zip(*samples)
        max_tokens = max([len(src_sentence) for src_review in src_reviews for src_sentence in src_review])
        max_sentences = max([len(src_review) for src_review in src_reviews])

        tgt_reviews = []
        gld_reviews = []
        for src_review in src_reviews:
            tgt_review = []
            gld_review = []

            for src_sentence in src_review:
                tgt_sentence = [self.bos_idx] + src_sentence
                gld_sentence = src_sentence + [self.eos_idx]

                src_sentence.extend([self.padding_idx] * (max_tokens - len(src_sentence)))
                tgt_sentence.extend([self.padding_idx] * (max_tokens + 1 - len(tgt_sentence)))
                gld_sentence.extend([self.padding_idx] * (max_tokens + 1 - len(gld_sentence)))

                tgt_review.append(tgt_sentence)
                gld_review.append(gld_sentence)

            src_review.extend([[self.padding_idx] * max_tokens] * (max_sentences - len(src_review)))
            tgt_review.extend([[self.padding_idx] * max_tokens] * (max_sentences - len(tgt_review)))
            gld_review.extend([[self.padding_idx] * max_tokens] * (max_sentences - len(gld_review)))

            tgt_reviews.append(tgt_review)
            gld_reviews.append(gld_review)

        return torch.tensor(src_reviews), torch.tensor(tgt_reviews), torch.tensor(gld_reviews)


def spm_train(filename, prefix, vocab_size=32000, coverage=1.0, model_type='unigram'):
    args = '--input={0} '.format(filename) \
         + '--model_prefix={0} '.format(prefix) \
         + '--vocab_size={0} '.format(vocab_size) \
         + '--character_coverage={0} '.format(coverage) \
         + '--model_type={0} '.format(model_type) \
         + '--hard_vocab_limit=false ' \
         + '--pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3'
    spm.SentencePieceTrainer.Train(args)

def spm_train_sentence_limit(filename, prefix, max_sents=4000000,
        vocab_size=32000, coverage=1.0, model_type='unigram'):
    args = '--input={0} '.format(filename) \
         + '--model_prefix={0} '.format(prefix) \
         + '--vocab_size={0} '.format(vocab_size) \
         + '--character_coverage={0} '.format(coverage) \
         + '--model_type={0} '.format(model_type) \
         + '--hard_vocab_limit=false ' \
         + '--input_sentence_size={0} --shuffle_input_sentence=true '.format(max_sents) \
         + '--pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3'
    spm.SentencePieceTrainer.Train(args)

def spm_train_from_json(fname_in, prefix, fname_tmp='./spm.tmp',
        max_sents=None, vocab_size=32000, coverage=1.0, model_type='unigram'):
    f = open(fname_in, 'r')
    data = json.load(f)
    f.close()

    fout = open(fname_tmp, 'w')
    print('writing sentences to temp file')
    for entity_data in data:
        for review_data in entity_data['reviews']:
            fout.write('\n'.join(review_data['sentences']) + '\n')
    fout.close()

    print('launching spm training')
    if max_sents is None:
        spm_train(fname_tmp, prefix, vocab_size, coverage, model_type)
    else:
        spm_train_sentence_limit(fname_tmp, prefix, max_sents,
                vocab_size, coverage, model_type)

    print('cleaning up')
    os.remove(fname_tmp)
