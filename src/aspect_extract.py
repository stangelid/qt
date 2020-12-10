import sys
import os
import os.path
import json
import re
import argparse
from collections import defaultdict
from random import seed
from time import time
import math

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer

import torch
from torch.utils.data import DataLoader
from torch.distributions.multinomial import Multinomial

from encoders import *
from quantizers import *
from train import *
from utils.data import *
from utils.summary import truncate_summary, RougeEvaluator

            
if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description='Extracts aspect summaries with a trained QT model.\n' + \
                        'For usage example, refer to:\n' + \
                        '\thttps://github.com/stangelid/qt')

    data_arg_group = argparser.add_argument_group('Data arguments')
    data_arg_group.add_argument('--summary_data',
            help='summarization benchmark data',
            type=str, default='../data/json/space_summ.json')
    data_arg_group.add_argument('--gold_data',
            help='gold data root directory',
            type=str, default='../data/gold')
    data_arg_group.add_argument('--gold_aspects',
            help='aspect categories to evaluate against (default: all SPACE aspects)',
            type=str, default='building,cleanliness,food,location,rooms,service')
    argparser.add_argument('--seedsdir',
            help='directory that holds aspect query words, i.e., seeds',
            type=str, default='../data/seeds')
    argparser.add_argument('--max_num_seeds',
            help='number of seed words per aspect',
            type=int, default=5)
    data_arg_group.add_argument('--sentencepiece',
            help='sentencepiece model file',
            type=str, default='../data/sentencepiece/spm_unigram_32k.model')
    data_arg_group.add_argument('--max_rev_len',
            help='maximum number of sentences per review (default: 150)',
            type=int, default=150)
    data_arg_group.add_argument('--max_sen_len',
            help='maximum number of tokens per sentence (default: 40)',
            type=int, default=40)
    data_arg_group.add_argument('--split_by',
            help='how to split summary data (use "alphanum" for SPACE)',
            type=str, default='alphanum')

    summ_arg_group = argparser.add_argument_group('Summarizer arguments')
    summ_arg_group.add_argument('--model',
            help='trained QT model to use',
            type=str, default='')
    summ_arg_group.add_argument('--manual_head',
            help='manually set aspect head for extraction (default: auto via entropy)',
            type=int, default=None)
    summ_arg_group.add_argument('--truncate_clusters',
            help='truncate cluster sampling to top-p % of clusters (if < 1) or top-k (if > 1)',
            type=float, default=1.0)
    summ_arg_group.add_argument('--num_cluster_samples',
            help='number of cluster samples (default: 300)',
            type=int, default=300)
    summ_arg_group.add_argument('--sample_sentences',
            help='enable 2-step sampling (sample sentences within cluster neighbourhood)',
            action='store_true')
    summ_arg_group.add_argument('--truncate_cluster_nn',
            help='truncate sentences that live in a cluster neighborhood (default: 5)',
            type=int, default=5)
    summ_arg_group.add_argument('--num_sent_samples',
            help='number of sentence samples per cluster sample (default: 30)',
            type=int, default=30)
    summ_arg_group.add_argument('--temp',
            help='temperature for sampling sentences within cluster (default: 10)',
            type=int, default=3)


    out_arg_group = argparser.add_argument_group('Output control')
    out_arg_group.add_argument('--outdir',
            help='directory to put summaries',
            type=str, default='../outputs')
    out_arg_group.add_argument('--max_tokens',
            help='summary budget in words (default: 40)',
            type=int, default=40)
    out_arg_group.add_argument('--min_tokens',
            help='minimum summary sentence length in words (default: 1)',
            type=int, default=1)
    out_arg_group.add_argument('--cos_thres', 
            help='cosine similarity threshold for extraction (default: 1.0)',
            type=float, default=1.0)
    out_arg_group.add_argument('--no_cut_sents',
            help='don\'t cut last summary sentence',
            action='store_true')
    out_arg_group.add_argument('--no_early_stop',
            help='allow last sentence to go over limit',
            action='store_true')
    out_arg_group.add_argument('--newline_sentence_split',
            help='one sentence per line (don\'t use if evaluating with ROUGE)',
            action='store_true')

    other_arg_group = argparser.add_argument_group('Other arguments')
    other_arg_group.add_argument('--run_id',
            help='unique run id (for outputs)',
            type=str, default='aspect_run1')
    other_arg_group.add_argument('--no_eval',
            help='don\'t evaluate (just write summaries)',
            action='store_true')
    other_arg_group.add_argument('--gpu',
            help='gpu device to use (default: -1, i.e., use cpu)',
            type=int, default=-1)
    other_arg_group.add_argument('--batch_size',
            help='the maximum batch size (default: 5)',
            type=int, default=5)
    other_arg_group.add_argument('--sfp', 
            help='system filename pattern for pyrouge',
            type=str, default='(.*)')
    other_arg_group.add_argument('--mfp',
            help='model filename pattern for pyrouge',
            type=str, default='#ID#_[012].txt')
    other_arg_group.add_argument('--seed',
            help='random seed',
            type=int, default=1)
    args = argparser.parse_args()

    seed(1)

    if args.gpu >= 0:
        device = torch.device('cuda:{0}'.format(args.gpu))
    else:
        device = torch.device('cpu')

    # set paths
    summ_data_path = args.summary_data
    model_path = args.model
    output_path = os.path.join(args.outdir, args.run_id)
    eval_path = args.outdir
    gold_path = args.gold_data
    seeds_path = args.seedsdir
    spm_path = args.sentencepiece

    assert args.model != '', 'Please give model path'

    # read aspect seed words
    aspects = args.gold_aspects.split(',')
    num_aspects = len(aspects)
    aspect_indices = {}
    aspect_seeds = {}
    for i, aspect in enumerate(aspects):
        aspect_indices[aspect] = i
        seeds = []
        f = open(os.path.join(seeds_path, aspect + '.txt'), 'r')
        for line in f:
            _, seed_word = line.split()
            seeds.append(seed_word)
        f.close()
        aspect_seeds[aspect] = set(seeds[:args.max_num_seeds])

    # aspect mapping tools
    token_pattern = re.compile(r'(?u)\b\w\w+\b')
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    # load summarization data
    f = open(summ_data_path, 'r')
    summ_data = json.load(f)
    f.close()

    # prepare summarization dataset
    summ_dataset = ReviewSummarizationDataset(summ_data, spmodel=spm_path,
            max_rev_len=args.max_rev_len, max_sen_len=args.max_sen_len)
    vocab_size = summ_dataset.vocab_size
    pad_id = summ_dataset.pad_id()
    bos_id = summ_dataset.bos_id()
    eos_id = summ_dataset.eos_id()
    unk_id = summ_dataset.unk_id()

    # wrapper for collate function
    collator = ReviewCollator(padding_idx=pad_id, unk_idx=unk_id,
                                bos_idx=bos_id, eos_idx=eos_id)

    # split dev/test entities
    summ_dataset.entity_split(split_by=args.split_by)

    # create entity data loaders
    summ_dls = {}
    summ_samplers = summ_dataset.get_entity_batch_samplers(args.batch_size)
    for entity_id, entity_sampler in summ_samplers.items():
        summ_dls[entity_id] = DataLoader(summ_dataset, batch_sampler=entity_sampler,
                collate_fn=collator.collate_reviews_with_ids)

    torch.manual_seed(args.seed)

    model = torch.load(model_path, map_location=device)
    nheads = model.encoder.output_nheads
    codebook_size = model.codebook_size
    d_model = model.d_model
    model.eval()

    ranked_entity_sentences = defaultdict(dict)
    with torch.no_grad():
        # encode and assign all review sentences for all entities
        all_texts = []
        all_assignments = []
        texts = {}
        assignments = {}
        distances = {}
        for entity_id, entity_loader in summ_dls.items():
            max_ntokens = 0
            entity_texts = []
            entity_assignments = []
            entity_distances = []
            for batch in entity_loader:
                src = batch[0].to(device)
                ids = batch[2]
                for full_id in ids:
                    entity_id, review_id = full_id.split('__')
                    entity_texts.extend(summ_dataset.reviews[entity_id][review_id])

                batch_size, nsent, ntokens = src.size()
                max_ntokens = max(max_ntokens, ntokens)

                noq_out, q_out, clusters, dists = model.cluster(src)

                entity_assignments.append(clusters.reshape(-1, nheads).detach())
                entity_distances.append(dists.detach().reshape(-1, nheads, codebook_size))

            entity_assignments = torch.cat(entity_assignments, dim=0)    # [S x H]
            entity_distances = torch.cat(entity_distances, dim=0) # [S x H x C]

            texts[entity_id] = entity_texts
            if args.manual_head is not None:
                assignments[entity_id] = \
                        entity_assignments[:,args.manual_head].unsqueeze(1).cpu() # [S x 1]
                distances[entity_id] = \
                        entity_distances[:,args.manual_head].unsqueeze(1).cpu() # [S x 1 x C]
                auto_head = False
            else:
                assignments[entity_id] = entity_assignments.cpu()   # [S x H]
                distances[entity_id] = entity_distances.cpu()       # [S x H x C]
                auto_head = True

            all_texts.extend(entity_texts)
            all_assignments.append(assignments[entity_id])

        # saves memory
        del model

        # all tokenized texts in one list
        tokenized_sentences = []
        for sentence in all_texts:
            tokens = [tok for tok in token_pattern.findall(sentence.lower())
                        if tok not in stop_words]
            tokens = [lemmatizer.lemmatize(tok) for tok in tokens]
            tokenized_sentences.append(tokens)

        # all cluster assignments in one matrix
        all_assignments = torch.cat(all_assignments, dim=0).to(device)

        # create cluster-to-aspect mapping
        cl_to_asp = {}
        cl_to_asp['raw_counts'] = torch.zeros((num_aspects, codebook_size)).to(device)
        for sentence_idx in range(all_assignments.shape[0]):
            sentence_clusters = all_assignments[sentence_idx]
            sentence_tokens = tokenized_sentences[sentence_idx]
            for token in sentence_tokens:
                for aspect in aspects:
                    if token in aspect_seeds[aspect]:
                        cl_to_asp['raw_counts'][aspect_indices[aspect]][sentence_clusters] += 1

        # add noise to max elements (per cluster) to avoid bias towards low aspect indices
        mask = cl_to_asp['raw_counts'] == cl_to_asp['raw_counts'].max(dim=0)[0]
        zeros = torch.zeros(mask.shape)
        noise = torch.randn(mask.shape[0], mask.shape[1]) * 0.001
        cl_to_asp['raw_counts'] += \
                torch.where(mask.to(device), noise.to(device), zeros.to(device))
        cl_to_asp['raw_counts'] = torch.abs(cl_to_asp['raw_counts'])

        # normalize
        cl_to_asp['p_a_given_cl'] = cl_to_asp['raw_counts'] / cl_to_asp['raw_counts'].sum(dim=0)
        cl_to_asp['p_cl_given_a'] = cl_to_asp['raw_counts'] / cl_to_asp['raw_counts'].sum(dim=1, keepdims=True)
        cl_to_asp['mapping'] = cl_to_asp['p_a_given_cl'].argmax(dim=0)

        # automatically determine best head to use for aspect summarization
        if auto_head:
            total_nsent = all_assignments.shape[0]
            total_assignments_onehot = \
                    torch.zeros(total_nsent, nheads, codebook_size).to(device) # [S x H x C]
            total_assignments_onehot.scatter_(2, all_assignments.unsqueeze(2), 1)

            # saves memory
            del all_assignments

            # number of cluster assignments for each head
            total_cluster_sizes = total_assignments_onehot.sum(dim=0)            # [H x C]

            # finds the head which produces every cluster
            cluster_heads = total_cluster_sizes.argmax(dim=0)

            # re-computes aspect probabilities, ensuring non is exactly zero
            counts = cl_to_asp['raw_counts'] + 1e-6
            p_a_given_cl = counts / counts.sum(dim=0)

            # computes cluster entropies
            p_logp = p_a_given_cl * p_a_given_cl.log()
            cluster_entropies = -p_logp.sum(dim=0)
            cluster_entropies_scattered = torch.zeros(nheads, codebook_size).to(device)
            cluster_entropies_scattered.scatter_(0,
                    cluster_heads.unsqueeze(0), cluster_entropies.unsqueeze(0))

            # computes number of clusters per head, to get mean head entropy
            clusters_per_head = (cluster_entropies_scattered > 0).sum(dim=1)
            head_scores = cluster_entropies_scattered.sum(dim=1) / clusters_per_head
            head_to_use = head_scores.argmin().item()

            # saves memory
            del counts
            del p_a_given_cl
            del p_logp

        for entity_id, entity_loader in summ_dls.items():
            nsent = assignments[entity_id].shape[0]
            if auto_head:
                used_nheads = 1
                assignments[entity_id] = \
                        assignments[entity_id][:,head_to_use].unsqueeze(1) # [S x 1]
                distances[entity_id] = \
                        distances[entity_id][:,head_to_use].unsqueeze(1)   # [S x 1 x C]
            else:
                used_nheads = assignments[entity_id].shape[1]

            # put on GPU
            assignments[entity_id] = assignments[entity_id].to(device)
            distances[entity_id] = distances[entity_id].to(device)

            assignments_onehot = \
               torch.zeros(nsent, used_nheads, codebook_size).to(device)   # [S x H x C]
            assignments_onehot.scatter_(2, assignments[entity_id].unsqueeze(2), 1)
            cluster_sizes = assignments_onehot.sum(dim=0)             # [H x C]

            # defining cluster neighborhoods
            # [k x H x C]             # [k x H x C]
            cluster_knn_sent_dist, cluster_knn_sent_idx = \
                    torch.topk(distances[entity_id], k=args.truncate_cluster_nn,
                            dim=0, largest=False)

            # no longer needed
            del assignments[entity_id]
            del distances[entity_id]

            for aspect in aspects:
                aspect_idx = aspect_indices[aspect]
                aspect_cluster_sizes = torch.zeros_like(cluster_sizes).to(device)
                aspect_cluster_sizes.copy_(cluster_sizes)
                aspect_cluster_sizes.add_(1e-6)
                aspect_mask = cl_to_asp['mapping'] != aspect_idx
                aspect_cluster_sizes[:,aspect_mask] = 0

                # truncating possible clusters to be sampled
                if args.truncate_clusters > 1:
                    vals_to_keep, indices_to_keep = \
                            aspect_cluster_sizes.topk(int(args.truncate_clusters), dim=1)
                    filtered_cluster_sizes = torch.zeros_like(aspect_cluster_sizes)
                    filtered_cluster_sizes.scatter_(1, indices_to_keep, vals_to_keep)
                    aspect_cluster_sizes = filtered_cluster_sizes
                elif args.truncate_clusters < 1:
                    num_clusters_to_keep = \
                            (torch.sum(aspect_cluster_sizes >= 1, dim=1)
                                    * args.truncate_clusters).round().long()
                    sorted_sizes, _ = aspect_cluster_sizes.sort(dim=1, descending=True)
                    min_sizes = sorted_sizes.gather(1, num_clusters_to_keep.unsqueeze(1))
                    indices_to_remove = torch.lt(aspect_cluster_sizes, min_sizes)
                    filtered_cluster_sizes = torch.zeros_like(aspect_cluster_sizes)
                    filtered_cluster_sizes.copy_(aspect_cluster_sizes)
                    filtered_cluster_sizes[indices_to_remove] = 0
                    aspect_cluster_sizes = filtered_cluster_sizes

                # normalize cluster probabilities
                aspect_cluster_sizes *= cl_to_asp['p_cl_given_a'][aspect_idx]

                # sampling clusters
                # [H x nSamples]
                sampled_clusters = aspect_cluster_sizes.multinomial(args.num_cluster_samples, replacement=True)
                # [k x H x nSamples]
                sampled_clusters = sampled_clusters.expand(args.truncate_cluster_nn, -1, -1)

                # gathering all possible sentences to be sampled or ranked
                # [k x H x nSamples]
                sampled_neighbourhoods = cluster_knn_sent_idx.gather(2, sampled_clusters)

                if args.sample_sentences:
                    # [k x H x nSamples]
                    sampled_neighbourhood_dist = \
                            cluster_knn_sent_dist.gather(2, sampled_clusters)

                    # multinomial over sentences
                    multi = Multinomial(total_count=args.num_sent_samples,
                            logits=-sampled_neighbourhood_dist.transpose(0, 2) / args.temp)

                    # [k x H x C]
                    samples = multi.sample().transpose(0, 2)

                    # [k x H x nSent]
                    scattered_sampled_sentences_counts = \
                            torch.zeros(args.truncate_cluster_nn, used_nheads, nsent).to(device)
                    scattered_sampled_sentences_counts.scatter_add_(2, sampled_neighbourhoods, samples)

                    # [nSent]
                    sampled_sentences_counts = scattered_sampled_sentences_counts.sum(dim=[0,1])
                else:
                    # [k*H*nSample]
                    sampled_sentences = sampled_neighbourhoods.flatten()

                    # ranking by number of times a cluster was sampled
                    # [nSent] 
                    sampled_sentences_counts = sampled_sentences.bincount(minlength=nsent)

                # getting ranked sentence texts
                ranked_sentence_counts, ranked_sentence_indices = \
                        sampled_sentences_counts.sort(descending=True)
                ranked_sentence_texts = [texts[entity_id][idx] for idx in ranked_sentence_indices]
                ranked_entity_sentences[entity_id][aspect] = ranked_sentence_texts

    # tfidf vectorizer used for cosine threshold
    if args.cos_thres != -1:
        vectorizer = TfidfVectorizer(decode_error='replace', stop_words='english')
        vectorizer.fit(all_texts)
    else:
        vectorizer = None

    # write summaries
    dict_results = {'dev':{}, 'test':{}, 'all':{}}
    all_outputs = []
    if args.newline_sentence_split:
        delim = '\n'
    else:
        delim = '\t'
    for aspect in aspects:
        aspect_output_path = os.path.join(output_path, aspect)
        os.makedirs(aspect_output_path, exist_ok=True)
        for entity_id in ranked_entity_sentences:
            if entity_id in summ_dataset.dev_entity_ids:
                file_path = os.path.join(aspect_output_path, 'dev_' + entity_id)
            else:
                file_path = os.path.join(aspect_output_path, 'test_' + entity_id)

            ranked_sentences = ranked_entity_sentences[entity_id][aspect]
            summary_sentences = truncate_summary(ranked_sentences,
                    max_tokens=args.max_tokens, cut_sents=(not args.no_cut_sents),
                    vectorizer=vectorizer, cosine_threshold=args.cos_thres,
                    early_stop=(not args.no_early_stop),
                    min_tokens=args.min_tokens)

            fout = open(file_path, 'w')
            fout.write(delim.join(summary_sentences))
            fout.close()

        if args.no_eval:
            continue 

        # evaluate summaries
        model_dir = os.path.join(gold_path, aspect)
        dev_evaluator = RougeEvaluator(system_dir=aspect_output_path,
                                   model_dir=model_dir,
                                   system_filename_pattern='dev_'+args.sfp,
                                   model_filename_pattern=args.mfp)
        test_evaluator = RougeEvaluator(system_dir=aspect_output_path,
                                   model_dir=model_dir,
                                   system_filename_pattern='test_'+args.sfp,
                                   model_filename_pattern=args.mfp)
        all_evaluator = RougeEvaluator(system_dir=aspect_output_path,
                                   model_dir=model_dir,
                                   system_filename_pattern='[^_]*_'+args.sfp,
                                   model_filename_pattern=args.mfp)


        outputs = dev_evaluator.evaluate()
        dict_results['dev'][aspect] = outputs['dict_output']
        all_outputs.append('{0} vs {1} [dev]'.format(args.run_id, aspect))
        all_outputs.append(outputs['short_output'] + '\n')

        outputs = test_evaluator.evaluate()
        dict_results['test'][aspect] = outputs['dict_output']
        all_outputs.append('{0} vs {1} [test]'.format(args.run_id, aspect))
        all_outputs.append(outputs['short_output'] + '\n')

        outputs = all_evaluator.evaluate()
        dict_results['all'][aspect] = outputs['dict_output']
        all_outputs.append('{0} vs {1} [all]'.format(args.run_id, aspect))
        all_outputs.append(outputs['short_output'] + '\n')

    if not args.no_eval:
        ftxt = open(os.path.join(eval_path, 'eval_{0}.txt'.format(args.run_id)), 'w')
        ftxt.write('\n'.join(all_outputs))
        ftxt.close()

        fjson = open(os.path.join(eval_path, 'eval_{0}.json'.format(args.run_id)), 'w')
        fjson.write(json.dumps(dict_results))
        fjson.close()
