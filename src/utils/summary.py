from pyrouge import Rouge155
from sklearn.metrics.pairwise import cosine_similarity
import string
import re
import logging

PUNCT = set(string.punctuation)

def truncate_summary(ranked_sentences, max_tokens=75, min_tokens=1,
        cut_sents=False, early_stop=True, remove_non_alpha=True,
        vectorizer=None, cosine_threshold=None):
    '''Truncates a summary by iteratively adding sentences
       until the max_tokens limit is passed. 
    '''
    count = 0
    summary = []
    summary_sentence_ids = []

    if vectorizer is not None:
        assert cosine_threshold > 0 and cosine_threshold <= 1, \
                'cosine threshold should be in (0,1]'
        sentence_vecs = vectorizer.transform(ranked_sentences)
        similarities = cosine_similarity(sentence_vecs)

    for i, sentence in enumerate(ranked_sentences):
        if remove_non_alpha and all(c.isdigit() or c in PUNCT for c in sentence):
            continue

        if len(sentence.split()) < min_tokens:
            continue

        if vectorizer is not None and i > 0:
            similarities_to_existing = similarities[i,summary_sentence_ids]
            if not all(similarities_to_existing < cosine_threshold):
                continue

        summary.append(sentence)
        summary_sentence_ids.append(i)

        count += len(sentence.split())
        if count > max_tokens:
            if cut_sents:
                last_sent = summary[-1].split()
                last_sent = last_sent[:len(last_sent)-count+max_tokens]
                if len(last_sent) > 0:
                    summary[-1] = ' '.join(last_sent)
                else:
                    summary = summary[:-1]
                break
            else:
                summary = summary[:-1]
                if early_stop:
                    break
                else:
                    count -= len(sentence.split())
                    summary_sentence_ids = summary_sentence_ids[:-1]

    return summary


class RougeEvaluator():
    '''Wrapper for pyrouge'''
    def __init__(self, system_filename_pattern='([0-9]*)',
                       model_filename_pattern='#ID#_[012].txt',
                       system_dir=None, model_dir=None,
                       log_level=logging.WARNING):
        self.system_dir = system_dir
        self.model_dir = model_dir

        self.r = Rouge155()
        self.r.log.setLevel(log_level)
        self.r.system_filename_pattern = system_filename_pattern
        self.r.model_filename_pattern = model_filename_pattern

        self.results_regex = \
                re.compile('(ROUGE-[12L]) Average_F: ([0-9.]*) \(95.*?([0-9.]*) - ([0-9.]*)')

    def evaluate(self, system_dir=None, model_dir=None):
        if system_dir is None:
            assert self.system_dir is not None, 'no system_dir given'
            system_dir = self.system_dir
        if model_dir is None:
            assert self.model_dir is not None, 'no model_dir given'
            model_dir = self.model_dir

        self.r.system_dir = system_dir
        self.r.model_dir = model_dir

        full_output = self.r.convert_and_evaluate()
        results = self.results_regex.findall(full_output)

        outputs = {}
        outputs['full_output'] = full_output
        outputs['dict_output'] = self.r.output_to_dict(full_output)
        outputs['short_output'] = '\n'.join(['  {0} {1} ({2} - {3})'.format(*r) for r in results])

        return outputs
