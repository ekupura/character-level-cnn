import nltk
from nltk.corpus import wordnet as wn


def wordnet_preprocess():
    nltk.download('wordnet')


def collect_synonyms_for_sentiment():
    nltk.download('wordnet')

    # collect synonyms for 'good'
    good_synonyms = []
    for ss in wn.synsets('good'):
        good_synonyms.extend(ss.lemma_names())
        good_synonyms.extend([sim.lemma_names()[0] for sim in ss.similar_tos()])
    good_synonyms = list(set(good_synonyms))

    # collect synonyms for 'bad'
    bad_synonyms = []
    for ss in wn.synsets('bad'):
        bad_synonyms.extend(ss.lemma_names())
        bad_synonyms.extend([sim.lemma_names()[0] for sim in ss.similar_tos()])
    bad_synonyms = list(set(bad_synonyms))

    return good_synonyms, bad_synonyms


if __name__ == '__main__':
    good, bad = collect_synonyms_for_sentiment()
