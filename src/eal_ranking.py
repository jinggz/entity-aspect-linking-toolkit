import os
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import numpy as np
import logging
from abc import abstractmethod
from gensim.test.utils import get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

import nlp_preprocessing
from wiki_crawler import EntityPage

logger = logging.getLogger('main')
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

class EAL:
    def __init__(self, model_file):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.load_model(model_file)

    @abstractmethod
    def load_model(self, model_file):
        raise NotImplementedError("You should implement this!")

    def get_aspects_dict(self, entity):
        # adapt to external use
        entity = entity.strip()
        EP = EntityPage(entity)
        self.logger.info('Connected to Wikipedia')
        if EP.soup:
            EP.build_page_dict()
        self.logger.info('Built dictionary of entity aspects...')
        return EP.page_dict

    @abstractmethod
    def get_vector(self, text_list):
        '''
        :param text_list:
        :type: list of str
        :return:
        '''
        raise NotImplementedError("You should implement this!")

    def get_aspects_vect(self, entity):
        y_aspects = []
        y_content = []
        for k, v in self.get_aspects_dict(entity).items():
            y_aspects.append(k)
            y_content.append(v['content'])
        y_feature = self.get_vector(y_content)
        return y_aspects, y_feature

    def cos_sim(self, a, b):
        return cosine_similarity(a, b).flatten()

    def get_prediction(self, sentence, entity):
        '''
        return the closest aspect of a given entity appeared in a given sentence
        :param sentence: a given sentence containing a representation of an entity
        :type: str
        :param entity: a given entity identified in the sentence
        :type: str
        :return: the closet aspect found in the Wikipedia page of the given entity
        :type: str
        '''

        # get aspect dict and aspects vector
        try:
            y_aspects, y_feature = self.get_aspects_vect(entity)
        except ValueError as error:
            self.logger.error(error)
            self.logger.error("The page of the entity contains no proper aspect.")
            return "", 0
        # sentence vector
        x_feature = self.get_vector([sentence])
        self.logger.info("calculating the most relevant aspect...")
        cos_ranking = self.cos_sim(x_feature, y_feature)
        if len(cos_ranking) == 1: # only contain 1 aspect (summary)
            y_pred = "summary"
            cos_sim = cos_ranking[0]
        else:
            ind = np.argpartition(cos_ranking, -2)[-2:]  # get indexes of 2 biggest values
            if ind[1] != 0: # if largest score is not 'summary'
                y_pred = y_aspects[ind[1]]
                cos_sim = cos_ranking[ind[1]]
            else:
                y_pred = y_aspects[ind[0]]
                cos_sim = cos_ranking[ind[0]]
        return y_pred, cos_sim


class TfidfRanking(EAL):

    def load_model(self, model_file):
        #load tfidf model
        with open(model_file, 'rb') as f:
            self.model = pickle.load(f)

        self.logger.info('started TF-IDF ranking with model file: %s' % model_file)

    def get_vector(self, text_list):
        '''
        using tfidf method
        :param text:
        :type: list of str
        :return: sparse matrix
        '''
        text_list = nlp_preprocessing.nlp_pipeline(text_list)
        return self.model.transform(text_list)

class WembRanking(EAL):
    def load_model(self, model_file):
        #load glove300d pretrained vector
        tmp_file = get_tmpfile("temp_word2vec.txt")
        glove2word2vec(model_file, tmp_file)
        self.model = KeyedVectors.load_word2vec_format(tmp_file, binary=False)

        self.logger.info('Glove pretrained vector model loaded')

    def get_vector(self, text_list):
        '''
        using word embedding method
        :param text:
        :type: list of str
        :return: sparse matrix
        '''
        text_list = nlp_preprocessing.nlp_pipeline(text_list)
        res = []
        for text in text_list:
            res.append(self.sentemb(text))
        return res

    def sentemb(self, sentence):
        '''
        return aggregate average embedding vector for a sentence
        :param sentence: a long string
        :type:str
        :return:
        '''
        words = sentence.split()
        article_embedd = []
        for word in words:
            try:
                embed_word = self.model[word]
                article_embedd.append(embed_word)
            except KeyError:
                continue
        if article_embedd == []:
            return np.zeros(300)
        else:
            article_embedd = np.asarray(article_embedd)
            avg = np.average(article_embedd, axis=0)
        return avg

if __name__ == '__main__':

    if os.getenv('method') in ['tfidf', 'wemb']:
        logger.info('ranking method set to ' + os.environ['method'])
    else:
        raise NameError("""Please set an environment variable to indicate which ranking method to use.\n
        Your options are: method='tfidf' or 'wemb'.\n""")

    dir = Path(__file__).parent.parent
    # set 'model_file' to your own path
    if os.environ['method'] == 'tfidf':
        model_file = Path.joinpath(dir, 'model', 'tfidf_model.pkl')
        eal = TfidfRanking(model_file)
    elif os.environ['method'] == 'wemb':
        model_file = Path.joinpath(dir, 'model', 'glove.6B.300d.txt')
        eal = WembRanking(model_file)

    while True:
        sentence = input('enter sentence:')
        entity = input('enter entity: ')
        logging.info('start predicting...')
        aspect_predicted, score = eal.get_prediction(sentence, entity)
        logging.info('end predicting.')
        print('Predicted most relevant aspect is: %s(score: %.4f)' % (aspect_predicted,score))


