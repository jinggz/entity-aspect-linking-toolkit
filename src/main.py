import os
import logging
from pathlib import Path
from eal_ranking import EAL, TfidfRanking, WembRanking

logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s %(name)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
logger.setLevel(logging.INFO)

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
    print('Predicted most relevant aspect is: %s(score: %.4f)' % (aspect_predicted, score))





