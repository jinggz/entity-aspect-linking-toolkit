from pathlib import Path
import bs4 as bs
import urllib.request
from urllib.parse import quote
import time
import pprint as pp
import logging
import json
import re

from data_reader import DataReader

logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
logger.setLevel(logging.INFO)

class EntityPage:
    def __init__(self):
        self.page_dict = dict()
        self.soup = None

    def get_connection(self, entity):
        try:
            # example: https://en.wikipedia.org/wiki/Segal%E2%80%93Bargmann_space
            connection = urllib.request.urlopen('https://en.wikipedia.org/wiki/' + quote(entity))
        except Exception as e:
            logger.exception(e, exc_info=False)
            return []
        return connection

    def retrieve_wiki_page(self, entity):
        # To connect and extract the page content of the Wiki Page of the entity
        connection = self.get_connection(entity)
        logger.info('got connection successfully')
        if connection:
            source = connection.read()
            soup = bs.BeautifulSoup(source, 'lxml')
            # remove unused tags
            [x.extract() for x in soup.find_all('sup')]
            [x.extract() for x in soup.find_all(class_='mw-editsection')]
            [x.extract() for x in soup.find_all(class_='image')]
            self.soup = soup

    def build_page_dict(self):

        body = self.soup.find(id='bodyContent')
        h2_list = self.get_h2_list()

        # extract the lead paragraphs before the first heading
        start = body.find(class_='mw-parser-output')
        leading = []
        for chi in start.children:
            if chi.name == 'p':
                leading.append(chi.text.strip())
            if chi.name == 'h2':
                break
        leading = " ".join(leading)
        self.page_dict['summary'] = leading
        # extract h2 headings and its content, entities and sub headings
        for h2 in h2_list:
            nextNode = h2.next_sibling
            p = []
            links = []
            heads = []
            while nextNode and nextNode.name != 'h2':
                if not isinstance(nextNode, bs.element.NavigableString):
                    p.append(nextNode.text.strip())
                    for link in nextNode.find_all('a', href=self.__not_file):
                        links.append(urllib.parse.unquote((link.get('href'))))
                    # get sub heads within a h2 head
                    if nextNode.name in ['h3', 'h4', 'h5', 'h6']:
                        heads.append(nextNode.text.strip().lower())
                nextNode = nextNode.next_sibling
            links = " ".join(links)
            contents = " ".join(p)
            self.page_dict[h2.text.strip().lower()] = {'content': contents, 'links': links, 'heads': heads}


    def get_h2_list(self):
        # http://trec-car.cs.unh.edu/process/dataselection.html
        # extract all h2 headings of a Wiki page and remove frequent headings like 'see also' etc.
        h_remove = ["see also", "contents",
                    "references",
                    "external links",
                    "notes",
                    "bibliography",
                    "gallery",
                    "publications",
                    "further reading",
                    "track listing",
                    "sources",
                    "cast",
                    "discography",
                    "awards",
                    "other"]
        body = self.soup.find(id='bodyContent')
        h2_set = []
        for h2 in body.find_all('h2'):
            if not h2.text.strip().lower() in h_remove:
                h2_set.append(h2)
        return h2_set

    #useless for now
    def get_text_bw_2h2(self, mark):
        '''
        get the content between 2 sections
        :param: mark: the start section name. typ: str
        :return: the whole text between 2 sections as a string
        '''
        contents = []
        links = []
        for elt in mark.parent.next_siblings:
            if elt.name == "h2":
                break
            try:
                for link in elt.find_all('a', href=self.__not_file):
                    links.append(urllib.parse.unquote((link.get('href'))))
                contents.append(elt.text)
            except Exception as e:
                logger.exception(e, exc_info=False)
        links = " ".join(links)
        contents = " ".join(contents)
        type(contents)
        return contents, links

    def __not_file(self, href):
        return href and not re.compile("File").search(href) and re.compile("wiki").search(href)


if __name__ == '__main__':

    dir = Path(__file__).parent.parent
    input_file = Path.joinpath(dir, 'output/sentences_eal_subj.json')
    output_file = Path.joinpath(dir, 'output/entity_dict/')
    if not Path(output_file).is_dir():
            Path(output_file).mkdir()


    # load the data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f, encoding='utf-8')
    logger.info('Sentence data loaded')

    # load list of entities
    entities = set(row['entity'] for row in data)
    entities.discard('None')
    # 19985 entities, among them 10410 unique entities
    logger.info('Remove empty entity')

    # begin crawling
    wiki_dict = dict()
    logger.info('Start to create dictionary for Wiki page... ')
    entities = list(entities)
    for entity in entities:
        instance = EntityPage()
        instance.retrieve_wiki_page(entity)
        if instance.soup:
            instance.build_page_dict()
        wiki_dict[entity] = instance.page_dict

    #TODO save the entire dic
    saver = DataReader()
    saver.save2json(Path.joinpath(output_file, entities[i] + '.json'), instance.page_dict)



    #         #logger.info('Saved the dictionary of %s to %s.' % (entity, output_file) )
    logger.info('Finished dictionary creation.')

    # pp.pprint(entity_sect_text_dict['Cello Suites (Bach)'])
    # saver = DataReader()
    # saver.save2json(Path.joinpath(dir, 'output/entity_dicts.json'), entity_sect_text_dict)
    # logger.info('Saved the dictionary of entities to file.')


