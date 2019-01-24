# Entity Aspect Linking System
The acpect here is referred to the section headings of a page from WikiPedia.

This repo is an implementation of entity aspect linking, which is used as a part of Toolkit on.

## Getting started
download tfidf model file in 'model' folder

download Glove pretrained 300d vector from: https://nlp.stanford.edu/projects/glove/ 

install requirements.txt via pip 

## Usage
1. set up environment variable 'method' to 'tfidf' or 'wemb'    
        
        os.environ['method'] = ['tfidf', 'wemb']
2. set 'model_file' in main.py to your local model path
3. python main.py

        if os.environ['method'] == 'tfidf':
            model_file = Path.joinpath(dir, 'model', 'tfidf_model.pkl')
            eal = TfidfRanking(model_file)
        elif os.environ['method'] == 'wemb':
            model_file = Path.joinpath(dir, 'model', 'glove.6B.300d.txt')
            eal = WembRanking(model_file)
            
        aspect_predicted = eal.get_prediction(sentence, entity)
## How it works
**wiki_crawler.py**: 

get an entity as the input, 
connect to Wikipedia website and return a built dictionary with section headings as keys and contents as values.

Regarding the headings of h2, I remove content and headings based on the selection criteria on (http://trec-car.cs.unh.edu/process/dataselection.html ). 
I also collected the paragraphs between the title and the table of the contents, named as 'lead_paragraph'.

**eal_ranking.py**: 

There are 2 ranking methods that you can choose. One is using tf-idf as the text vector representation. 
The other is using the average word embeddings which uses the pre-trained Glove vectors.

Then get an sentence and an entity as the input,
retrieve the entity dictionary via wiki_crawler.py,
and then return the closest aspect of the entity for the sentence
