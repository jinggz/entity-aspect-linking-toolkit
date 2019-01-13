# Entity Aspect Linking System
The acpect here is referred to the section headings of a page from WikiPedia.

This branch is for EXTERNAL use. DO NOT MERGE.

## Getting started
download model file in 'model' folder

install requirements.txt via pip 

set 'model_file' in main.py to your local model path

## Usage
python main.py

    eal = EAL(model_file)
    aspect_predicted = eal.get_prediction(sentence, entity)
## How it works
wiki_crawler.py: get an entity as the input, 
connect to Wikipedia website and return a built dictionary with section headings as keys and contents as values.

Regarding the headings of h2, I remove content and headings based on the selection criteria on (http://trec-car.cs.unh.edu/process/dataselection.html ). 
I also collected the paragraphs between the title and the table of the contents, named as 'lead_paragraph'.

tfidf_ranking.py: get an sentence and an entity as the input,
retrieve the entity dictionary via wiki_crawler.py,
and then return the closest aspect of the entity for the sentence