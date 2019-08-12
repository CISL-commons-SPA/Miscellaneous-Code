import os
from snorkel import SnorkelSession
from snorkel.parser import TSVDocPreprocessor #data parsing 
from snorkel.parser.spacy_parser import Spacy ##data parsing package
from snorkel.parser import CorpusParser #for parsing data
from snorkel.models import Document, Sentence ##for counting
from snorkel.models import candidate_subclass #for creating candidates
from snorkel.candidates import Ngrams, CandidateExtractor ##extraction
from snorkel.matchers import PersonMatcher #Extraction
from util import number_of_people
from util import load_external_labels

#Initialize a snorkel session
session = SnorkelSession()
n_docs = 500 if 'CI' in os.environ else 2591


#load the corpus, by reading the documents
doc_preprocessor = TSVDocPreprocessor('articles.tsv.txt', max_docs=n_docs)

#parsing the document
corpus_parser = CorpusParser(parser=Spacy())
corpus_parser.apply(doc_preprocessor, count = n_docs)

#counting documents processed and sentences processed
print("Documents:", session.query(Document).count())
print("Sentences:", session.query(Sentence).count())

#creating candidates to make predictions
Spouse = candidate_subclass('Spouse', ['person1','person2'])

#extraction
ngrams = Ngrams(n_max=7)
person_matcher = PersonMatcher(longest_match_only=True)
cand_extractor = CandidateExtractor(Spouse, [ngrams,ngrams],[person_matcher,person_matcher])

docs = session.query(Document).order_by(Document.name).all()
train_sents = set()
dev_sents = set()
test_sents = set()
#split data into sets
for i, doc in enumerate(docs):
    for s in doc.sentences:
        if number_of_people(s)<=5:
            if i % 10 == 8:
                dev_sents.add(s)
            elif i % 10 == 9:
                test_sents.add(s)
            else:
                train_sents.add(s)
                
#extractor
for i, sents in enumerate([train_sents,dev_sents,test_sents]):
    cand_extractor.apply(sents, split=i)
    print("Number of candidates:", session.query(Spouse).filter(Spouse.split==i).count())

missed = load_external_labels(session, Spouse, annotator_name='gold')

