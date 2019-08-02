import nltk
import pprint
import sys
import os
import time
import subprocess
import spacy
from collections import defaultdict
from nltk.tag import StanfordPOSTagger
from nltk import word_tokenize
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.chunk import *
from nltk.chunk.util import *
from nltk.chunk.regexp import *
from pprint import pprint
from pprint import pformat
from nltk import Tree

"""
Configurable parameter descriptions:
   use_threshold: Use Wordnet Senses to filter out "ambiguous verbs"
   WNVerbsOnly: When using threshold, limit sense count to verbs only
   threshold = >0: When using threshold, this is the maximum 
        number of senses allowed for a given word 
   lemmatize_cues: Whether or not to lemmatize discourse cues
   lemmatize_candidates: Whether or not to lemmatize candidate verbs within the sentence
   debug: print debug stuff
   UsePhraseExtraction: Toggle whether or not to use causal 
        phrase extraction to detect verbs
   addGirju: Add Girju's causal cue list to existing cue list
   sentChunkNumber: Number of sentences to part-of-speech tag at a time
   maxSents: Total Number of sentences to process (-1 -> process all sentences)
   ranking: Use Text-Mining ranking to filter causal phrases
"""
use_threshold = False
WNVerbsOnly = True
NERTagNHs = True
threshold = 1
lemmatize_candidates = True
lemmatize_cues = True
debug = False
UsePhraseExtraction = True
addGirju = True
sentChunkNumber = 100000
maxSents = -1
ranking = False
ID = "WithoutRanking"

"""
    Discourse cue file, and headline files to read through
"""
dir_path = os.path.dirname(os.path.realpath(__file__))
cue_file = dir_path + \
    "\\Data-Files\\CausalCues_WithoutModifiers_WithoutComments.txt"
sentenceFile = dir_path + "\\Data-Files\\india-news-headlines.txt"
sentenceFile2 = dir_path + "\\Data-Files\\1million-abcnews-date-text.txt"

# Parse Sem Eval set based on whether Sem-Eval dataset is used
ParseSemEval = (sentenceFile == dir_path + "DataFiles\\semtest.txt")

print("Use threshold: %r\nWNVerbsOnly: %r\nthreshold: %d\n\
Lemmatize Candidates: %r\nLemmatize Cues: %r\n\
Add Girju's Cues: %r\nDebug: %r\nUse Phrase Extraction: %r\n\
NER Tag Noun-Heads: %r\nParse Sem Eval: %r\nsentChunkNumber: %d\n\
Max Sentences: %d\n\n" % \
      (use_threshold, WNVerbsOnly , threshold, lemmatize_candidates,\
       lemmatize_cues, addGirju, debug,UsePhraseExtraction,NERTagNHs,\
       ParseSemEval,sentChunkNumber,maxSents))

# Misc Parameters
nlp = spacy.load("en_core_web_lg")
nlp_wiki = spacy.load("xx_ent_wiki_sm")
jar = dir_path + "\\stanford-postagger-2018-10-16\\stanford-postagger.jar"
model = dir_path + "\\stanford-postagger-2018-10-16\\models\\english-left3words-distsim.tagger"
pos_tagger = StanfordPOSTagger(model, jar, encoding='utf8')
cue_dict = defaultdict(set)
NLP_NER_DICT = { "PERSON" : "person", "NORP" : "organization",\
                 "FAC" : "landmark", "ORG" : "organization",\
                 "GPE" : "location", "LOC" : "location","PRODUCT" : "product",\
                 "EVENT" : "event", "WORK_OF_ART" : "art", "LAW" : "law",\
                 "LANGUAGE" : "language", "DATE" : "date", "TIME" : "time",\
                 "PERCENT" : "percent", "MONEY" : "money",\
                 "QUANTITY" : "quantity", "ORDINAL" : "ordinal",\
                 "CARDINAL" : "cardinal","PER" : "person" }


# Statistics
total = 0
correct_causal = 0
correct_noncausal = 0
type_1 = 0
type_2 = 0


# Set Parameter String to all paramters, starting with sentence file name
parameter_string = '_' + sentenceFile.\
    split("\\")[-1].replace(".txt","").strip().upper()
if (use_threshold):
    parameter_string += "_Threshold_" + str(threshold)
    if (WNVerbsOnly):
        parameter_string += "_VerbsOnly"
if (lemmatize_candidates):
    parameter_string += "_LemmaCandidates"
if (lemmatize_cues):
    parameter_string += "_LemmaCues"
if (debug):
    parameter_string += "_Debug"
if (UsePhraseExtraction):
    parameter_string += "_UsePhraseExtraction"
if (NERTagNHs):
    parameter_string += "_NERTagNHs"
if (ID != "0"):
    parameter_string += "_" + ID
    
causalOutput = open("causalOutput" + parameter_string + ".txt","w",buffering=1)

# Logs print statements to file
class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        #output_string = "PythonOutput"
        self.log = open("PythonOutput" + parameter_string+ ".log", "w",buffering=1)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass    

sys.stdout = Logger()

# Initialize chunking
tag_patterns = r"""
  NP: {<DT|.*P\$|CD>?<JJ>*<NN.*>+}   # chunk determiner/possessive, adjectives and noun
      {<PRP>}                 # Personal Pronouns
  PP: {<IN><NP>}              # Prepositional Phrase
  VB: {<VB.*>}                 # VERBS
  NPG: {<NP|PP>((((<\,><CC>)|<CC>)*<NP|PP>)|(<\,><NP|PP>(?!<VB>)))+}   # Noun/Prepositional Phrase group
  CP: {<NPG|NP>((<VB><NPG|NP>)|VP)}        # Potentially Causal Phrase
  
"""
chunk_parser = nltk.RegexpParser(tag_patterns)


        

"""
Get word from synset name 
e.g. 'excitement.n.04' would return 'excitement'
"""
def getName(synset):
    return synset.name().split('.')[0]


"""
    Determines whether or not the provided 'noun' string is 'Rank One' 
    - Return's true if there is a 'causation class hypernym' within the
        hypernym tree of every sense of the provided noun
"""
causationClasses = set(['human_action', 'phenomenon', 'state', 'psychological_feature'])
def isRankOne(noun):
    lem = WordNetLemmatizer().lemmatize(noun,pos="n").lower()
    if (debug):
        print("Ranking Noun Head: %s" % noun)
    senses = wn.synsets(lem)
    if (debug):
            print("\tSenses found:",senses)
    if (len(senses) == 0):
        return False
    for sense in senses:
        classes = set()
        if (debug):
            print("\t\tGoing through paths for sense: %s" % sense.name().split('.')[0])
        for path in sense.hypernym_paths():
            if (debug):
                print("\t\t\tGetting through path:",path)
            for synset in path:
                classes.add(getName(synset))
            if (len(classes & causationClasses) != 0):
                if (debug):
                    print("\t\t\t\t Causation Class Hypernym '%s' found in path for sense '%s' of noun head '%s'! " % (list((classes & causationClasses))[0],str(sense.name().split('.')[2]),noun))
                break
        if (len(classes & causationClasses) == 0):
            if (debug):
                print("\t\t\t\t Causation Class Hypernym not found in any path for sense '%s' of noun head '%s'! Returning false... " % (str(sense.name().split('.')[2]),noun))
            return False
    if (debug):
        print("\t\t\t\t Causation Class Hypernym found for every sense of noun head '%s'! Returning True" % (noun))
    return True

"""
    Grab's the noun head of the provided Noun Phrase Group (NPG) and returns it
"""
def GrabNounHead(NPG):
    if (debug):
        print("Grabbing Noun-Head from NPG (Noun-Phrase-Group) '%s':" % POStoString(NPG))
    if (NPG.label() == "NP" or NPG.label() == "PP"):
        if (debug):
            print("\tSince detected Noun Phrase (Would-Be-Group) is a '%s', simply returning last word: %s"\
                  % (NPG.label(),NPG.leaves()[-1][0]))
        return NPG.leaves()[-1][0]
    for child in NPG:
        if (type(child) == Tree and (child.label() == "NP" or child.label() == "PP")):
            if (debug):
                print("\tDetected first phrase in Noun-Phrase-Group ('%s','%s'), returning last word from phrase: %s" % (POStoString(child),child.label(),child.leaves()[-1][0]))
            return child.leaves()[-1][0]
        


"""
    Check's if the provided token is recognized by WordNet
"""
def checkWN(entList,token):
    for entity in entList:
        if (token in entity):
            return WNValid(entity.text)
    return None

"""
    Makes the provided entity WordNet readable before checking whether it is recognized
    by WordNet
"""
def WNValid(entText):
    WNString = entText.replace(" ","_")
    synsets = wn.synsets(WNString)
    if (len(synsets) != 0):
        if (debug):
            print("\t\t\tWas able to parse noun-head as wordnet synset, returning: %s" % WNString)
        return synsets[0].name().split('.')[0]
    return None

"""
    Parses the noun head if it is a part of a recognized named entity
    - Returns a pair: <NE,NH>
        NE: The unparsed named entity of the noun head (None if NH isn't a NE)
        NH: The lemmatized noun head
"""
def ParseNounHead(sentence,noun_head):
    if (NERTagNHs == False):
        if (debug):
            print("\tReturning Noun Head '%s' immediately since we're not NER Tagging(Parameter 'NERTagNHs' = false)\n" % (noun_head))
        return None,WordNetLemmatizer().lemmatize(noun_head,pos="n").lower()
    if (debug):
        print("\tParsing extracted Noun-Head '%s' (incase it is a part of a named entity)" % noun_head)
        print("\tRunning Spacy NLP using both spacy's and wiki's NER model")
    doc = nlp(sentence)
    doc_wiki = nlp_wiki(sentence)
    if (debug):
        print("\tSpacy NER Model identified: ",doc.ents)
        print("\tWiki NER Model identified: ",doc.ents)
    # Get noun head from doc token
    spacyToken = None
    wikiToken = None
    for token in doc:
        if (token.text == noun_head):
            spacyToken = token
    for token in doc_wiki:
        if (token.text == noun_head):
            wikiToken = token
    # Check if wordnet recognize's the entity as a synset
    if (debug):
        print("\tChecking if noun-head can be parsed as wordnet synset...")
        print("\t\tUsing SpacyNERModel:")
    if spacyToken != None:
        checkSpacyWordnet = checkWN(doc.ents,spacyToken)
        if (checkSpacyWordnet):
            return spacyToken.text,checkSpacyWordnet
        if (debug):
            print("\t\t\tWas unable to parse")
    if (debug):
        print("\t\tUsing WikiNERModel:")
    if wikiToken != None:
        checkWikiWordnet = checkWN(doc_wiki.ents,wikiToken)
        if (checkWikiWordnet):
            return wikiToken.text,checkWikiWordnet
        if (debug):
            print("\t\t\tWas unable to parse")
            print("Will instead return entity tags (if noun-head is an entity)")
    # Check if token is otherwise a part of a tagged entity
    if (wikiToken != None and (wikiToken.ent_iob_ == 'I' or wikiToken.ent_iob_ == 'B')):
        if (wikiToken.ent_type_ != "MISC"):
            return wikiToken.text,WordNetLemmatizer().lemmatize(wikiToken.text.replace(" ","_"),pos="n").lower()
    if (spacyToken != None and (spacyToken.ent_iob_ == 'I' or spacyToken.ent_iob_ == 'B')):
        if (NLP_NER_DICT.get(spacyToken.ent_type_)):
            return spacyToken.text,WordNetLemmatizer().lemmatize(spacyToken.text.replace(" ","_"),pos="n").lower()
        
    # Simply return the noun_head if it isn't recognized as a part of an entity
    #return noun_head,noun_head
    return None,WordNetLemmatizer().lemmatize(noun_head,pos="n").lower()

"""
    Converts POS-Tagged tree to sentence
"""
def POStoString(tree):
    ret = ""
    for i in range(len(tree.leaves())):
        ret += tree.leaves()[i][0] + ' '
    return ret.strip()

"""
   Test output for a line of text (obsolete)
"""
def PipeLineTest(line):
    print("\n" + '-'*120)
    print("\nInput Line:\n\t%s\n" % line[:-1],end='')
    sentence = ""
    val = -1
    if (ParseSemEval):
        print("Converting Line to Sentence and Causal/Non-Causal Tag:\n\t" + "parseLine(line):\n\t\t",end='')
        print("Sentence: %s\n\t\tNon-Causal(0)/Causal(1): %d" % (sentence,val))
        sentence,val = parseLine(line)
    else:
        print("Getting and stripping sentence...",end='')
        sentence = line.replace('\n',"").replace("\"","").strip()
    print("Analyzing sentence for Parts-Of-Speech and discourse cues:\n\t" + "Cue_Found = analyze(sentence):\n\t\t",end='')
    Cue_Found,NPG1,NPG2 = analyze(sentence)
    print("\nSince ",end='')
    if (Cue_Found != None):
        print("Cue_Found != None \n==> discourse cue found \n==> sentence is detected as CAUSAL!")

        causalOutput.write("Sentence: %s\nCue Found: %s\nNPG1:\n\t%s\nNPG2:\n\t%s\n\n" % (sentence,Cue_Found,POStoString(NPG1),POStoString(NPG2)))
    else:
        print("Cue_Found == None \n==> discourse cue not found in sentence \n==> sentence is detected as NON-CAUSAL!")
    if (val != -1):
        print("Causal Detection was ",end='')
        if ((Cue_Found == None and val == 0) or (Cue_Found != None and val == 1)):
            print("CORRECT!")
        else:
            print("INCORRECT!")
            if (Cue_Found == None):
                print("Sentence was actually CAUSAL! (False Negative/Type 2 Error)")
            else:
                print("Sentence was actually NON-CAUSAL! (False Positive/Type 1 Error)")
    return Cue_Found

"""
Parses a Sem-Eval line into the sentence string and whether or not it's causal
Val = 1 -> Sentence is causal
Val = 0 -> Sentence is non-causal
"""
def parseLine(line):
    val = int(line[-2])
    sentence = line[:-2].replace("\"","").strip()
    return sentence,val

"""
Attempts to match the given candidate verb (formatted as index "candidate" 
s.t. verb = sentence[candidate]) to a discourse cue in cue_dict
"""
def matchToCue(candidate,sentence):
    if (debug):
        print("\t\t\t\tLemmatize Verb Candidate Flag is ",end='')
    if (lemmatize_candidates):
        if (debug):
            print("set, lemmatizing:\n\t\t\t\t\t%s --> " % (sentence[candidate]),end='')
        word = WordNetLemmatizer().lemmatize(word=sentence[candidate],pos="v")
        if (debug):
            print(word)
    else:
        word = sentence[candidate]
        if (debug):
            print("not set, candidate '" + word + "' is being compared to cues directly")
    if (debug):
        print("\t\t\t\t" + "Checking if word is in cue dictionary:")
    if (cue_dict.get(word) == None):
        if (debug):
            print("\t\t\t\t\t" + "Candidate verb not found, returning 'None'")
        return None
    else:
        if (debug):
            print("\t\t\t\t" + "Matched candidate verb to causal cue list")
        matched_string = None
        if (debug):
            print("\t\t\t\t\t" + "Attempting to match candidate verb to remaining part of discourse cue:" + "\n\t\t\t\t\t\t" + "Options:",end='')
            print(cue_dict.get(word))
        for value in cue_dict.get(word):
            if (debug):
                print("\t\t\t\t\t\t\t" + "Attempting to match to: '%s'" % value)
            if (matched_string != None and matched_string != ""):
                break
            if (value == ""):
                if (debug):
                    print("\t\t\t\t\t\t\t\t" + "Matched to empty string")
                    print("\t\t\t\t\t\t\t\t" + "(which means '%s' is a single-word cue)," % word)
                    print("\t\t\t\t\t\t\t\t" + "will check for other, multi-word options.")                    
                matched_string = ""
                continue
            else:
                match_to = ' '.join(sentence[candidate+1:candidate + len(value.split(' '))])
                if (debug):
                    print("\t\t\t\t\t\t\t\t" + "Remaining part of sentence to match to:" + match_to)
                if (value == match_to):
                    matched_string = value
        if (matched_string == None):
            if (debug):
                print("\t\t\t\t"+ "Failed to match candidate verb to discourse cue, returning None")
            return None
        else:
            if (debug):
                print("\t\t\t\t"+ "Matched candidate verb/phrase to discourse cue, returning \"" + word + matched_string + "\"")
            if (matched_string != ""):
                return word + " " + matched_string
            else:
                return word

"""
 Analyze Sentence for discourse cues via either:
          1)Looking for all verbs within the sentence (any verbs tagged with "VB.*")
          2)Looking for verbs within causal phrases (e.g. Noun-Phrase -> Verb -> Noun Phrase)
"""
def analyze(sentence,POS_Tags):
    if (debug):
        print("Tagging using Stanford POS Tagger ('pos_tagger.tag(word_tokenize(sentence))')\n\t\t",end = '')
    if (debug):
        print(POS_Tags)
    if (debug == 1):
        print("\tSentence converted to word-tokenized list (mimicking POS-Tagger tokenization):\n\t\t",end=''),print(sentence)
    candidates = list()
    NPG1 = list()
    NPG2 = list()
    if (debug == 1):
        print("\n\t"+ "Finding Candidates By: ",end= '')
    if (UsePhraseExtraction == 1):
        if (debug):
            print("Matching Verbs that are within Detected Causal Phrases")
        SentenceTree = chunk_parser.parse(POS_Tags)
        if (debug):
            print("\t\tChunked Sentence:\n\t\t\t",end='')
            SentenceTree.pprint(indent=20,margin=150)
        for child in SentenceTree:
            if (type(child) == Tree and child.label() == "CP"):
                if (debug == 1):
                    print("\t\t" + "Causal Phrase Detected:\n\t\t\t", child.leaves())
                    print("\t\t\t" + "Verb:",child[1].leaves()[0][0])
                candidates.append(SentenceTree.leaves().index(child[1].leaves()[0]))
                NPG1.append(child[0])
                NPG2.append(child[2])
    else:
        print("Attempting to Discourse Match to All Verbs in Sentence")
        for i in range(0,len(POS_Tags)):
            if (len(POS_Tags[i][1]) >= 2 and POS_Tags[i][1][:2] == "VB"):
                if (debug == 1):
                    print("\t\t\t",end=''),print(POS_Tags[i])
                candidates.append(i)
    if (debug and len(candidates) == 0):
        print("\t\t\t--No Candidates Found--")
    elif (debug == 1):
        print()
    if (debug == 1 and len(candidates) > 0):
        print("\t\t" + "Iterating through verb-candidates to potentially match to discourse cue:")
        print("\t\t" + "--Return Value = 'None' means verb did not match any discourse cues (therefore, verb does not imply sentence is causal)\n\t\t--Return Value != 'None' means verb matched a discourse cue and sentence is assumed causal")
    for i in candidates:
        if (debug):
            print("\n\t\t\t" + "matchToCue() for candidate '" + sentence[i] + "':")
        return_val = matchToCue(i,sentence)
        if (debug):
            print("\t\t\t\t" + "Return Value: ",end='')
            if (return_val == None):
                print("None")
            else:
                print(return_val)
        if (return_val != None):
            if (debug):
                print("\t\t\t" + "Non-Empty Return Value (i.e. discourse cue '%s') matched, returning discourse cue" % return_val)
            return return_val,NPG1[candidates.index(i)],NPG2[candidates.index(i)]
        if (debug):
            print("\t\t\t" + "Discourse cue not found for current candidate")
    if (debug):
        print("\t\t\t" + "Candidate verbs did not match any discourse cues, returning None'")
    return None,None,None

"""
 Increment/print all statistical data based on the results of the current sentence
"""
def incrementData(choice,val):

    global total,correct_noncausal,correct_causal,type_1,type_2
    total += 1
    if (val == choice or val == -1):
        if (choice == 0):
            correct_noncausal += 1
        else:
            correct_causal += 1
    else:
        if (val == 0):
            type_1 += 1
        else:
            type_2 += 1    
    if total % sentChunkNumber == 0:
        print("Sentences Evaluated: %d " % (total),end='')
        if (ParseSemEval):
            print("\n\tAccuracy: %f (%d/%d) " % ((correct_causal+correct_noncausal)/total,correct_causal+correct_noncausal,total),end='') 
            print("\n\tFalse Positives: %f (%d/%d) False Negatives: %f (%d/%d)" % (type_1/total,type_1,total,type_2/total,type_2,total),end='')
            print("\n\tTrue Positives: %f (%d/%d) True Negative: %f (%d/%d)" % (correct_causal/total,correct_causal,total,correct_noncausal/total,correct_noncausal,total))
        else:
            print("\n\tCausal Sentences: %f (%d/%d) Non-Causal Sentences: %f (%d/%d)" % (correct_causal/total,correct_causal,total,correct_noncausal/total,correct_noncausal,total))            
    
"""
   Read-In Causal Cues from provided cue_file
"""
def readInCues(cue_file):
    inFile = open(cue_file)
    for line in inFile:
        line = line.replace('\n',"").strip()
        line = line.split(' ')
        if (lemmatize_cues):
            verb = WordNetLemmatizer().lemmatize(line[0],'v')
        else:
            verb = line[0]
        if (use_threshold):
            count = 0
            synsets = wn.synsets(verb)
            if (WNVerbsOnly):
                synsets = wn.synsets(verb,pos=wn.VERB)
            for synset in synsets:
                if verb in synset.name():
                    count += 1
            if (count > threshold or count == 0):
                continue            
        if (len(line) == 1):        
            cue_dict[verb].add("")
        else:
            cue_dict[verb].add(' '.join(line[1:]))
    return cue_dict

"""
Acquires causal score of each cause-effect pair based on:
-frequency of the evaluated cause-effect pair
-frequency of the 'cause' to cause other effects (and same for the 'effect')
-number of sentences evaluated
-total number of cause-effect pairs
"""
def getCausalScore(causalNetwork, numSentences):
    effect_dict = defaultdict(int)
    for effects in causalNetwork.values():
        for effect in effects.keys():
            if effect_dict.get(effect) == None:
                effect_dict[effect] = effects[effect][0]
            else:
                effect_dict[effect] += effects[effect][0]
    
    numEvidences = sum([freq for freq in effect_dict.values()])
    alpha = 0.66
    
    for cause in causalNetwork.keys():
        n_causes = sum([effect[0] for effect in causalNetwork[cause].values()])
        p_cause = float(n_causes)/numEvidences
        for effect in causalNetwork[cause].keys():
            p_effect = float(effect_dict[effect])/numEvidences
            p_cause_effect = float(causalNetwork[cause][effect][0])/numSentences
            if ((p_cause**alpha)*p_effect != 0):
                CS_nec = p_cause_effect/((p_cause**alpha)*p_effect)
            else:
                CS_nec = 0
            if (p_cause*(p_effect**alpha) != 0):
                CS_suf = p_cause_effect/(p_cause*(p_effect**alpha))
            else:
                CS_suf = 0
            causalNetwork[cause][effect].append([CS_nec,CS_suf])
            lambs = [ 0.5, 0.7, 0.9, 1.0]
            causalNetwork[cause][effect].append(defaultdict(float))
            for lamb in lambs:
                causalScore = (CS_nec**lamb)*(CS_suf**(1-lamb))
                causalNetwork[cause][effect][2][lamb] = causalScore
    return causalNetwork
    

"""
   Functions for timing sections of code
"""
def s0():
    global start 
    start = time.time()
def s1():
    global stop
    stop = time.time()
    return delt()
def delt():
    return stop - start


if __name__ == "__main__":
    # Read in Discourse Cue List into Dictionary
    cue_dict = readInCues(cue_file)
    # Add Girju's causal list
    if (addGirju):
        inFile = dir_path + "\\Data-Files\\girju.txt"
        use_threshold = False
        girju_dict = readInCues(inFile)
        for verb in girju_dict.keys():
            for value in girju_dict[verb]:
                cue_dict[verb].add(value)
    Causal_Sentences = list()
    causalOutput.write("NEW_FILE:%s ============================================================================================ NEW_FILE:%s\n"% (sentenceFile,sentenceFile))
    causalOutput.write("Use threshold: %r\nWNVerbsOnly: %r\nthreshold: %d\nLemmatize Candidates: %r\nLemmatize Cues: %r\nDebug: %r\nUse Phrase Extraction: %r\nParse Sem Eval: %r\n\n" % (use_threshold, WNVerbsOnly, threshold, lemmatize_candidates, lemmatize_cues, debug,UsePhraseExtraction,ParseSemEval))
    
    # Tokenize and read in sentence
    readData = open(sentenceFile,'r')
    Cue_Found = None
    causalNetwork = defaultdict(dict)
    sentLst = list()
    for line in readData:
        sentLst.append(word_tokenize(line.strip().replace('\n',"")))
        if (maxSents != -1 and len(sentLst) == maxSents/2):
            break
    readData.close()
    readData = open(sentenceFile2,'r')
    for line in readData:
        sentLst.append(word_tokenize(line.strip().replace('\n',"")))
        if (maxSents != -1 and len(sentLst) >= maxSents):
            break

    print(len(sentLst))
    i_start = 0
    i_end = 0
    s0()
    while (i_end  != len(sentLst)):
        val = -1
        i_end += sentChunkNumber
        if (i_end > len(sentLst)):
            i_end = len(sentLst)
        POS_Lst = pos_tagger.tag_sents(sentLst[i_start:i_end])
        for i in range(len(POS_Lst)):
            Cue_Found,NPG1,NPG2 = analyze(sentLst[i_start+i],POS_Lst[i])
            sentence = TreebankWordDetokenizer().detokenize(sentLst[i_start+i])
            choice = 0
            if (Cue_Found != None):
                if (debug):
                    print("Sentence perceived to be CAUSAL:")
                    print("\tNoun-Phrase (Group) 1: %s" % POStoString(NPG1))
                    print("\tCue Found: %s" % Cue_Found)
                    print("\tNoun-Phrase (Group) 2: %s\n" % POStoString(NPG2))
                choice = 1
                NE1,NH1 = ParseNounHead(sentence,GrabNounHead(NPG1))
                NE2,NH2 = ParseNounHead(sentence,GrabNounHead(NPG2))
                if (debug):
                    print("Sentence: %s\nCue Found: %s\nNPG1: %s\nNPG1head: %s\nNPG2: %s\nNPG2head: %s\n" % (sentence,Cue_Found,POStoString(NPG1),NE1,POStoString(NPG2),NE2),end = '')             
                causalOutput.write("Sentence: %s\nCue Found: %s\nNPG1: %s\nNPG1head: %s\nNPG2: %s\nNPG2head: %s\n" % (sentence,Cue_Found,POStoString(NPG1),NE1,POStoString(NPG2),NE2))
                if (ranking):
                    NH1Rank = isRankOne(NH1) or NE1 != None
                    NH2Rank = isRankOne(NH2) or NE2 != None
                    if (NH1Rank and NH2Rank):
                        if (debug):
                            print("Causal Ranking: Rank One")
                        causalOutput.write("Causal Ranking:Rank One\n")
                        if (causalNetwork.get(NH1) == None):
                            causalNetwork[NH1] = defaultdict(list)
                        if (causalNetwork[NH1].get(NH2) == None):
                            causalNetwork[NH1][NH2].append(0)
                        causalNetwork[NH1][NH2][0] += 1
                    elif (NH2Rank):
                        choice = 0
                        if (debug):
                            print("Causal Ranking: Rank Two")
                        causalOutput.write("Causal Ranking:Rank Two\n")
                    else:
                        choice = 0
                        if (debug):
                            print("Causal Ranking: Unranked")
                        causalOutput.write("Causal Ranking:Unranked\n")
                    causalOutput.write('\n')
                else:
                    if (causalNetwork.get(NH1) == None):
                        causalNetwork[NH1] = defaultdict(list)
                    if (causalNetwork[NH1].get(NH2) == None):
                        causalNetwork[NH1][NH2].append(0)
                    causalNetwork[NH1][NH2][0] += 1                    
                
            elif (debug):
                print("Sentence perceived to be NONCAUSAL")
            incrementData(choice,val)            
        i_start = i_end
    print("It took",s1(),"seconds to process", len(sentLst),"sentences")
    causalNetwork = getCausalScore(causalNetwork,len(sentLst))
    networkFile = open("causalNetwork" + parameter_string + ".txt",'w')
    networkFile.write(pformat(causalNetwork))
    networkFile.write("\n\n=====================================================================\n\n")
    causal_lst = list()
    for cause in causalNetwork.keys():
        for effect in causalNetwork[cause].keys():
            causal_lst.append({'a: key' : cause + ' , ' + effect, 'b: data' : causalNetwork[cause][effect] })
    networkFile.write(pformat(sorted(causal_lst,key = lambda pair : pair['b: data'][2][1.0],reverse=True)))
    networkFile.close()
    