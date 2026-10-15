from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
import nltk

import pandas as pd
import html2text
import re

from spacy.lang.en import STOP_WORDS
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.corpus import stopwords

nltk.download(['stopwords', 'wordnet'])
stops = list(set(ENGLISH_STOP_WORDS)) + list(set(stopwords.words('english') +  list(set(STOP_WORDS)) + ["http"]))

class EmotionVectorMaker:
    def __init__(self, use_intensity = True):
        if use_intensity:
            fileEmotion = "VectorMakers/NRC-Emotion-Intensity-Lexicon-v1.txt"
        else:
            fileEmotion = "VectorMakers/NRC-emotion-lexicon-wordlevel-alphabetized-v0.92.txt"

        table = pd.read_csv(fileEmotion,  names=["word", "emotion", "itensity"], sep='\t')

        #create the dictionary with the word/emotion/score
        self.emotion_dic = dict()
        self.lmtzr = WordNetLemmatizer()
        for index, row in table.iterrows():
            #add first as it is given in the lexicon
            print(f"word:{row['word']}, emotion:{row['emotion']}")
            word_str = str(row['word'])
            temp_key = word_str + '#' + row['emotion']
            self.emotion_dic[temp_key] = row['itensity']

            #add in the normal noun form
            temp_key_n = self.lmtzr.lemmatize(word_str) + '#' + row['emotion']
            self.emotion_dic[temp_key_n] = row['itensity']
            
            #add in the normal verb form
            temp_key_v = self.lmtzr.lemmatize(word_str, 'v') + '#' + row['emotion']
            self.emotion_dic[temp_key_v] = row['itensity']  
        h = html2text.HTML2Text()
        h.ignore_links = True

    def getEmotionItensity(self, word, emotion):
        key = word + "#" + emotion
        try:
            return self.emotion_dic[key]
        except:
            return 0.0
    
    def isWordInEmotionFile(self, word):
        # Slightly faster implementation
        for key in self.emotion_dic.keys():
            if key.startswith(word + "#"):
                return True
        return False
    
    def isStopWord(self, word):
        if word in stops:
            return True
        else:
            return False
    
    def calculateEmotion(self, emotions, word):
        emotions["Anger"] += self.getEmotionItensity(word, "anger")
        emotions["Anticipation"] += self.getEmotionItensity(word, "anticipation")
        emotions["Disgust"] += self.getEmotionItensity(word, "disgust")
        emotions["Fear"] += self.getEmotionItensity(word, "fear")
        emotions["Joy"] += self.getEmotionItensity(word, "joy")
        emotions["Sadness"] += self.getEmotionItensity(word, "sadness")
        emotions["Surprise"] += self.getEmotionItensity(word, "surprise")
        emotions["Trust"] += self.getEmotionItensity(word, "trust")

    def getEmotionVecFor_Word(self, word):
        emotions = {"Anger": 0.0,
            "Anticipation": 0.0,
            "Disgust": 0.0,
            "Fear": 0.0,
            "Joy": 0.0,
            "Sadness": 0.0,
            "Surprise": 0.0,
            "Trust": 0.0,}
        emotions["Anger"] += self.getEmotionItensity(word, "anger")
        emotions["Anticipation"] += self.getEmotionItensity(word, "anticipation")
        emotions["Disgust"] += self.getEmotionItensity(word, "disgust")
        emotions["Fear"] += self.getEmotionItensity(word, "fear")
        emotions["Joy"] += self.getEmotionItensity(word, "joy")
        emotions["Sadness"] += self.getEmotionItensity(word, "sadness")
        emotions["Surprise"] += self.getEmotionItensity(word, "surprise")
        emotions["Trust"] += self.getEmotionItensity(word, "trust")
        return emotions

    def getEmotionVectorList(self, text, useSynset = True):
        #parse the description
        str = re.sub("[^a-zA-Z]+", " ", text) # replace all non-letters with a space
        pat = re.compile(r'[^a-zA-Z ]+')
        str = re.sub(pat, '', str).lower() #  convert to lowercase

        splits = str.split()

        emtion_vec_list = []
        
        #iterate over words array
        for split in splits:
            if not self.isStopWord(split):
                #first check if the word appears as it does in the text
                if self.isWordInEmotionFile(split): 
                    emtion_vec_list.append((split, self.getEmotionVecFor_Word(split)))
                    
                # check the word in noun form (bats -> bat)
                elif self.isWordInEmotionFile(self.lmtzr.lemmatize(split)):
                    emtion_vec_list.append((split, self.getEmotionVecFor_Word(self.lmtzr.lemmatize(split))))
                    
                # check the word in verb form (ran/running -> run)
                elif self.isWordInEmotionFile(self.lmtzr.lemmatize(split, 'v')):
                    emtion_vec_list.append((split, self.getEmotionVecFor_Word(self.lmtzr.lemmatize(split, 'v'))))

                # check synonyms of this word
                elif useSynset and wordnet.synsets(split) is not None:
                    found_syn = False
                    for syn in wordnet.synsets(split)[0:1]:
                        for l in syn.lemmas():
                            if self.isWordInEmotionFile(l.name()):
                                emtion_vec_list.append((split, self.getEmotionVecFor_Word(l.name())))
                                found_syn = True
                                break
                        if found_syn:
                            break

        return emtion_vec_list

    def getEmotionVector(self, text, removeObj = False, useSynset = True):
        #create the initial emotions
        emotions = {"Anger": 0.0,
                    "Anticipation": 0.0,
                    "Disgust": 0.0,
                    "Fear": 0.0,
                    "Joy": 0.0,
                    "Sadness": 0.0,
                    "Surprise": 0.0,
                    "Trust": 0.0,
                    "Objective": 0.0}
        #parse the description
        str = re.sub("[^a-zA-Z]+", " ", text) # replace all non-letters with a space
        pat = re.compile(r'[^a-zA-Z ]+')
        str = re.sub(pat, '', str).lower() #  convert to lowercase

        #split string
        splits = str.split()
        
        #iterate over words array
        for split in splits:
            if not self.isStopWord(split):
                #first check if the word appears as it does in the text
                if self.isWordInEmotionFile(split): 
                    self.calculateEmotion(emotions, split)
                    
                # check the word in noun form (bats -> bat)
                elif self.isWordInEmotionFile(self.lmtzr.lemmatize(split)):
                    self.calculateEmotion(emotions, self.lmtzr.lemmatize(split))
                    
                # check the word in verb form (ran/running -> run)
                elif self.isWordInEmotionFile(self.lmtzr.lemmatize(split, 'v')):
                    self.calculateEmotion(emotions, self.lmtzr.lemmatize(split, 'v'))  
                    
                # check synonyms of this word
                elif useSynset and wordnet.synsets(split) is not None:
                    # only check the first two "senses" of a word, so we don't stray too far from its intended meaning
                    # for syn in wordnet.synsets(split)[0:1]:
                    #     for l in syn.lemmas():
                    #         if isWordInEmotionFile(l.name()):
                    #             calculateEmotion(emotions, l.name())
                    #             continue
                                
                    # # none of the synonyms matched something in the file
                    # emotions["Objective"] += 1
                    found_syn = False
                    for syn in wordnet.synsets(split)[0:1]:
                        for l in syn.lemmas():
                            if self.isWordInEmotionFile(l.name()):
                                self.calculateEmotion(emotions, l.name())
                                found_syn = True
                                break
                        if found_syn:
                            break

                    if not found_syn:
                        emotions["Objective"] += 1
                    
                else:
                    # not found in the emotion file, assign a score to Objective instead
                    emotions["Objective"] += 1

        # remove the Objective category if requested
        if removeObj:
            del emotions['Objective']
            
        total = sum(emotions.values())
        for key in sorted(emotions.keys()):
            try:
                # normalize the emotion vector
                emotions[key] = (1.0 / total) * emotions[key]
            except:
                emotions[key] = 0

        return emotions