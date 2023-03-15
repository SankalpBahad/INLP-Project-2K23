import pprint
import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.corpus import stopwords
stopwords_en = set(stopwords.words('english'))


def tokenize(document, word):
    tokenizer = nltk.RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(document)
    tokens = [token for token in tokens if token not in stopwords_en and token.isalpha()]
    tokens = [token for token in tokens if token != word]
    return set(tokens)



def get_sense(Test_word, Text):
    Text = Text.lower()
    Test_word = Test_word.lower()

    # tokenize the Text and get synsets for the Test_word
    Text_tokens = tokenize(Text, Test_word)
    synsets = wordnet.synsets(Test_word)

    # initialize data structures
    # weights = [0] * len(synsets)
    weights=[]
    for i in range(len(synsets)):
        weights.append(0)
    num_synsets = len(synsets)
    synsets_num_per_word = {}

    # count synsets_num_per_word for each Text token
    for Text_token in Text_tokens:
        synsets_num_per_word[Text_token] = 1

        for Word_sense in synsets:
            if Text_token in Word_sense.definition():
                synsets_num_per_word[Text_token] += num_synsets
                continue

            for example in Word_sense.examples():
                if Text_token in example:
                    synsets_num_per_word[Text_token] += num_synsets
                    break

    # calculate weights for each synset
    for index, Word_sense in enumerate(synsets):
        comparison = set()
        for example in Word_sense.examples():
            for token in tokenize(example, Test_word):
                comparison.add(token)
        for token in tokenize(Word_sense.definition(), Test_word):
            comparison.add(token)
        for token in Text_tokens:
            if token in comparison:
                weights[index] += np.log(synsets_num_per_word[token] / num_synsets)

    # find synset with highest weight
    max_weight = max(weights)
    index = weights.index(max_weight)
    return synsets[index]




Text = input('Text containing the word:\t')
Test_word = input('Test_word:\t')
Word_sense = get_sense(Test_word, Text)
print('Definition of the word given here (after disambiguation) is:', Word_sense.definition())
