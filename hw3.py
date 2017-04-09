#!/usr/bin/python3
import nltk
import wikipedia

def readData():
    text = ""
    with open('text.txt', 'r') as txt:
        text = txt.read()
    return text

def extract_entit(neChunked):
    data = {}
    for entity in [x for x in neChunked if isinstance(x, nltk.tree.Tree)]:
        text = " ".join([word for word, tag in entity.leaves()])
        ent = entity.label()
        data[text] = ent
    return list(data.keys())


def get_description_from_wiki(entity):
    try:
        sentence = wikipedia.summary(entity, sentences=1)
    except wikipedia.exceptions.DisambiguationError as e:
        sentence = wikipedia.summary(e.options[0], sentences=1)
    except wikipedia.exceptions.PageError:
        return 'Thing'
    from re import sub
    sentence = sub(r'\(.+\)', '', sentence)
    text_pos = nltk.pos_tag(nltk.word_tokenize(sentence))
    custPatt = nltk.RegexpParser('NP: {(is|was|were|are)?<JJ>?<NN|NNS>+}')
    return ', '.join(extract_entit(custPatt.parse(text_pos)))


TOP_X_ENTITIES = 10
text = readData()
print("Input text has: ",len(nltk.sent_tokenize(text)), "sentences")

text_pos = nltk.pos_tag(nltk.word_tokenize(text))
neChunked = nltk.ne_chunk(text_pos, binary=False)
nltk_entities = extract_entit(neChunked)[0:TOP_X_ENTITIES]

cust_patt = nltk.RegexpParser('NP: {<PRP|DT>?<JJ>*<NN|NNS>}')
custom_entities = extract_entit(cust_patt.parse(text_pos))[0:TOP_X_ENTITIES]

nltk_processed = {}
for i in range(len(nltk_entities)):
    nltk_processed[nltk_entities[i]] = get_description_from_wiki(nltk_entities[i])

custom_processed = {}
for i in range(len(custom_entities)):
    custom_processed[custom_entities[i]] = get_description_from_wiki(custom_entities[i])


print('###################################################')
print('nltk-based classification :')
print('###################################################')
val_key = extract_entit(nltk.ne_chunk(text_pos))
for x in range (0,TOP_X_ENTITIES):
    print(list(val_key.keys())[x],": ",list(val_key.values())[x])

print('###################################################')
print('wikipedia-based classification using nltk entities:')
print('###################################################')
val_key = list(nltk_processed.keys())[0:TOP_X_ENTITIES]
for i in val_key:
    print(i, ": ", nltk_processed[i])

print('###################################################')
print('wikipedia-based classification using custom patterns:')
print('###################################################')
val_key = list(custom_processed.keys())[0:TOP_X_ENTITIES]
for i in val_key:
    print(i,": ", custom_processed[i])





