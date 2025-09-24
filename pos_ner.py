
import nltk
import re
from nltk import sent_tokenize, word_tokenize, pos_tag, ne_chunk
from collections import defaultdict

# Download all necessary NLTK resources
nltk.download('all')

# Load the article
with open("news article-1.txt", "r", encoding="utf-8") as f:
    text = f.read()

# 1. Detect Sentences
sentences = sent_tokenize(text)
for a, b in enumerate(sentences, 1):
    print(f"{a}. {b}")

# POS tagging for each sentence
for a, b  in enumerate(sentences , 1):
    words = nltk.word_tokenize(b)
    pos_tags = nltk.pos_tag(words)
    print(f"\nSentence : {a}")
    print("POS Tags:", pos_tags)

# 3. Named Entity Recognition (NER) 
entities = defaultdict(lambda: defaultdict(list))

for a, b in enumerate(sentences, 1):
    words = word_tokenize(b)
    pos_tags = pos_tag(words)
    ne_tree = ne_chunk(pos_tags)
    for subtree in ne_tree:
        if hasattr(subtree, 'label'):
            entity_type = subtree.label()
            if entity_type in ['PERSON', 'GPE', 'ORGANIZATION']:
                entity = ' '.join([word for word, tag in subtree.leaves()])
                entities[f"Sentence {a}"][entity_type].append(entity)

for sent, ent_dict in entities.items():
    print(f"\n{sent}:")
    for ent_type, ent_list in ent_dict.items():
        print(f"  {ent_type}: {', '.join(ent_list)}")