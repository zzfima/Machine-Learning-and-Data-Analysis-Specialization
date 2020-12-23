# Cosine distance is often used in text analysis to measure the similarity between texts.
# Smaller distance means more similarity

import re
import numpy as np
from scipy.spatial.distance import cosine

print('*** Cosine distance is often used in text analysis to measure the similarity between texts ***')
print('*** Smaller distance means more similarity ***')

# add sentences to list as low-case
sentences_raw = []
with open('sentences.txt') as rf:
    for line in rf:
        sentences_raw.append(line.lower().strip())

# remove from sentences all, except words
sentences_filtered = []
for s in sentences_raw:
    sentences_filtered.append(list(filter(None, re.split('[^a-z]', s))))

# create unique list of words
words_set = set()
for r in sentences_filtered:
    for w in r:
        words_set.add(w)
words_set_list = list(words_set)

# create matrix, where row is sentence number, column is word. Filled by zeroes
rows_count = len(sentences_filtered)
words_count = len(words_set_list)
matrix = np.zeros((rows_count, words_count))

# fill matrix in a way: count word x in sentence y
for row_number in range(rows_count):
    for word_number in range(words_count):
        cnt = sentences_filtered[row_number].count(words_set_list[word_number])
        matrix[row_number, word_number] = cnt

# find out cosine distance between first sentence and others
# cosine distance between sentence and itself = 0
cosine_distance_list = []
for row in matrix[1:]:
    cosine_distance_list.append(cosine(matrix[0], row))
print('cosine distances between first sentence and other: ', cosine_distance_list)
