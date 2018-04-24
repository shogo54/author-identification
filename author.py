import numpy as np
import re
import nltk
import csv
import operator
import sklearn
from sklearn.naive_bayes import MultinomialNB

sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
word_tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')


def predictAuthors(training_fvs, labels, test_fvs):
    """
    Predict author using Naive bayes Classifier

    :param training_fvs: list of training feature vectors
    :param labels: list of actual labels mapped onto training_fvs
    :param test_fvs: list of test feature vectors
    :return: list of predicted labels mapped onto tes_fvs
    """
    clf = MultinomialNB()
    clf.fit(training_fvs, labels)
    return clf.predict(test_fvs)


def train(filename, supervised=False):
    """
    read the file into data
    :param filename:
    :param supervised:
    :return data:
    """
    p = re.compile(r'[a-zA-Z0-9_:;,."?\' ]+')
    data = []
    all_text = {}
    infile = csv.DictReader(open(filename), delimiter=',', quotechar='"')
    for row in infile:
        text_id = row['id']
        text = row['text']
        author = row['author'] if supervised else None

        # remove special characters
        new_text = ''
        for word in text:
            for letter in word:
                reg = p.match(letter)
                if reg is not None:
                    new_text += reg.group()

        data.append((text_id, new_text, author))
        if supervised:
            if author not in all_text.keys():
                all_text[author] = ''
            else:
                sentences = sentence_tokenizer.tokenize(new_text)
                all_text[author] += ' '.join(sentences) + ' '
                # print("{} {} {}".format(text_id, text, author))
    if supervised:
        return data, all_text
    else:
        return data


def lexicalFeatures(training_data, test_data):
    # create lexical and punctuational feature vectors
    print 'processing lexical and punctuation features...'
    lexical_train, punct_train = fvsLexical(training_data)
    lexical_test, punct_test = fvsLexical(test_data)
    return (lexical_train[0], lexical_train[1], lexical_test[0]), (punct_train[0], punct_train[1], punct_test[0])


def fvsLexical(data):
    """
    Compute feature vectors for word and punctuation features
    :param data:
    :return:
    """
    fvs_lexical = np.zeros((len(data), 3))
    fvs_punct = np.zeros((len(data), 3))
    labels_lexical = [''] * len(data)
    labels_punct = [''] * len(data)
    for e, (id, text, author) in enumerate(data):
        # print id, text
        tokens = nltk.word_tokenize(text.lower())
        words = word_tokenizer.tokenize(text.lower())
        sentences = sentence_tokenizer.tokenize(text)
        vocab = set(words)
        words_per_sentence = np.array([len(word_tokenizer.tokenize(s)) for s in sentences])

        # update fvs_lexical and labels_lexical
        # average number of words per sentence
        fvs_lexical[e, 0] = words_per_sentence.mean()
        # sentence length variation
        fvs_lexical[e, 1] = words_per_sentence.std()
        # lexical diversity
        fvs_lexical[e, 2] = len(vocab) / float(len(words))
        # put author label
        labels_lexical[e] = author

        # update fvs_punct and labels_punct
        # commas per sentence
        fvs_punct[e, 0] = tokens.count(',') / float(len(sentences))
        # semicolons per sentence
        fvs_punct[e, 1] = tokens.count(';') / float(len(sentences))
        # colons per sentence
        fvs_punct[e, 2] = tokens.count(':') / float(len(sentences))
        # put author label
        labels_punct[e] = author

    return (fvs_lexical, labels_lexical), (fvs_punct, labels_punct)


def syntacticFeatures(all_text_dict, test_data):
    print 'processing syntactic features...'
    # make all_text_dict into list of tuple (id, text, author) so that it can be treated as a data type here
    all_data = list((None, sentences, author) for author, sentences in all_text_dict.items())
    train_fvs, train_labels = fvsSyntax(all_data)
    test_fvs, test_labels = fvsSyntax(test_data)

    return train_fvs, train_labels, test_fvs


def fvsSyntax(data):
    """
    Extract feature vector for part of speech frequencies
    """

    def token_to_pos(text):
        tokens = nltk.word_tokenize(text)
        return [p[1] for p in nltk.pos_tag(tokens)]

    texts_pos = [token_to_pos(text) for id, text, author in data]
    pos_list = ['NN', 'NNP', 'DT', 'IN', 'JJ', 'NNS']
    fvs_syntax = np.array([[text.count(pos) for pos in pos_list]
                           for text in texts_pos]).astype(np.float64)
    labels_syntax = [author for id, text, author in data]

    return fvs_syntax, labels_syntax


def bagOfWordsFeatures(all_text_dict, test_data):
    print 'processing bag of words features...'
    # create all of the word set
    wordset = set()
    # make all_text_dict into list of tuple (id, text, author) so that it can be treated as a data type here
    all_data = []
    for author, sentences in all_text_dict.items():
        words = word_tokenizer.tokenize(sentences.lower())
        for word in words:
            wordset.add(word)
        all_data.append((None, sentences, author))

    # Return a dictionary that maps each word from wordset to a unique index starting at 0
    # and going up to N-1, where N is the len(wordset).
    windex = {}
    sort_words = sorted(list(wordset))
    for i in range(len(sort_words)):
        word = sort_words[i]
        windex[word] = i

    # Compute the bag of words in the whole text by each author
    train_fvs, train_labels = fvsBagOfWords(all_data, windex)
    test_fvs, test_labels = fvsBagOfWords(test_data, windex)

    return train_fvs, train_labels, test_fvs


def fvsBagOfWords(data, windex):
    fvs_bow = np.zeros((len(data), len(windex)))
    labels_bow = [''] * len(data)
    for e, (id, text, author) in enumerate(data):
        all_tokens = nltk.word_tokenize(text.lower())
        fdist = nltk.FreqDist(all_tokens)
        sorted_fdist = reversed(sorted(fdist.items(), key=operator.itemgetter(1)))
        for (word, count) in sorted_fdist:
            if word not in windex:
                continue
            index = windex[word]
            fvs_bow[e, index] = count / float(len(all_tokens))

        labels_bow[e] = author
    return fvs_bow, labels_bow


def probability(training_data):
    total_count_map = {}
    for id, text, author in training_data:
        if author not in total_count_map.keys():
            total_count_map[author] = 0
        total_count_map[author] += 1

    for k, v in total_count_map.items():
        total_count_map[k] = v * 100 / float(len(training_data))

    count_list = reversed(sorted(total_count_map.items(), key=operator.itemgetter(1)))
    result_list = []
    print '\n', 'random probability'
    for ele in count_list:
        print ele
        result_list.append(ele)

    return result_list


if __name__ == '__main__':
    # part1: process file data
    print 'training the machine on data...'
    training_data, all_text_dict = train('train.csv', True)
    test_data = train('test.csv')

    # part2: put all feature vectors and labels into a list
    feature_sets = list(lexicalFeatures(training_data, test_data))
    feature_sets.append(bagOfWordsFeatures(all_text_dict, test_data))
    feature_sets.append(syntacticFeatures(all_text_dict, test_data))

    # part3: create classification
    classifications = [predictAuthors(fvs, labels, test) for fvs, labels, test in feature_sets]

    # part4: evaluate the probability of random choice
    count_list = probability(training_data)

    # part5: print the result table
    print '\n', 'result table'
    final_answer = {}
    for results in classifications:
        print ' '.join(results)
        for test_count, result in enumerate(results, 0):
            if test_count not in final_answer:
                final_answer[test_count] = []
            final_answer[test_count].append(result)

    # part6: process the results and print the final result
    print '\n', 'final result'

    # create the list of test id
    test_id_list = [''] * len(test_data)
    for e, (id, text, author) in enumerate(test_data):
        test_id_list[e] = id

    # process the results
    for k, v in final_answer.items():
        count_map = {'EAP': 0, 'MWS': 0, 'HPL': 0}
        for name in v:
            count_map[name] += 1
        max_val = []
        max_count = 0
        for name, num in count_map.items():
            if max_count < num:
                max_val = [name]
                max_count = num
            elif max_count == num:
                max_val.append(name)

        if len(max_val) > 1:
            for ele in max_val:
                if ele == count_list[0][0]:
                    max_val = count_list[0][0]
                elif ele == count_list[1][0]:
                    max_val = count_list[1][0]
                else:
                    max_val = count_list[2][0]
        else:
            max_val = max_val[0]

        final_answer[k] = max_val

    for i in range(len(test_id_list)):
        print '{}\t{}'.format(test_id_list[i], final_answer[i])
