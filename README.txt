Currently a sample contest on Kaggle is to perform authorship identification on sample sentences from horror authors HP Lovecraft (HPL), Mary Wollstonecraft Shelley (MWS), and Edgar Allen Poe (EAP).

https://www.kaggle.com/c/spooky-author-identification

I've included the training and testing data files for you, along with some very simple Python code to get started parsing.
The training data is labeled with the author of each sentence, while the test data is not labeled.

Classify each of the unknown sentences using at least 4 techniques, and report on your results.
You can try Naive Bayes like we are doing for automated essay scoring, but try 3 other techniques.

This article gives you an idea of some of the features that you could try for classification:

http://www.aicbt.com/authorship-attribution/

So for example, you could try lexical variety, punctuation, and part of speech (POS) tagging using nltk.
You would need to adjust this code.


If you are feeling ambitious, try a neural network based approach.
Here are a bunch of papers you might use for inspiration (haven't read them yet but they look reasonable):

https://arxiv.org/pdf/1506.04891.pdf
https://cs224d.stanford.edu/reports/MackeStephen.pdf
http://www.aclweb.org/anthology/E17-2106