This project is to perform authorship identification on sample sentences from three horror authors authors HP Lovecraft (HPL), Mary Wollstonecraft Shelley (MWS), and Edgar Allen Poe (EAP).

The project was created for a sample contest on Kaggle:
https://www.kaggle.com/c/spooky-author-identification

The project includes the training and testing data files. 
The training data is labeled with the author of each sentence, while the test data is not labeled.

The classifier is based on Naive Bayes, and will feed the training data and predict each of the unknown sentences.

The implementation is inspired from the following article:
http://www.aicbt.com/authorship-attribution

in the future, try a neural network based approach. 
The artilces bellow might be useful:
https://arxiv.org/pdf/1506.04891.pdf
https://cs224d.stanford.edu/reports/MackeStephen.pdf
http://www.aclweb.org/anthology/E17-2106
