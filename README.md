# Author Identification

This project is to perform authorship identification on sample sentences from three horror authors authors HP Lovecraft (HPL), Mary Wollstonecraft Shelley (MWS), and Edgar Allen Poe (EAP).

The project was created for a sample contest on [Kaggle](https://www.kaggle.com/c/spooky-author-identification)

The project includes the training and testing data files. 
The training data is labeled with the author of each sentence, while the test data is not labeled.

The classifier is based on Naive Bayes, and will feed the training data and predict each of the unknown sentences.


## Prerequisites

To run the code, make sure that you install all packages that the project is using. The project is using the following packages: 
- numpy
- nltk
- sklearn

To ensure that you install the packages above, run the following command on your console: 

```python -m pip install --user numpy nltk sklearn```


## Acknowledgements

The implementation is inspired from the following article:<br/>
- http://www.aicbt.com/authorship-attribution

## Future References

in the future, I can apply a neural network based approach to this project. 
The artilces bellow might be useful:<br/>
- https://arxiv.org/pdf/1506.04891.pdf
- https://cs224d.stanford.edu/reports/MackeStephen.pdf
- http://www.aclweb.org/anthology/E17-2106
