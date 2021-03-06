# Author Identification

This is a machine learning program to perform authorship identification on sample sentences from three horror authors HP Lovecraft (HPL), Mary Wollstonecraft Shelley (MWS), and Edgar Allen Poe (EAP).

The project was created for a sample contest on [Kaggle](https://www.kaggle.com/c/spooky-author-identification)



<!-- implementation details -->
## Implementation Details

The classifier is based on Naive Bayes, and will feed the training data and predict each of the unknown sentences.

The project includes the training and testing data files. 
The training data is labeled with the author of each sentence, while the test data is not labeled.

The followings are the feature vectors that the program uses for prediction.
1. bag of words (I put all of the training texts into lists labeled with author and create bag of words based on it. Then read each test text and classify it with bag of words)
2. parts of speech (syntax features)
3. lexical features (average number of words per a sentence, sentence length variation, and lexical diversity)
4. punctuation features  (commas, semicolons, and colons per a sentence)



<!-- Prerequisites -->
## Prerequisites

To run the code, make sure that you install all packages that the project is using. The project is using the following packages: 
- [numpy][numpy-url]
- [nltk][nltk-url]
- [sklearn][sklearn-url]

To ensure that you install the packages above, run the following command on your console: 

```python -m pip install --user numpy nltk sklearn```



<!-- LICENSE -->
## License

Distributed under the MIT License. See [`LICENSE`][license-url] for more information.



<!-- CONTACT -->
## Contact

Shogo Akiyama - shogo.a.0504@gmail.com

Project Link: [https://github.com/shogo54/author-identification][project-url]



<!-- Acknowledgements -->
## Acknowledgements

The implementation is inspired from the following article:<br/>
- http://www.aicbt.com/authorship-attribution



<!-- Future References -->
## Future References

in the future, I can apply a neural network based approach to this project. 
The artilces bellow might be useful:<br/>
- https://arxiv.org/pdf/1506.04891.pdf
- https://cs224d.stanford.edu/reports/MackeStephen.pdf
- http://www.aclweb.org/anthology/E17-2106



<!-- MARKDOWN LINKS & IMAGES -->
[project-url]: https://github.com/shogo54/author-identification/
[license-url]: LICENSE
[numpy-url]: https://numpy.org/
[sklearn-url]: https://scikit-learn.org/
[nltk-url]: https://www.nltk.org/
