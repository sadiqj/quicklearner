quicklearner
============

quicklearner is a Java library for very simple creation of classifiers. It's not designed for performance but rather ease of getting started.

It's currently experimental code and largely lacks good documentation and tests. The interface is also in a state of flux and comments are welcome!

How to use it
-------------

You create a Learner instance as follows:

    LearnerBuilder builder = Learner.builder();

    builder.addExample(.. label .., .. Map<String,Double> or Set<String> of features ..);
    .. more examples ..

    Learner learner = builder.build();

    System.out.println("Accuracy was: " + learner.getAccuracy());

    String label = learner.classify(.. Map<String,Double> or Set<String> of features ..);

Once built, a Learner is immutable and is thread-safe. The features used for training can be sets of Strings (used to indicate whether a feature is present or not.. this works well for textual classification) or a map of String-Double entries.

The Learner instance also has another method getLabelProbabilities which returns all of the available labels and their probabilities. In addition, during the training phase, a cross validation is carried out on the Learner and the resulting accuracy is avaiable with the getAccuracy getter on the Learner instance.

The underlying learning algorithm is regularised logistic regression using batch gradient descent with feature normalisation. I may add an SVM implementation in the near future, though i've tried to keep the interface generic enough that switching things out shouldn't be a problem.

Persistence
-----------

You can serialise the Learners to a byte array, which can then subsequently be used to recreate a Learner as follows:

    byte[] bytes = learner.serialise();

    .. store to disk/database ..

    .. at some later time ..

    Learner learner = Learner.load(bytes);


TODO
----

There are a few tests that generate linearly separable test sets for two-class and multi-class classifiers as well as test serialisation/reloading but I plan on adding a suite of performance tests to watch for regressions in classification performance. 
