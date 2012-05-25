package com.toao.quicklearner;

import java.util.List;
import java.util.Map;
import java.util.Set;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;

public class LearnerBuilder
{
	private static Logger sLogger = LoggerFactory.getLogger(LearnerBuilder.class);

	private List<Example> examples = Lists.newArrayList();

	private int cvFolds = 10;

	private double learningRate = 0.01;

	private double regulariser = 0.4;

	protected LearnerBuilder addExample(String label, Map<String, Double> features)
	{
		examples.add(new Example(label, features));
		return this;
	}

	protected LearnerBuilder addExample(String label, Set<String> features)
	{
		Map<String, Double> newFeatures = Maps.newHashMap();

		for (String feature : features)
		{
			newFeatures.put(feature, 1.0);
		}

		examples.add(new Example(label, newFeatures));
		return this;
	}

	public LearnerBuilder setCVFolds(int cvFolds)
	{
		this.cvFolds = cvFolds;
		return this;
	}

	public LearnerBuilder setLearningRate(double learningRate)
	{
		this.learningRate = learningRate;
		return this;
	}

	public LearnerBuilder setRegularisation(double regulariser)
	{
		this.regulariser = regulariser;
		return this;
	}

	public Learner build()
	{
		// Find all the labels

		Set<String> labelSet = Sets.newHashSet();

		for (Example example : examples)
		{
			labelSet.add(example.getLabel());
		}

		List<String> labels = Lists.newArrayList(labelSet);

		// First do cross validation

		int correct = 0;
		int tries = 0;

		for (int c = 0; c < cvFolds; c++)
		{
			List<Example> trainingExamples = Lists.newArrayList();
			List<Example> testExamples = Lists.newArrayList();

			for (int i = 0; i < examples.size(); i++)
			{
				if (i % cvFolds == c)
				{
					testExamples.add(examples.get(i));
				}
				else
				{
					trainingExamples.add(examples.get(i));
				}
			}

			Learner learner = train(regulariser, learningRate, labels, trainingExamples, 0.0);

			for (int i = 0; i < testExamples.size(); i++)
			{
				tries++;
				Example example = testExamples.get(i);

				String probLabel = learner.classify(example.getFeatures());

				if (probLabel == example.getLabel())
				{
					correct++;
				}
			}
		}

		double accuracy = ((double) correct) / tries;

		sLogger.debug("build - final training pass.. ");

		Learner learner = train(regulariser, learningRate, labels, examples, accuracy);

		sLogger.debug("build - tries: {}, correct: {}, accuracy: {}", new Object[]
		{ tries, correct, ((10000 * correct) / tries) / 100.0 });

		return learner;
	}

	private Learner train(double regulariser, double learningRate, List<String> labels, List<Example> trainingExamples, double accuracy)
	{
		InternalTrainer trainer = new InternalTrainer();
		List<InternalLearner> internalLearners = Lists.newArrayList();

		if (labels.size() == 2)
		{
			InternalLearner learn = trainer.learn(regulariser, learningRate, labels.get(1), trainingExamples);

			internalLearners.add(learn);
		}
		else
		{
			for (int c = 0; c < labels.size(); c++)
			{
				InternalLearner internalLearner = trainer.learn(regulariser, learningRate, labels.get(c), trainingExamples);

				internalLearners.add(internalLearner);
			}
		}

		Learner learner = new Learner(internalLearners, labels, accuracy);

		return learner;
	}
}
