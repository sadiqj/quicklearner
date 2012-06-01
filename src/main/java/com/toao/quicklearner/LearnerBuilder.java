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

	private final int cvFolds = 10;

	private final double learningRate = 0.1;

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

	public Learner build()
	{
		if( examples.size() < 1 )
		{
			throw new RuntimeException("There must be a least one training example provided.");
		}
		
		// Find all the labels

		Set<String> labelSet = Sets.newHashSet();

		for (Example example : examples)
		{
			labelSet.add(example.getLabel());
		}

		List<String> labels = Lists.newArrayList(labelSet);

		// We will split off a validation dataset of about 20% of the examples
		List<Example> validation = Lists.newArrayList();
		List<Example> trainingExamples = Lists.newArrayList();
		
		for( int c = 0; c < examples.size() ; c++ )
		{
			if( c % 5 == 0 )
			{
				validation.add( examples.get(c) );
			}
			else
			{
				trainingExamples.add( examples.get(c) );
			}
		}
		
		// Now we try to find the best regularisation parameter
		double regulariser = 0.0;
		double bestCost = Double.MAX_VALUE;
		
		for( int p = -6 ; p < 6 ; p++ )
		{
			double tmpReg = Math.pow(-2.0, p);
			
			double totalCost = 0.0;
			
			Learner learner = train(regulariser, learningRate, labels, trainingExamples, 0.0);
			
			double examples = validation.size();
			
			for (int i = 0; i < validation.size(); i++)
			{
				Example example = validation.get(i);

				Map<String, Double> labelProbs = learner.getLabelProbabilities(example.getFeatures());
				
				for( String label : labelProbs.keySet() )
				{
					if( label.equals(example.getLabel()) )
					{
						totalCost -= Math.log(labelProbs.get(label)) / (4 * examples);
					}
					else
					{
						totalCost -= Math.log(1.0 - labelProbs.get(label)) / (4 * examples);
					}
				}
			}
			
			sLogger.debug("build - total validation cost: {} for regularisation parameter: {}", totalCost, regulariser);
			
			if( totalCost < bestCost )
			{
				bestCost = totalCost;
				regulariser = tmpReg;
			}
		}
		
		sLogger.debug("build - best regularisation parameter was: {}", regulariser);
		
		// Now do cross validation to estimate the learner's accuracy

		int correct = 0;
		int tries = 0;

		for (int c = 0; c < cvFolds; c++)
		{
			List<Example> cvTrainingExamples = Lists.newArrayList();
			List<Example> testExamples = Lists.newArrayList();

			for (int i = 0; i < trainingExamples.size(); i++)
			{
				if (i % cvFolds == c)
				{
					testExamples.add(trainingExamples.get(i));
				}
				else
				{
					cvTrainingExamples.add(trainingExamples.get(i));
				}
			}

			Learner learner = train(regulariser, learningRate, labels, cvTrainingExamples, 0.0);

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
		// We only do Logistic at the moment
		LogisticTrainer trainer = new LogisticTrainer();
		List<LogisticInternalLearner> internalLearners = Lists.newArrayList();

		if (labels.size() == 2)
		{
			LogisticInternalLearner learn = trainer.learn(regulariser, learningRate, labels.get(1), trainingExamples);

			internalLearners.add(learn);
		}
		else
		{
			for (int c = 0; c < labels.size(); c++)
			{
				LogisticInternalLearner internalLearner = trainer.learn(regulariser, learningRate, labels.get(c), trainingExamples);

				internalLearners.add(internalLearner);
			}
		}

		Learner learner = new LogisticLearner(internalLearners, labels, accuracy);

		return learner;
	}
}
