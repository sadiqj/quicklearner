package com.toao.quicklearner;

import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;

public class InternalTrainer
{
	private static Logger sLogger = LoggerFactory.getLogger(InternalTrainer.class);
	
	protected InternalLearner learn(double regulariser, double learningRate, String positiveLabel, List<Example> examples)
	{
		sLogger.debug("build - called with positive label: {}", positiveLabel);

		sLogger.debug("build - found {} examples", examples.size());

		Set<String> featureSet = Sets.newHashSet();

		for (Example example : examples)
		{
			for (String key : example.getFeatures().keySet())
			{
				featureSet.add(key);
			}
		}

		sLogger.debug("build - found {} unique features", featureSet.size());

		List<String> features = Lists.newArrayList(featureSet);

		Map<String, Integer> featureIdx = Maps.newHashMap();

		for (int c = 0; c < features.size(); c++)
		{
			featureIdx.put(features.get(c), c);
		}

		double numExamples = examples.size();

		Collections.sort(features);

		Map<String, Double> featureStdDev = Maps.newHashMap();
		Map<String, Double> featureMean = Maps.newHashMap();

		featureMean.put("_bias", 0.0);
		featureStdDev.put("_bias", 1.0);

		for (int c = 0; c < features.size(); c++)
		{
			final String feature = features.get(c);

			if (!feature.equals("_bias"))
			{
				double tmpSum = 0.0, tmpSquared = 0.0;

				for (final Example example : examples)
				{
					ImmutableMap<String, Double> featureMap = example.getFeatures();

					if (featureMap.containsKey(feature))
					{
						double val = featureMap.get(feature);
						tmpSum += val;
						tmpSquared += val * val;
					}
				}

				final double stdDev = Math.sqrt(tmpSquared / numExamples - Math.pow(tmpSum / numExamples, 2.0));
				final double mean = tmpSum / numExamples;

				featureStdDev.put(feature, stdDev);
				featureMean.put(feature, mean);
			}
		}

		double delta = 0.0;
		int counter = 0;
		double previousCost = 0.0;
		double currentCost;
		double[] currentOmega = new double[features.size()];

		for (int i = 0; i < 1000; i++)
		{
			double[] omegaSum = new double[features.size()];

			sLogger.debug("build - starting learning iteration {}, delta: {}", i, delta);
			currentCost = 0.0;

			for (final Example example : examples)
			{
				final ImmutableMap<String, Double> exampleFeatures = example.getFeatures();

				double sum = 0.0;

				for (final String feature : exampleFeatures.keySet())
				{
					int idx = featureIdx.get(feature);

					double omega = currentOmega[idx];

					sum += omega * (exampleFeatures.get(feature) - featureMean.get(feature)) / featureStdDev.get(feature);
				}

				double hx = 1 / (1 + Math.exp(-sum));

				double cost = 0.0;
				double y = 0.0;

				if (example.getLabel().equals(positiveLabel))
				{
					cost = Math.log(hx);
					y = 1.0;
				}
				else
				{
					cost = Math.log(1 - hx);
				}

				cost *= -(1 / numExamples);

				currentCost += cost;

				double diff = hx - y;

				for (final String feature : exampleFeatures.keySet())
				{
					int idx = featureIdx.get(feature);

					omegaSum[idx] += diff * (exampleFeatures.get(feature) - featureMean.get(feature)) / featureStdDev.get(feature);
				}
			}

			sLogger.debug("build - cost at iteration {} = {}", i, currentCost);

			for (int c = 0; c < currentOmega.length; c++)
			{
				currentCost += regulariser * Math.pow(currentOmega[c], 2.0);
				currentOmega[c] = currentOmega[c] - (learningRate * ((1 / numExamples) * omegaSum[c] + (regulariser / (double)numExamples) * currentOmega[c]));
			}

			if (i > 0)
			{
				delta = previousCost - currentCost;
				sLogger.debug("build - cost delta after iteration = {}", delta);

				if (delta < 0)
				{
					sLogger.debug("build - cost oscillating, reducing learning rate from {} to {}", learningRate, 0.9 * learningRate);
					learningRate = 0.5 * learningRate;
				}
				else if (delta < 0.001)
				{
					counter++;

					if (counter > 2)
					{
						break;
					}
				}
			}

			previousCost = currentCost;
		}

		Map<String,Double> featuresOmega = Maps.newHashMap();
				
		for( String feature : features )
		{
			int idx = featureIdx.get(feature);
			
			featuresOmega.put(feature, currentOmega[idx]);
		}
		
		return new InternalLearner(featuresOmega, featureMean, featureStdDev);
	}
}
