package com.toao.quicklearner;

import java.util.Map;

public class InternalLearner
{
	private final Map<String, Double> featuresOmega;
	private final Map<String, Double> featureMean;
	private final Map<String, Double> featureStdDev;

	protected InternalLearner(Map<String, Double> featuresOmega, Map<String, Double> featureMean, Map<String, Double> featureStdDev)
	{
		this.featuresOmega = featuresOmega;
		this.featureMean = featureMean;
		this.featureStdDev = featureStdDev;
	}

	public double classify(Map<String, Double> features)
	{
		double sum = 0.0;

		for (final String feature : features.keySet())
		{
			if (featuresOmega.containsKey(feature))
			{
				double omega = featuresOmega.get(feature);

				sum += omega * (features.get(feature) - featureMean.get(feature)) / featureStdDev.get(feature);
			}
		}

		double hx = 1 / (1 + Math.exp(-sum));

		return hx;
	}
}
