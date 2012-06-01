package com.toao.quicklearner;

import java.util.Map;

import com.google.common.collect.ImmutableMap;
import com.toao.quicklearner.persistence.LearnerModel.Model.Logistic.InternalLogistic;
import com.toao.quicklearner.persistence.LearnerModel.Model.Logistic.InternalLogistic.Builder;

public class LogisticInternalLearner
{
	private final Map<String, Double> featureOmega;
	private final Map<String, Double> featureMean;
	private final Map<String, Double> featureStdDev;

	protected LogisticInternalLearner(Map<String, Double> featuresOmega, Map<String, Double> featureMean, Map<String, Double> featureStdDev)
	{
		this.featureOmega = featuresOmega;
		this.featureMean = featureMean;
		this.featureStdDev = featureStdDev;
	}

	public double classify(Map<String, Double> features)
	{
		double sum = 0.0;

		for (final String feature : features.keySet())
		{
			if (featureOmega.containsKey(feature))
			{
				double omega = featureOmega.get(feature);

				sum += omega * (features.get(feature) - featureMean.get(feature)) / featureStdDev.get(feature);
			}
		}

		double hx = 1 / (1 + Math.exp(-sum));

		return hx;
	}

	protected LogisticInternalLearner(InternalLogistic internal)
	{
		int i = 0;
		
		ImmutableMap.Builder<String, Double> omegaBuilder = ImmutableMap.builder();
		ImmutableMap.Builder<String, Double> meanBuilder = ImmutableMap.builder();
		ImmutableMap.Builder<String, Double> stdDevBuilder = ImmutableMap.builder();
		
		for( String feature : internal.getFeatureList() )
		{
			double omega = internal.getOmega(i);
			double mean = internal.getMean(i);
			double stdDev = internal.getStdDev(i);
			
			omegaBuilder.put(feature, omega);
			meanBuilder.put(feature, mean);
			stdDevBuilder.put(feature, stdDev);
			
			i++;
		}
		
		featureOmega = omegaBuilder.build();
		featureMean = meanBuilder.build();
		featureStdDev = stdDevBuilder.build();
	}
	
	public InternalLogistic save()
	{
		Builder builder = InternalLogistic.newBuilder();
		
		for( String feature : featureOmega.keySet() )
		{
			builder.addFeature(feature);
			builder.addMean(featureMean.get(feature));
			builder.addOmega(featureOmega.get(feature));
			builder.addStdDev(featureStdDev.get(feature));
		}
		
		return builder.build();
	}
}
