package com.toao.quicklearner;

import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Maps;

public class Learner
{
	private static Logger sLogger = LoggerFactory.getLogger(Learner.class);
	private final List<String> classes;
	private final List<InternalLearner> internalLearners;
	private final double accuracy;

	protected Learner(List<InternalLearner> internalLearners, List<String> classes, double accuracy)
	{
		this.accuracy = accuracy;
		// Classes must be size 2 or greater
		// Internal learners must be 1 if classes.size is 2 otherwise must be equal to classes in size
		this.internalLearners = ImmutableList.copyOf(internalLearners);
		this.classes = ImmutableList.copyOf(classes);
	}

	private Map<String,Double> mapFromSet(Set<String> features)
	{
		Map<String,Double> map = Maps.newHashMap();
		
		for( String feature : features )
		{
			map.put(feature, 1.0);
		}
		
		return map;
	}
	
	public String classify(Set<String> features)
	{
		 return classify(mapFromSet(features));
	}
	
	public String classify(Map<String, Double> features)
	{
		Map<String, Double> classProbs = getClassProbabilities(features);
		
		double bestProb = Double.MIN_VALUE;
		String bestClass = null;
				
		for( Entry<String, Double> e : classProbs.entrySet() )
		{
			String className = e.getKey();
			double classProb = e.getValue();
			
			if( classProb > bestProb )
			{
				bestClass = className;
				bestProb = classProb;
			}
		}
		
		return bestClass;
	}
	
	public Map<String,Double> getClassProbabilities(Map<String,Double> features)
	{
		Map<String,Double> classProbs = Maps.newHashMap();

		if( classes.size() == 2 )
		{
			InternalLearner learner = internalLearners.get(0);
			
			double prob = learner.classify(features);

			classProbs.put(classes.get(1), prob);
			classProbs.put(classes.get(0), 1.0 - prob);
		}
		else
		{
			for( int c = 0; c < classes.size() ; c++ )
			{
				String currentClass = classes.get(c);
				InternalLearner learner = internalLearners.get(c);
				
				double classProb = learner.classify(features);
				
				classProbs.put(currentClass, classProb);
			}
		}
		
		return classProbs;
	}
	
	public static LearnerBuilder builder()
	{
		return new LearnerBuilder();
	}

	public double getAccuracy()
	{
		return accuracy;
	}
}
