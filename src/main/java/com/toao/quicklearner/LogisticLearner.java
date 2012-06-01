package com.toao.quicklearner;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkNotNull;

import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.toao.quicklearner.persistence.LearnerModel;
import com.toao.quicklearner.persistence.LearnerModel.Model;
import com.toao.quicklearner.persistence.LearnerModel.Model.LearnerType;
import com.toao.quicklearner.persistence.LearnerModel.Model.Logistic;
import com.toao.quicklearner.persistence.LearnerModel.Model.Logistic.InternalLogistic;

public class LogisticLearner extends Learner
{
	private final List<String> labels;
	private final List<LogisticInternalLearner> internalLearners;

	protected LogisticLearner(List<LogisticInternalLearner> internalLearners, List<String> labels, double accuracy)
	{
		this.accuracy = accuracy;
		checkNotNull(internalLearners);
		checkNotNull(labels);
		checkArgument(labels.size() >= 2, "Must be two or more labels.");
		checkArgument((labels.size() == 2 && internalLearners.size() == 1) || (labels.size() > 2 && internalLearners.size() == labels.size()), "Multi-label classification requires one learner per label");
		this.internalLearners = ImmutableList.copyOf(internalLearners);
		this.labels = ImmutableList.copyOf(labels);
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
		Map<String, Double> classProbs = getLabelProbabilities(features);
		
		double bestProb = Double.MIN_VALUE;
		String bestLabel = null;
				
		for( Entry<String, Double> e : classProbs.entrySet() )
		{
			String labelName = e.getKey();
			double labelProb = e.getValue();
			
			if( labelProb > bestProb )
			{
				bestLabel = labelName;
				bestProb = labelProb;
			}
		}
		
		return bestLabel;
	}
	
	public Map<String,Double> getLabelProbabilities(Map<String,Double> features)
	{
		Map<String,Double> labelProbs = Maps.newHashMap();

		if( labels.size() == 2 )
		{
			LogisticInternalLearner learner = internalLearners.get(0);
			
			double prob = learner.classify(features);

			labelProbs.put(labels.get(1), prob);
			labelProbs.put(labels.get(0), 1.0 - prob);
		}
		else
		{
			for( int c = 0; c < labels.size() ; c++ )
			{
				String currentLabel = labels.get(c);
				LogisticInternalLearner learner = internalLearners.get(c);
				
				double labelProb = learner.classify(features);
				
				labelProbs.put(currentLabel, labelProb);
			}
		}
		
		return labelProbs;
	}

	protected LogisticLearner(Logistic logistic)
	{
		this.accuracy = logistic.getAccuracy();
		
		this.labels = ImmutableList.copyOf(logistic.getLabelsList());
		
		this.internalLearners = Lists.newArrayList();
		
		for( InternalLogistic i : logistic.getInternalsList() )
		{
			this.internalLearners.add( new LogisticInternalLearner(i) );
		}
	}
	
	@Override
	public byte[] serialise()
	{
		Model.Builder modelBuilder = LearnerModel.Model.newBuilder();
		
		modelBuilder.setLearnerType(LearnerType.LOGISTIC);
		
		Logistic.Builder logisticBuilder = modelBuilder.getLogisticBuilder();
		
		for( LogisticInternalLearner internal : internalLearners )
		{
			logisticBuilder.addInternals( internal.save() );
		}
		
		logisticBuilder.setAccuracy(accuracy);
		
		for( String label : labels )
		{
			logisticBuilder.addLabels(label);
		}
		
		return modelBuilder.build().toByteArray();
	}
}
