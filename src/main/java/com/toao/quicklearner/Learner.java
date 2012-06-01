package com.toao.quicklearner;

import java.util.Map;

import com.google.protobuf.InvalidProtocolBufferException;
import com.toao.quicklearner.persistence.LearnerModel.Model;
import com.toao.quicklearner.persistence.LearnerModel.Model.LearnerType;

public abstract class Learner
{
	public double accuracy;
	
	public abstract String classify(Map<String, Double> features);

	public static Learner load(byte[] savedModel) throws InvalidProtocolBufferException
	{
		Model model = Model.parseFrom(savedModel);
		
		if( model.getLearnerType().equals(LearnerType.LOGISTIC) )
		{
			return new LogisticLearner(model.getLogistic());
		}
		
		throw new RuntimeException("Unable to find matching learner for model type");
	}
	
	public abstract byte[] serialise();
	
	public abstract Map<String, Double> getLabelProbabilities(Map<String, Double> features);
	
	public static LearnerBuilder builder()
	{
		return new LearnerBuilder();
	}
	
	public double getAccuracy()
	{
		return accuracy;
	}
}
