package com.toao.quicklearner;

import static org.junit.Assert.assertTrue;

import java.util.Map;
import java.util.Random;

import org.junit.Test;

import com.google.common.collect.ImmutableMap;
import com.google.protobuf.InvalidProtocolBufferException;
import com.toao.quicklearner.Learner;
import com.toao.quicklearner.LearnerBuilder;

public class TestClassification
{
	@Test
	public void testWithFakeLinearlySeperableDataTwoClass() throws InvalidProtocolBufferException
	{
		LearnerBuilder builder = Learner.builder();
		
		Random random = new Random();
		
		for( int c = 0 ; c < 100 ; c++ )
		{
			if( random.nextDouble() < 0.5 )
			{
				Map<String,Double> featuresMap = ImmutableMap.of("x", -1.0 + random.nextDouble()*2.0);
				
				builder.addExample("false", featuresMap);
			}
			else
			{
				Map<String,Double> featuresMap = ImmutableMap.of("x", 5.0 + random.nextDouble()*2.0);
				
				builder.addExample("true", featuresMap);
			}
		}
		
		Learner learner = builder.build();
		
		testTwoClassLearner(learner);
		
		// Now we check that serialisation actually works properly
		
		byte[] savedLearner = learner.serialise();
		
		Learner newLearner = Learner.load(savedLearner);
		
		testTwoClassLearner(newLearner);
	}
	
	private void testTwoClassLearner(Learner learner)
	{
		Map<String, Double> labelProbabilities = learner.getLabelProbabilities(ImmutableMap.of("x", 0.0));
		
		assertTrue("Returned only true/false labels", labelProbabilities.keySet().size() == 2 && labelProbabilities.containsKey("true") && labelProbabilities.containsKey("false"));
		
		assertTrue("Correctly classifies test as false", labelProbabilities.get("false") > 0.5 && labelProbabilities.get("false") > labelProbabilities.get("true"));
				
		labelProbabilities = learner.getLabelProbabilities(ImmutableMap.of("x", 6.0));
		
		assertTrue("Returned only true/false labels", labelProbabilities.keySet().size() == 2 && labelProbabilities.containsKey("true") && labelProbabilities.containsKey("false"));
		
		assertTrue("Correctly classifies test as true", labelProbabilities.get("true") > 0.5 && labelProbabilities.get("true") > labelProbabilities.get("false"));
	}
	
	@Test
	public void testWithFakeLinearlySeperableDataMultiClass() throws InvalidProtocolBufferException
	{
		LearnerBuilder builder = Learner.builder();
		
		Random random = new Random();
		
		for( int c = 0 ; c < 100 ; c++ )
		{
			if( random.nextDouble() < 0.33 )
			{
				Map<String,Double> featuresMap = ImmutableMap.of("x", -5.0 + random.nextDouble()*2.0);
				
				builder.addExample("blue", featuresMap);
			}
			else if( random.nextDouble() < 0.66 )
			{
				Map<String,Double> featuresMap = ImmutableMap.of("x", 5.0 + random.nextDouble()*2.0);
				
				builder.addExample("green", featuresMap);
			}
			else
			{
				Map<String,Double> featuresMap = ImmutableMap.of("y", 3.0 + random.nextDouble()*2.0);
				
				builder.addExample("red", featuresMap);
			}
		}
		
		Learner learner = builder.build();
		
		testMultiClassLearner(learner);
		
		// Now we check that serialisation actually works properly, again
		
		byte[] savedLearner = learner.serialise();
		
		Learner newLearner = Learner.load(savedLearner);
		
		testMultiClassLearner(newLearner);
	}

	private void testMultiClassLearner(Learner learner)
	{
		Map<String, Double> labelProbabilities = learner.getLabelProbabilities(ImmutableMap.of("x", -6.0));
		
		assertTrue("Returned only blue/green/red labels", labelProbabilities.keySet().size() == 3 && labelProbabilities.containsKey("blue") && labelProbabilities.containsKey("green") && labelProbabilities.containsKey("red"));
		
		assertTrue("Correctly classifies test as blue", labelProbabilities.get("blue") > 0.5 && labelProbabilities.get("blue") > labelProbabilities.get("green") && labelProbabilities.get("blue") > labelProbabilities.get("red"));
				
		labelProbabilities = learner.getLabelProbabilities(ImmutableMap.of("x", 6.0));
		
		assertTrue("Correctly classifies test as green", labelProbabilities.get("green") > 0.5 && labelProbabilities.get("green") > labelProbabilities.get("blue") && labelProbabilities.get("green") > labelProbabilities.get("red"));
		
		labelProbabilities = learner.getLabelProbabilities(ImmutableMap.of("y", 4.0));
		
		assertTrue("Correctly classifies test as red", labelProbabilities.get("red") > 0.5 && labelProbabilities.get("red") > labelProbabilities.get("blue") && labelProbabilities.get("red") > labelProbabilities.get("green"));
	}
}
