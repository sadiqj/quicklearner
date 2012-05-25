package com.toao.quicklearner;

import java.util.Map;

import com.google.common.collect.ImmutableMap;

public class Example
{
	private final String label;
	private final ImmutableMap<String, Double> features;

	protected Example(String label, Map<String, Double> features)
	{
		this.label = label;
		ImmutableMap.Builder<String,Double> map = ImmutableMap.builder(); 
		this.features = map.putAll(features).put("_bias", 1.0).build();
	}

	public String getLabel()
	{
		return label;
	}

	public ImmutableMap<String, Double> getFeatures()
	{
		return features;
	}

	public Example get()
	{
		return this;
	}
}
