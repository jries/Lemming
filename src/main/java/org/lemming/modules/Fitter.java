package org.lemming.modules;

import java.util.Map;

import org.lemming.math.FitterType;
import org.lemming.pipeline.Element;
import org.lemming.pipeline.Module;
import org.lemming.pipeline.Store;

public class Fitter extends Module {

	private FitterType ftype;
	private String inputKey;
	private long start;
	private Store output;
	private int size;

	public Fitter(final FitterType ftype, int size){
		this.ftype = ftype;
		this.size = size;
	}
	
	@Override
	protected void beforeRun() {
		// this module accepts two inputs
		inputKey = inputs.keySet().iterator().next();
		output = outputs.values().iterator().next();
		while (inputs.get(inputKey).isEmpty())
			pause(10);
		start = System.currentTimeMillis();
		
		
		
		
	}

	@Override
	public void process(Map<String, Element> data) {

	}
	
	/*public double[] fitAutoWeightedCentroid(IterableInterval<T> op, ThresholdingType ttype){
		final Histogram1d<T> hist = new Histogram1d<>(new Real1dBinMapper<T>(0, 0, 0, true));
        final T thresh = op.firstElement().createVariable();
        thresh.setReal(new FindThreshold<>(ttype, thresh.createVariable()).compute(hist).getRealFloat());
		return null;
	}*/
	
	@Override
	protected void afterRun() {
		System.out.println("Fitting done in "
				+ (System.currentTimeMillis() - start) + "ms.");
	}

}
