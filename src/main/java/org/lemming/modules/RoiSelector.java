package org.lemming.modules;

import java.util.Map;

import net.imglib2.roi.RectangleRegionOfInterest;

import org.lemming.pipeline.Element;
import org.lemming.pipeline.Localization;
import org.lemming.pipeline.SingleRunModule;
import org.lemming.pipeline.Store;

@SuppressWarnings("deprecation")
public class RoiSelector extends SingleRunModule {
	
	private RectangleRegionOfInterest roi;
	private long start;
	private String inputKey;
	private int counter=0;
	private boolean keep;
	private Store output;

	public RoiSelector(final double x,final double y,final double xLength,final double yLength, boolean keep){
	 	roi = new RectangleRegionOfInterest(new double[]{x,y}, new double[]{xLength,yLength});
	 	this.keep = keep;
	}
	
	public RoiSelector(final double[] origin, final double[] extent, final boolean keep){
		roi = new RectangleRegionOfInterest(origin, extent);
		this.keep = keep;
	}
	
	@Override
	protected void beforeRun(){ 
		inputKey = inputs.keySet().iterator().next();
		output = outputs.values().iterator().next();
		start = System.currentTimeMillis();
		for ( String key : inputs.keySet()){
			while (inputs.get(key).isEmpty()) pause(10);
		}
		
	}
	
	@Override
	public void process(Map<String, Element> data) {
		Localization loc = (Localization) data.get(inputKey);
		if (loc==null) return;
		
		if (roi.contains(new double[]{loc.getX(),loc.getY()})){
			output.put(loc); // put ROI to output store
			counter++;
		}
		
		if (keep)
			inputs.get(inputKey).put(loc); //put it back to the input store
		
		if(loc.isLast())
			cancel();			
	}
	
	@Override
	protected void afterRun(){
		System.out.println("ROI done with " + counter + " elements in " + (System.currentTimeMillis()-start) + "ms.");
	}

}
