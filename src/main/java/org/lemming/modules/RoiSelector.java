package org.lemming.modules;

import java.util.Map;

import net.imglib2.roi.RectangleRegionOfInterest;

import org.lemming.pipeline.Element;
import org.lemming.pipeline.Localization;
import org.lemming.pipeline.Module;

@SuppressWarnings("deprecation")
public class RoiSelector extends Module {
	
	private RectangleRegionOfInterest roi;
	private String outputKey;
	private long start;
	private String inputKey;
	private int counter=0;

	public RoiSelector(final double x,final double y,final double xLength,final double yLength){
	 	roi = new RectangleRegionOfInterest(new double[]{x,y}, new double[]{xLength,yLength});
	}
	
	public RoiSelector(final double[] origin, final double[] extent){
		roi = new RectangleRegionOfInterest(origin, extent);
	}
	
	@Override
	protected void beforeRun(){ 
		inputKey = inputs.keySet().iterator().next();
		outputKey = outputs.keySet().iterator().next();
		start = System.currentTimeMillis();
		for ( String key : inputs.keySet()){
			while (inputs.get(key).isEmpty()) pause(10);
		}
	}
	
	@SuppressWarnings("unchecked")
	@Override
	public void process(Map<String, Element> data) {
		Localization loc = (Localization) data.get(inputKey);
		if (loc==null) return;
		
		if (roi.contains(new double[]{loc.getX(),loc.getY()})){
			outputs.get(outputKey).put(loc); // put ROI to output store
			counter++;
		}
		
		inputs.get(inputKey).put(loc); //put it back to the input store
		
		if(loc.isLast())
			cancel();		
	}
	
	@Override
	protected void afterRun(){
		System.out.println("ROI done with " + counter + " elements in " + (System.currentTimeMillis()-start) + "ms.");
	}

}
