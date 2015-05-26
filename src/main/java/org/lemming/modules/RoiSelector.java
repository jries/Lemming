package org.lemming.modules;

import net.imglib2.roi.RectangleRegionOfInterest;

import org.lemming.pipeline.Element;
import org.lemming.pipeline.Localization;
import org.lemming.pipeline.SingleRunModule;
import org.lemming.pipeline.Store;

@SuppressWarnings("deprecation")
public class RoiSelector extends SingleRunModule {
	
	private RectangleRegionOfInterest roi;
	private long start;
	private int counter=0;
	private Store output;

	public RoiSelector(final double x,final double y,final double xLength,final double yLength){
	 	roi = new RectangleRegionOfInterest(new double[]{x,y}, new double[]{xLength,yLength});
	}
	
	public RoiSelector(final double[] origin, final double[] extent){
		roi = new RectangleRegionOfInterest(origin, extent);
	}
	
	@Override
	protected void beforeRun(){ 
		iterator = inputs.keySet().iterator().next();
		output = outputs.values().iterator().next();
		start = System.currentTimeMillis();		
	}
	
	@Override
	public void process(Element data) {
		Localization loc = (Localization) data;
		if (loc==null) return;
		
		if (roi.contains(new double[]{loc.getX(),loc.getY()})){
			output.put(loc); // put ROI to output store
			counter++;
		}
		
		if(loc.isLast())
			cancel();			
	}
	
	@Override
	protected void afterRun(){
		System.out.println("ROI done with " + counter + " elements in " + (System.currentTimeMillis()-start) + "ms.");
	}

}
