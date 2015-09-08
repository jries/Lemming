package org.lemming.modules;

import net.imglib2.roi.RectangleRegionOfInterest;

import org.lemming.interfaces.Element;
import org.lemming.pipeline.FrameElements;
import org.lemming.pipeline.Localization;
import org.lemming.pipeline.SingleRunModule;

@SuppressWarnings("deprecation")
public class RoiSelector extends SingleRunModule {
	
	private RectangleRegionOfInterest roi;
	private long start;
	private int counter=0;
	private int skipFrames;

	public RoiSelector(final double x,final double y,final double xLength,final double yLength, final int skipFrames){
	 	roi = new RectangleRegionOfInterest(new double[]{x,y}, new double[]{xLength,yLength});
	 	this.skipFrames = skipFrames;
	}
	
	public RoiSelector(final double[] origin, final double[] extent){
		roi = new RectangleRegionOfInterest(origin, extent);
	}
	
	@Override
	protected void beforeRun(){ 
		start = System.currentTimeMillis();		
	}
	
	@Override
	public Element process(Element data) {
		
		FrameElements fe = (FrameElements) data;
		if (fe==null) return null;
		if (fe.getNumber() < skipFrames) return null;
		
		for ( Element el : fe.getList()){
			Localization loc = (Localization) el;
				if (roi.contains(new double[]{loc.getX(),loc.getY()})){
				newOutput(loc); // put ROI to output store
				counter++;
			}
		}
		
		if(fe.isLast())
			cancel();
		return null;			
	}
	
	@Override
	protected void afterRun(){
		System.out.println("ROI done with " + counter + " elements in " + (System.currentTimeMillis()-start) + "ms.");
	}

	@Override
	public boolean check() {
		return inputs.size()==1 && outputs.size()>=1;
	}

}
