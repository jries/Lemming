package org.lemming.modules;

import ij.ImagePlus;
import ij.process.FloatProcessor;

import org.lemming.interfaces.Element;
import org.lemming.pipeline.Localization;
import org.lemming.pipeline.SingleRunModule;

public class HistogramRenderer extends SingleRunModule {
	
	private int xBins;
	private int yBins;
	private double xmin;
	private double xmax;
	private double ymin;
	private double ymax;
	private float[] values;
	private ImagePlus ip;
	protected String title = "LemMING!"; // title of the image
	private long counter = 0;
	private long start;

	public HistogramRenderer(){
		this(256,256,0,256,0,256);
	}

	public HistogramRenderer(int xBins, int yBins, double xmin, double xmax, double ymin, double ymax) {
		this.xBins = xBins;
		this.yBins = yBins;
		this.xmin = xmin;
		this.xmax = xmax;
		this.ymin = ymin;
		this.ymax = ymax;
		values = new float[xBins*yBins];
		ip = new ImagePlus(title, new FloatProcessor(xBins, yBins,values));
		ip.setDisplayRange(0, 5);
		ip.show();		
	}
	
	@Override
	protected void beforeRun() {
		start = System.currentTimeMillis();
	}

	@Override
	public Element process(Element data) {
		Localization loc = (Localization) data;
		if(loc==null) return null;
		
		counter ++;
		
		if(loc.isLast())
			cancel();
		
		double x = loc.getX();
		double y = loc.getY();
        if ( (x >= xmin) && (x <= xmax) && (y >= ymin) && (y <= ymax)) {
        	double xwidth = (xmax - xmin) / xBins;
        	double ywidth = (ymax - ymin) / yBins;
        	long xindex = Math.round((x - xmin) / xwidth);
        	long yindex = Math.round((y - ymin) / ywidth);
        	values[(int) (xindex+yindex*xBins)]++;
        }		
		
        if (counter%100==0)
        	ip.updateAndDraw();
        
		return null;
	}
	
	@Override
	public void afterRun(){
		ip.updateAndDraw();
		System.out.println("Rendering done in "
				+ (System.currentTimeMillis() - start) + "ms.");
		while(ip.isVisible()) pause(10);
	}

	@Override
	public boolean check() {
		// TODO Auto-generated method stub
		return false;
	}

}
