package org.lemming.outputs;

import java.util.Timer;
import java.util.TimerTask;

import ij.ImagePlus;
import ij.process.FloatProcessor;

import org.lemming.data.Localization;
import org.lemming.data.Rendering;
import org.lemming.input.SI;

public class HistogramRender extends SI<Localization> implements Rendering {
	
	int xBins=256; // the number of bins to use to segment the x-axis
	int yBins=256; // the number of bins to use to segment the y-axis
	
	float maxVal=-Float.MAX_VALUE; // keeps track of the maximum value in the histogram
	float[] values; // list to contain the histogram values

	double xmin=0.0; // the minimum x-localization value to include in the histogram image
	double xmax=(double)(xBins); //the maximum x-localization value to include in the histogram image
	double ymin=0.0; // the minimum y-localization value to include in the histogram image
	double ymax=(double)(yBins); //the maximum y-localization value to include in the histogram image

	Timer t = new Timer();
	
	String title = "LemMING!"; // title of the image
	
	FloatProcessor fp; // from ImageJ
	ImagePlus ip; // from ImageJ
	
	public HistogramRender() {
		this(256,256,0,256,0,256);
	}

	public HistogramRender(int xBins, int yBins, double xmin, double xmax, double ymin, double ymax) {
		this.xBins = xBins;
		this.yBins = yBins;
		this.xmin = xmin;
		this.xmax = xmax;
		this.ymin = ymin;
		this.ymax = ymax;
		initialize();		
	}

	public void setRange(double xmin, double xmax, double ymin, double ymax){
		this.xmin = xmin;
		this.xmax = xmax;
		this.ymin = ymin;
		this.ymax = ymax;
	}

	public void setBins(int xBins, int yBins){
		this.xBins = xBins;
		this.yBins = yBins;
		initialize();
	}

	public void setTitle(String title){
		this.title = title;
		ip.setTitle(title);
	}
	
	private void initialize(){
		values = new float[xBins*yBins];
		fp = new FloatProcessor(xBins, yBins,values);
		ip = new ImagePlus(title, fp);
		ip.show();
	}

	@Override
	public void run() {
		t.schedule(new TimerTask() {
			@Override
			public void run() {
				update();
			}
		}, 100, 100);		
				
		super.run();
	}

	@Override
	public void process(Localization element) {
		double x = element.getX();
		double y = element.getY();
        if ( (x >= xmin) && (x <= xmax) && (y >= ymin) && (y <= ymax)) {
        	double xwidth = (xmax - xmin) /(double)xBins;
        	double ywidth = (ymax - ymin) /(double)yBins;
        	int xindex = (int)((x - xmin) / xwidth);
        	int yindex = (int)((y - ymin) / ywidth);
        	float val = values[xindex+yindex*xBins]++;
        	if (val > maxVal)
        		maxVal = val;
			//fp.setf(xindex, yindex, val);	
        }
	}
	
	void update() {
        if (ip==null)
        	return;
        
        ip.updateAndDraw();
		ip.setDisplayRange(0, maxVal);	
	}

}
