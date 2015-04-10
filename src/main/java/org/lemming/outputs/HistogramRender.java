package org.lemming.outputs;

import java.util.Timer;
import java.util.TimerTask;

import ij.ImagePlus;
import ij.process.FloatProcessor;

import org.lemming.interfaces.Localization;
import org.lemming.interfaces.Rendering;

/**
 * 
 * @author Thomas Pengo, Ronny Sczech
 *
 */
public class HistogramRender extends SingleInput<Localization> implements Rendering {
	
	private int xBins=256; // the number of bins to use to segment the x-axis
	private int yBins=256; // the number of bins to use to segment the y-axis
	
	//private float maxVal=-Float.MAX_VALUE; // keeps track of the maximum value in the histogram
	private float[] values; // list to contain the histogram values

	private double xmin=0.0; // the minimum x-localization value to include in the histogram image
	private double xmax=(double)(xBins); //the maximum x-localization value to include in the histogram image
	private double ymin=0.0; // the minimum y-localization value to include in the histogram image
	private double ymax=(double)(yBins); //the maximum y-localization value to include in the histogram image

	private Timer t = new Timer();
	
	private String title = "LemMING!"; // title of the image
	
	private FloatProcessor fp; // from ImageJ
	private ImagePlus ip; // from ImageJ
	
	/**
	 * 
	 */
	public HistogramRender() {
		this(256,256,0,256,0,256);
	}

	/**
	 * @param xBins - value
	 * @param yBins - value
	 * @param xmin - value
	 * @param xmax - value
	 * @param ymin - value
	 * @param ymax - value
	 */
	public HistogramRender(int xBins, int yBins, double xmin, double xmax, double ymin, double ymax) {
		this.xBins = xBins;
		this.yBins = yBins;
		this.xmin = xmin;
		this.xmax = xmax;
		this.ymin = ymin;
		this.ymax = ymax;
		initialize();		
	}

	/**
	 * @param xmin - value
	 * @param xmax - value
	 * @param ymin - value
	 * @param ymax - value
	 */
	public void setRange(double xmin, double xmax, double ymin, double ymax){
		this.xmin = xmin;
		this.xmax = xmax;
		this.ymin = ymin;
		this.ymax = ymax;
	}

	/**
	 * @param xBins - value
	 * @param yBins - value
	 */
	public void setBins(int xBins, int yBins){
		this.xBins = xBins;
		this.yBins = yBins;
		initialize();
	}

	/**
	 * @param title - Window Title
	 */
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
		if(element==null) return;
		if(element.isLast()){ 
			stop();
			//t.cancel();
			System.out.println("Rendering finished:"+element.getID()); 
			return;
		}
		double x = element.getX();
		double y = element.getY();
        if ( (x > xmin) && (x <= xmax) && (y > ymin) && (y <= ymax)) {
        	double xwidth = (xmax - xmin) /(double)xBins;
        	double ywidth = (ymax - ymin) /(double)yBins;
        	int xindex = (int) Math.floor((x - xmin) / xwidth);
        	int yindex = (int) Math.floor((y - ymin) / ywidth);
        	values[xindex+yindex*xBins]++;
        }
	}
		
	void update() {
        if (ip==null)
        	return;
        
        ip.updateAndDraw();
		ip.setDisplayRange(0, 5);	
	}

}
