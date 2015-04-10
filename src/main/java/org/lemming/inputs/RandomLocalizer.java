package org.lemming.inputs;

import java.util.Random;

import org.lemming.data.XYLocalization;
import org.lemming.interfaces.Localization;

/**
 * A generator of N random localizations between (0,0) and (width,height)
 * 
 * @author Joe Borbely
 *
 */
public class RandomLocalizer extends SingleOutput<Localization> {

	/** The number of localizations to create */
	private int N;
	
	/** the image width, in pixels */
	private double dwidth;
	
	/** the image height, in pixels */
	private double dheight;
	
	/**
	 * A generator of N random localizations between (0,0) and (width,height).
	 * 
	 * @param N - the number of molecules to generate
	 * @param width - the image width, in pixels (e.g. 256)
	 * @param height - the image height, in pixels (e.g. 256)*/	
	public RandomLocalizer(int N, int width, int height) {
		this.N = N;
		dwidth = (double)width;
		dheight = (double)height;
	}
	
	@Override
	public boolean hasMoreOutputs() {
		return N>0;
	}

	@Override
	public Localization newOutput() {
		Random rand = new Random();
    	N--;
    	return new XYLocalization(dwidth*rand.nextDouble(), dheight*rand.nextDouble());
	}
	
	@Override
	public void afterRun(){
		XYLocalization lastLoc = new XYLocalization(0,0);
		lastLoc.setLast(true);
		output.put(lastLoc);
	}
	
}
