package org.lemming.inputs;

import java.util.Random;

import org.lemming.data.Localization;
import org.lemming.data.XYLocalization;

/**
 * A generator of N random localizations between (0,0) and (width,height)
 * 
 * @author Joe Borbely
 *
 */
public class RandomLocalizer extends SO<Localization> {

	/** The number of localizations to create */
	int N;
	
	/** the image width, in pixels */
	double dwidth;
	
	/** the image height, in pixels */
	double dheight;
	
	/** Generate a list of N, randomly-located molecules and put the 
	 * localizations into a Store.
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
}
