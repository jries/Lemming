package org.lemming.inputs;

import java.util.Random;

import org.lemming.data.Localization;
import org.lemming.data.Store;
import org.lemming.data.XYLocalization;

/**
 * A generator of N random localizations between (0,0) and (width,height)
 * 
 * @author Joe Borbely
 *
 */
public class RandomLocalizer extends SO<Localization> {

	Store<Localization> localizations;	
	int N, width, height;
	
	/** Generate a list of N, randomly-located molecules and put the 
	 * localizations into a Store.
	 * 
	 * @param N - the number of molecules to generate
	 * @param width - the image width, in pixels (e.g. 256)
	 * @param height - the image height, in pixels (e.g. 256)*/	
	public RandomLocalizer(int N, int width, int height) {
		this.N = N;
		this.width = width;
		this.height = height;
	}
	
	@Override
	public boolean hasMoreOutputs() {
		return N>0;
	}

	@Override
	public Localization newOutput() {
		Random rand = new Random();
    	double x = (double)rand.nextInt(width) + rand.nextDouble();
    	double y = (double)rand.nextInt(height) + rand.nextDouble();
    	N--;
    	return new XYLocalization(x, y);
	}
}
