package org.lemming.inputs;

import java.util.AbstractList;
import java.util.ArrayList;
import java.util.Random;

import org.lemming.data.Localization;
import org.lemming.data.XYLocalization;
import org.lemming.interfaces.Source;

/**
 * A generator of N random localizations between (0,0) and (width,height)
 * 
 * @author Joe Borbely
 *
 */
public class RandomLocalizer implements Source<Localization> {

	/** The number of localizations to create */
	int N;
	
	/** the image width, in pixels */
	double dwidth;
	
	/** the image height, in pixels */
	double dheight;
	
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
	public void beforeRun() {
	}

	@Override
	public void afterRun() {
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
