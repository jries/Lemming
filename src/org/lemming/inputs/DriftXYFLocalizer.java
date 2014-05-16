package org.lemming.inputs;

import java.util.Random;

import org.lemming.data.XYFLocalization;

/**
 * A generator of N random localizations between (0,0) and (width,height)
 * 
 * @author Joran Deschamps
 *
 */
public class DriftXYFLocalizer extends SO<XYFLocalization> {

	/** The number of localizations to create */
	int N,Ntot;
	
	/** the image width, in pixels */
	double dwidth = 100;
	
	/** the image height, in pixels */
	double dheight = 100;

	private double[] X = {20,40,80};
	private double[] Y = {15,50,60};
	
	/**
	 * 
	 * 
	 * @param N - the number of localizations to generate
	 * @param width - the image width, in pixels (e.g. 256)
	 * @param height - the image height, in pixels (e.g. 256)*/	
	public DriftXYFLocalizer(int N) {
		this.N = N;
		Ntot = N;
	}
	
	@Override
	public boolean hasMoreOutputs() {
		return N>0;
	}

	@Override
	public XYFLocalization newOutput() {
    	N--;
    	Random rand = new Random();
    	int t = N%3;
    	    	
    	double x = X[t]+0.5*Math.cos(((double) Ntot-N)/(double) Ntot)+0.1*rand.nextGaussian();
    	double y = Y[t]+0.5*Math.sin(((double) Ntot-N)/(double) Ntot)+0.1*rand.nextGaussian();
    	
    	return new XYFLocalization(Ntot-N, x, y);
	}
}
