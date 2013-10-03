package org.lemming.input;

import java.util.Random;

import org.lemming.data.Localization;
import org.lemming.data.Store;
import org.lemming.data.XYLocalization;
import org.lemming.interfaces.Source;
import org.lemming.outputs.NullStoreWarning;

public class RandomLocalizer implements Source<Localization> {

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
	
	boolean hasMore = true;

	@Override
	public void run() {
		
		if (localizations==null) {new NullStoreWarning(this.getClass().getName()); return;}
		
		long ID = 0L;
		double x, y;
		Random rand = new Random();
	    for (int i = 0; i < N; i++){
	    	x = (double)rand.nextInt(width-1) + rand.nextDouble();
	    	y = (double)rand.nextInt(height-1) + rand.nextDouble();
	    	localizations.put(new XYLocalization(x, y, ID++));
	    }
	    
	    hasMore = false;
	}

	@Override
	public void setOutput(Store<Localization> s) {
		localizations = s;
	}
	
	@Override
	public boolean hasMoreOutputs() {
		return hasMore;
	}
}
