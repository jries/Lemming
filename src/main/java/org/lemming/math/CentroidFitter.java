package org.lemming.math;

import net.imglib2.Cursor;
import net.imglib2.IterableInterval;
import net.imglib2.type.numeric.RealType;

public class CentroidFitter<T extends RealType<T>> implements FitterInterface  {
	
	private IterableInterval<T> op;
	private double thresh;

	public CentroidFitter(IterableInterval<T> op, double threshold){
		this.op = op;
		this.thresh = threshold;
	}
	
	@Override
	public double[] fit(){
		
		Cursor<T> c = op.cursor();
		int n = op.numDimensions();
		
		double [] r = new double[n*2];
		double sum = 0;
		
		while (c.hasNext()){
			 c.fwd();
			 double s = c.get().getRealDouble();
			 if (s>thresh)
				 for (int i = 0; i < n; i++)
					 r[i] += c.getIntPosition(i) * s;
				 sum += s;
		}
		
		if (sum == 0) return null;
		
		for (int i = 0; i < n; i++) 
			r[i] /= sum;
		
		double[] dev = new double[n];
		c.reset();
		while (c.hasNext()){
			c.fwd();
			double s = c.get().getRealDouble();
			if (s>thresh)
				 for (int i = 0; i < n; i++){
					 dev[i] += Math.abs(c.getIntPosition(i)-r[i])*s;
				 }
		}
		
		for (int i = 0; i < n; i++) 
			r[i+n] = dev[i]/sum;
		
		return r;		
	}
}
