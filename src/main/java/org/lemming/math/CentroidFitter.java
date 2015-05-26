package org.lemming.math;

import net.imglib2.Cursor;
import net.imglib2.IterableInterval;
import net.imglib2.type.numeric.RealType;

public class CentroidFitter<T extends RealType<T>> implements FitterInterface  {
	
	private IterableInterval<T> op;
	private double thresh;
	double[] center;

	public CentroidFitter(IterableInterval<T> op_, double threshold_){
		op = op_;
		thresh = threshold_;
		center = new double[op.numDimensions()];
		for (int d=0; d<op.numDimensions();++d)
			center[d] = op.min(d)+(op.dimension(d)/2);		
	}
	
	@Override
	public double[] fit(){
		
		Cursor<T> c = op.cursor();
		int n = op.numDimensions();
		
		double [] r = new double[n*2];
		double sum = 0;
		
		while (c.hasNext()){
			 c.fwd();
			 
			 double s = c.get().getRealDouble()-thresh;
			 if (s>0){
				 for (int i = 0; i < n; i++){
					 int pos = c.getIntPosition(i);
					 r[i] += (center[i] - pos) * s;
				 }
				 sum = sum + s;
			 }
		}
		
		if (sum == 0) return null;
		
		for (int i = 0; i < n; i++) 
			r[i] = (r[i] / sum) + center[i];
		
		double[] dev = new double[n];
		c.reset();
		while (c.hasNext()){
			c.fwd();
			double s = c.get().getRealDouble()-thresh;
			if (s>0)
				 for (int i = 0; i < n; i++){
					 dev[i] += Math.abs(c.getIntPosition(i)-r[i])*s;
				 }
		}
		
		for (int i = 0; i < n; i++) 
			r[i+n] = dev[i]/sum;
		
		return r;		
	}
}
