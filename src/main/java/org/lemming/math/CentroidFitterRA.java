package org.lemming.math;

import net.imglib2.Cursor;
import net.imglib2.RandomAccess;
import net.imglib2.RealPoint;
import net.imglib2.type.numeric.RealType;
import net.imglib2.view.IntervalView;

/**
 * Calculating centroids on a {@link #RandomAccessibleInterval}
 * 
 * @author Ronny Sczech
 *
 * @param <T>
 */
public class CentroidFitterRA<T extends RealType<T>> implements FitterInterface  {
	
	private IntervalView<T> op;
	private double thresh;
	RealPoint center;

	public CentroidFitterRA(IntervalView<T> op_, double threshold_){
		op = op_;
		thresh = threshold_;
		center = new RealPoint(op.numDimensions());
		for (int d=0; d<op.numDimensions();++d)
			center.setPosition(op.min(d)+(op.dimension(d)/2), d);		
	}
	
	@Override
	public double[] fit(){
		
		Cursor<T> c = op.cursor();
		int n = op.numDimensions();
		
		double [] r = new double[n*2+1];
		double sum = 0;
		
		while (c.hasNext()){
			 c.fwd();
			 
			 double s = c.get().getRealDouble()-thresh;
			 if (s>0){
				 for (int i = 0; i < n; i++){
					 int pos = c.getIntPosition(i);
					 r[i] += (center.getDoublePosition(i) - pos) * s;
				 }
				 sum = sum + s;
			 }
		}
		
		if (sum == 0) return null;
		
		for (int i = 0; i < n; i++) 
			r[i] = (r[i] / sum) + center.getDoublePosition(i);
		
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
			r[i+n] = Math.sqrt(dev[i]/sum);
		
		RandomAccess<T> ra = op.randomAccess();
		for (int i = 0; i < n; i++){
			ra.setPosition(StrictMath.round(r[i]), i);
		}
		r[n*2] = ra.get().getRealDouble();
		return r;		
	}
}
