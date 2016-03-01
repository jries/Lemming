package org.lemming.math;

import java.util.ArrayList;
import java.util.List;

import org.lemming.interfaces.Element;
import org.lemming.modules.Fitter;
import org.lemming.pipeline.Localization;

import net.imglib2.Cursor;
import net.imglib2.Interval;
import net.imglib2.RandomAccess;
import net.imglib2.RealPoint;
import net.imglib2.img.Img;
import net.imglib2.type.numeric.RealType;
import net.imglib2.view.IntervalView;
import net.imglib2.view.Views;

/**
 * Calculating centroids on a {@link #RandomAccessibleInterval}
 * 
 * @author Ronny Sczech
 *
 * @param <T>
 */
public class CentroidFitterRA<T extends RealType<T>>  {
	
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

	public static <T extends RealType<T>> List<Element> fit(List<Element> sliceLocs, Img<T> pixels, int halfKernel, float pixelDepth) {
		final List<Element> found = new ArrayList<>();
        //final Rectangle imageRoi = ip.getRoi();
        long[] imageMin = new long[2];
        long[] imageMax = new long[2];
        for (Element el : sliceLocs) {
            final Localization loc = (Localization) el;
             
            long x = Math.round(loc.getX().doubleValue()/pixelDepth);
			long y = Math.round(loc.getY().doubleValue()/pixelDepth);
			pixels.min(imageMin);
			pixels.max(imageMax);
			final Interval roi = Fitter.cropInterval(imageMin,imageMax,new long[]{x - halfKernel,y - halfKernel},new long[]{x + halfKernel,y + halfKernel});
			final CentroidFitterRA<T> cf = new CentroidFitterRA<T>(Views.interval(pixels, roi),0);
            final double[] res = cf.fit();
         
            found.add(new Localization(res[0]*pixelDepth, res[1]*pixelDepth, res[4], 1L));
        }
 
        return found;
	}
}
