package org.lemming.processors;

import java.util.ArrayList;
import java.util.List;

import net.imglib2.Cursor;
import net.imglib2.Interval;
import net.imglib2.RandomAccess;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.algorithm.region.localneighborhood.Neighborhood;
import net.imglib2.algorithm.region.localneighborhood.RectangleShape;
import net.imglib2.type.numeric.RealType;
import net.imglib2.util.Intervals;
import net.imglib2.view.Views;

import org.lemming.data.Frame;
import org.lemming.data.Localization;
import org.lemming.data.XYFLocalization;
import org.lemming.processors.SISO;

public class PeakFinder<T extends RealType<T>, F extends Frame<T>> extends SISO<F,Localization> {

	/** The intensity of a pixel must be greater than {@code threshold} to be considered a local maximum */
	double threshold;

	public PeakFinder(double threshold) {
		this.threshold = threshold;
 	}
	
	@Override
	public void process(F frame) {
		process1(frame);
	}
	
	public void process1(Frame<T> frame) {
		//double[] pixels = (double[]) frame.getPixels();
		//float[] pixels = (float[]) frame.getPixels();
		
		Interval interval = Intervals.expand( frame.getPixels(), -1 );
		
		RandomAccessibleInterval<T> source = Views.interval( frame.getPixels(), interval );
		
		final Cursor< T > center = Views.iterable( source ).cursor();

		final RectangleShape shape = new RectangleShape( 1, true );

		for ( final Neighborhood< T > localNeighborhood : shape.neighborhoods( source ) )
		{
		    // what is the value that we investigate?
		    // (the center cursor runs over the image in the same iteration order as neighborhood)
		    final T centerValue = center.next();
		    
		    if (centerValue.getRealDouble() < threshold) 
		    	continue;
		
		    // keep this boolean true as long as no other value in the local neighborhood
		    // is larger or equal
		    boolean isMaximum = true;
		
		    // check if all pixels in the local neighborhood that are smaller
		    for ( final T value : localNeighborhood )
		    {
		        // test if the center is smaller than the current pixel value
		        if ( centerValue.compareTo( value ) <= 0 )
		        {
		            isMaximum = false;
		            break;
		        }
		    }
		    
		    if (isMaximum)
		    	output.put(new XYFLocalization(frame.getFrameNumber(), center.getIntPosition(0), center.getIntPosition(1)));
		}
		
		//for now just print the results to the console
		//List<Integer> localMax = new ArrayList<Integer>();
				
		//System.out.println(Long.toString(frameNo)+":"+localMax.toString());
	}
		
	public void process2(Frame<T> frame) {
		//double[] pixels = (double[]) frame.getPixels();
		//float[] pixels = (float[]) frame.getPixels();
		
		Interval interval = Intervals.expand( frame.getPixels(), -1 );
		
		RandomAccessibleInterval<T> source = Views.interval( frame.getPixels(), interval );
		
		final Cursor< T > center = Views.iterable( source ).cursor();
		
		RandomAccess<T> ra = source.randomAccess();
		
		while (center.hasNext()) {
			center.fwd();
			
			double val = center.get().getRealDouble(); 
			if (val >= threshold) {
				
				ra.setPosition(center);
				
				ra.fwd(0); if (val <= ra.get().getRealDouble()) break;
				ra.bck(1); if (val <= ra.get().getRealDouble()) break;
				ra.bck(0); if (val <= ra.get().getRealDouble()) break;
				ra.bck(0); if (val <= ra.get().getRealDouble()) break;
				ra.fwd(1); if (val <= ra.get().getRealDouble()) break;
				ra.fwd(1); if (val <= ra.get().getRealDouble()) break;
				ra.fwd(0); if (val <= ra.get().getRealDouble()) break;
				ra.fwd(0); if (val <= ra.get().getRealDouble()) break;
				
				output.put(new XYFLocalization(frame.getFrameNumber(), center.getIntPosition(0), center.getIntPosition(1)));
			}
		}

	}
	
}
