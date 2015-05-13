package org.lemming.modules;

import java.util.Map;

import org.lemming.pipeline.Element;
import org.lemming.pipeline.Frame;
import org.lemming.pipeline.Localization;
import org.lemming.pipeline.Module;

import net.imglib2.Cursor;
import net.imglib2.Interval;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.algorithm.neighborhood.Neighborhood;
import net.imglib2.algorithm.neighborhood.RectangleShape;
import net.imglib2.type.numeric.RealType;
import net.imglib2.util.Intervals;
import net.imglib2.view.Views;

public class PeakFinder<T extends RealType<T>, F extends Frame<T>> extends Module {
	
	private int size;
	private double threshold;
	private String outputKey;
	private String inputKey;

	/**
	 * @param threshold - threshold for subtracting background
	 * @param size - kernel size 
	 * @param out - output store
	 * @param in - input store
	 */
	public PeakFinder(final double threshold, final int size, final String in, final String out) {
		System.currentTimeMillis();
		setThreshold(threshold);
		this.size = size;
		outputKey = out;
		inputKey = in;
		setNumThreads();
 	}

	private void setThreshold(double threshold) {
		this.threshold = threshold;
	}

	@SuppressWarnings("unchecked")
	@Override
	public void process(Map<String, Element> data) {
		
		F frame = (F) data.get(inputKey);
		if (frame==null) return;
		
		process1(frame, data);
		if (frame.isLast()){ // make the poison pill
			System.out.println("Last frame finished:"+frame.getFrameNumber());
			Localization lastloc = new Localization(frame.getFrameNumber(), 0, 0);
			lastloc.setLast(true);
			data.put(outputKey,lastloc);
			cancel();
			return;
		}
		if (frame.getFrameNumber() % 100 == 0)
			System.out.println("Frames finished:"+frame.getFrameNumber());
		
	}
	
	private void process1(final F frame, Map<String, Element> data) {
		Interval interval = Intervals.expand( frame.getPixels(), -size );
		
		RandomAccessibleInterval<T> source = Views.interval( frame.getPixels(), interval );
		
		final Cursor< T > center = Views.iterable( source ).cursor();

		final RectangleShape shape = new RectangleShape( size, true );

		for ( final Neighborhood< T > localNeighborhood : shape.neighborhoods( source ) )
		{
		    // what is the value that we investigate?
		    // (the center cursor runs over the image in the same iteration order as neighborhood)
		    final T centerValue = center.next();
		    
		    if (centerValue.getRealDouble() < getThreshold()) 
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
		    	data.put(outputKey,new Localization(frame.getFrameNumber(), center.getIntPosition(0), center.getIntPosition(1)));
		}
	}

	/**
	 * @return Threshold
	 */
	public double getThreshold() {
		return threshold;
	}

}
