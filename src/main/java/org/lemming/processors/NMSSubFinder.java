package org.lemming.processors;

import java.util.ArrayList;

import net.imglib2.Cursor;
import net.imglib2.Point;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.numeric.RealType;
import net.imglib2.util.Intervals;
import net.imglib2.view.IntervalView;
import net.imglib2.view.SubsampleIntervalView;
import net.imglib2.view.Views;

import org.lemming.data.XYFLocalization;
import org.lemming.interfaces.Frame;
import org.lemming.interfaces.Localization;
import org.lemming.processors.LocalExtrema.MaximumFinder;

/**
 * @author Ronny Sczech
 *
 * @param <T> - pixel type
 * @param <F> - frame type
 */
public class NMSSubFinder<T extends RealType<T>, F extends Frame<T>> extends SingleInputSingleOutput<F, Localization> {

	private boolean hasMoreOutputs;
	private double cutoff;
	private int size;
	private long start;
	private ArrayList<Long> tt = new ArrayList<Long>();
	
	/**
	 * @param threshold - minimum threshold for a peak 
	 * @param size - size of the kernel
	 * 
	 */
	public NMSSubFinder(final double threshold, final int size){
		hasMoreOutputs = true;
		this.size = size;
		cutoff = threshold;
		start = System.nanoTime();
	}
	
	@Override
	public boolean hasMoreOutputs() {
		return hasMoreOutputs;
	}

	@Override
	public void process(F frame) {
		if (frame==null) return;
		process1(frame);
		if (frame.isLast()){
			long end = System.nanoTime();
			System.out.println("Last frame finished:"+frame.getFrameNumber()+" in "+(end-start)/1000000+" ms");
			double sum = 0.0;
			for (double value : tt)
				sum += value;
			System.out.println("Time to Subsample: "+ sum/1000000);
			XYFLocalization lastloc = new XYFLocalization(frame.getFrameNumber(), 0, 0);
			lastloc.setLast(true);
			output.put(lastloc);
			hasMoreOutputs = false;
			stop();
			return;
		}
		if (frame.getFrameNumber() % 500 == 0)
			System.out.println("Frames finished:"+frame.getFrameNumber());
	}

	private void process1(F frame) {
		final RandomAccessibleInterval<T> interval = frame.getPixels();
	
		int step = size + 1;
		final SubsampleIntervalView<T> subview = Views.subsample(interval, step); // make a sparse view of the image with a step size
		
		final Cursor<T> center = Views.flatIterable( subview ).cursor();
		final IntervalView<T> withBorder = Views.interval( Views.extendBorder(interval), Intervals.expand( interval, size)); // care for image borders
		MaximumFinder<T> MaxFinder = new MaximumFinder<T>( cutoff ); // set threshold
		while(center.hasNext()){
			center.fwd();
			Point p = MaxFinder.check( withBorder, center, step , size); // the actual peak finder
			if ( p != null ) // add localization	
				output.put(new XYFLocalization(frame.getFrameNumber(), p.getIntPosition(0), p.getIntPosition(1)));
		}				
	}
	
}
