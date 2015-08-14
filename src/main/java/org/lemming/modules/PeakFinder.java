package org.lemming.modules;

import java.util.ArrayList;
import java.util.List;

import org.lemming.interfaces.Element;
import org.lemming.interfaces.Frame;
import org.lemming.interfaces.Store;
import org.lemming.pipeline.FrameElements;
import org.lemming.pipeline.Localization;
import org.lemming.pipeline.MultiRunModule;
import org.lemming.pipeline.Settings;

import net.imglib2.Cursor;
import net.imglib2.Interval;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.algorithm.neighborhood.Neighborhood;
import net.imglib2.algorithm.neighborhood.RectangleShape;
import net.imglib2.type.numeric.RealType;
import net.imglib2.util.Intervals;
import net.imglib2.view.Views;

public class PeakFinder<T extends RealType<T>, F extends Frame<T>> extends MultiRunModule {

	private int size;
	private double threshold;
	private long start;
	private int counter;
	private Store output;
	@SuppressWarnings("unused")
	private Settings settings;

	/**
	 * @param threshold
	 *            - threshold for subtracting background
	 * @param size
	 *            - kernel size
	 * @param out
	 *            - output store
	 * @param in
	 *            - input store
	 */
	public PeakFinder(Settings settings, final double threshold, final int size) {
		setThreshold(threshold);
		this.size = size;
		this.settings = settings;
	}

	private void setThreshold(double threshold) {
		this.threshold = threshold;
	}

	@Override
	protected void beforeRun() {
		// for this module there should be only one key
		output = outputs.values().iterator().next(); 
		start = System.currentTimeMillis();
	}

	@SuppressWarnings("unchecked")
	@Override
	public Element process(Element data) {

		F frame = (F) data;
		if (frame == null)
			return null;
		
		if (frame.isLast()) { // make the poison pill
			pause(10);
			process1(frame,true);
			cancel();
			return null;
		}
		process1(frame,false);
		return null;
	}

	private void process1(final F frame, boolean b) {
		Interval interval = Intervals.expand(frame.getPixels(), -size);

		RandomAccessibleInterval<T> source = Views.interval(frame.getPixels(), interval);

		final Cursor<T> center = Views.iterable(source).cursor();

		final RectangleShape shape = new RectangleShape(size, true);
		
		List<Element> found = new ArrayList<>();

		for (final Neighborhood<T> localNeighborhood : shape
				.neighborhoods(source)) {
			// what is the value that we investigate?
			// (the center cursor runs over the image in the same iteration
			// order as neighborhood)
			final T centerValue = center.next();

			if (centerValue.getRealDouble() < getThreshold())
				continue;

			// keep this boolean true as long as no other value in the local
			// neighborhood
			// is larger or equal
			boolean isMaximum = true;

			// check if all pixels in the local neighborhood that are smaller
			for (final T value : localNeighborhood) {
				// test if the center is smaller than the current pixel value
				if (centerValue.compareTo(value) <= 0) {
					isMaximum = false;
					break;
				}
			}

			if (isMaximum){
				found.add(new Localization(frame.getFrameNumber(), 
						center.getIntPosition(0), center.getIntPosition(1)));
				counter++; 
			}
		}
		
		if (found.isEmpty()) return;
		
		FrameElements fe = null;
		if (b){
			fe = new FrameElements(found, frame.getFrameNumber());
			fe.setLast(true);
		} else {
			fe = new FrameElements(found, frame.getFrameNumber());
		}
		output.put(fe);
	}

	/**
	 * @return Threshold
	 */
	public double getThreshold() {
		return threshold;
	}

	@Override
	protected void afterRun() {
		System.out.println("PeakFinder found "
				+ counter + " peaks in "
				+ (System.currentTimeMillis() - start) + "ms.");
	}

	@Override
	public boolean check() {
		// TODO Auto-generated method stub
		return false;
	}

}
