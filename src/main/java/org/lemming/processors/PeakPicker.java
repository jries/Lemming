package org.lemming.processors;

import java.util.ArrayList;

import net.imglib2.RandomAccess;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.algorithm.math.PickImagePeaks;
import net.imglib2.type.numeric.RealType;
import net.imglib2.util.Intervals;
import net.imglib2.view.IntervalView;
import net.imglib2.view.Views;

import org.lemming.data.XYFLocalization;
import org.lemming.interfaces.Frame;
import org.lemming.interfaces.Localization;

/**
 * @author Ronny Sczech
 *
 * @param <T> - data type
 * @param <F> - frame type
 */
public class PeakPicker<T extends RealType<T>, F extends Frame<T>> extends SingleInputSingleOutput<F,Localization> {

	private boolean hasMoreOutputs = true;
	private float threshold;
	
	/**
	 * @param threshold - threshold
	 */
	public PeakPicker(final float threshold){
		this.threshold = threshold;
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
			System.out.println("Last frame finished:"+frame.getFrameNumber());
			XYFLocalization lastloc = new XYFLocalization(frame.getFrameNumber(), 0, 0);
			lastloc.setLast(true);
			output.put(lastloc);
			hasMoreOutputs  = false;
			stop();
			return;
		}
		if (frame.getFrameNumber() % 500 == 0)
			System.out.println("Frames finished:"+frame.getFrameNumber());
	}

	private void process1(F frame) {
		final RandomAccessibleInterval<T> interval = frame.getPixels();
		final IntervalView<T> source = Views.interval( Views.extendBorder(interval), Intervals.expand( interval, 1));		
		
		PickImagePeaks<T> p = new PickImagePeaks<T>(source);
		ArrayList< long[] > peaks = new ArrayList< long[] >();
		p.setSuppression(5);
		p.setAllowBorderPeak(false);
		if (p.process())
			peaks  = p.getPeakList();
		if (peaks.isEmpty()) return;
		RandomAccess<T> r = source.randomAccess();
		T v = r.get().createVariable();
		v.setReal(threshold);
		for (long[] peak : peaks){
			r.setPosition(peak);
			if (v.compareTo(r.get()) < 0)
				output.put(new XYFLocalization(frame.getFrameNumber(), peak[0], peak[1]));
		}
	}

}
