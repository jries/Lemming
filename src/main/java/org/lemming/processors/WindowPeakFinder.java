package org.lemming.processors;

import net.imglib2.Cursor;
import net.imglib2.Interval;
import net.imglib2.RandomAccess;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.numeric.RealType;
import net.imglib2.util.Intervals;
import net.imglib2.view.Views;

import org.lemming.data.XYFLocalization;
import org.lemming.data.XYFwLocalization;
import org.lemming.interfaces.Frame;

/**
 * @author Ronny Sczech
 *
 * @param <T> - data type
 * @param <F> - frame type
 */
public class WindowPeakFinder<T extends RealType<T>, F extends Frame<T>> extends PeakFinder<T, F> {

	/**
	 * @param threshold - threshold for subtracting background
	 */
	public WindowPeakFinder(double threshold) {
		super(threshold);
	}

	@Override
	public void process(F frame) {
		if (frame==null) return;
		process3(frame);
		if (frame.isLast()){
			System.out.println("Last frame finished:"+frame.getFrameNumber());
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

	private void process3(F frame) {
		float[] pixels =  new float[9];
		
		Interval interval = Intervals.expand( frame.getPixels(), -1 );
		
		RandomAccessibleInterval<T> source = Views.interval( frame.getPixels(), interval );
		
		final Cursor< T > center = Views.iterable( source ).cursor();
		
		RandomAccess<T> ra = source.randomAccess();
		
		while (center.hasNext()) {
			center.fwd();
			
			double val = center.get().getRealDouble(); 
			if (val >= getThreshold()) {
				
				ra.setPosition(center);

				float v;
				ra.fwd(0); v = ra.get().getRealFloat(); pixels[5] = v; if (val <= v) break;
				ra.bck(1); v = ra.get().getRealFloat(); pixels[2] = v; if (val <= v) break;
				ra.bck(0); v = ra.get().getRealFloat(); pixels[1] = v; if (val <= v) break;
				ra.bck(0); v = ra.get().getRealFloat(); pixels[0] = v; if (val <= v) break;
				ra.fwd(1); v = ra.get().getRealFloat(); pixels[3] = v; if (val <= v) break;
				ra.fwd(1); v = ra.get().getRealFloat(); pixels[6] = v; if (val <= v) break;
				ra.fwd(0); v = ra.get().getRealFloat(); pixels[7] = v; if (val <= v) break;
				ra.fwd(0); v = ra.get().getRealFloat(); pixels[8] = v; if (val <= v) break;
				pixels[4] = (float) val;
				
				output.put(new XYFwLocalization(pixels, frame.getFrameNumber(), center.getIntPosition(0), center.getIntPosition(1)));
			}
		}
	}
}
