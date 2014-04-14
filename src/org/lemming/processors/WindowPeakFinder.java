package org.lemming.processors;

import net.imglib2.Cursor;
import net.imglib2.Interval;
import net.imglib2.RandomAccess;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.numeric.RealType;
import net.imglib2.util.Intervals;
import net.imglib2.view.Views;

import org.lemming.data.Frame;
import org.lemming.data.XYFLocalization;
import org.lemming.data.XYFwLocalization;

public class WindowPeakFinder<T extends RealType<T>, F extends Frame<T>> extends PeakFinder<T, F> {

	public WindowPeakFinder(double threshold) {
		super(threshold);
	}
	
	int size = 1;	

	@Override
	public Array<Localization> process(F frame) {
		float[] pixels =  new float[9];
		
		Interval interval = Intervals.expand( frame.getPixels(), -1 );
		
		RandomAccessibleInterval<T> source = Views.interval( frame.getPixels(), interval );
		
		final Cursor< T > center = Views.iterable( source ).cursor();
		
		RandomAccess<T> ra = source.randomAccess();
		
                Array<Localization> result;
		while (center.hasNext()) {
			center.fwd();
			
			double val = center.get().getRealDouble(); 
			if (val >= threshold) {
				
				ra.setPosition(center);
				
				/**
				 *  0 | 1 | 2
				 * -----------
				 *  3 | 4 | 5
				 * -----------
				 *  6 | 7 | 8 
				 */

				float v;
				ra.fwd(0); v = ra.get().getRealFloat(); pixels[5] = v; if (val <= v) continue;
				ra.bck(1); v = ra.get().getRealFloat(); pixels[2] = v; if (val <= v) continue;
				ra.bck(0); v = ra.get().getRealFloat(); pixels[1] = v; if (val <= v) continue;
				ra.bck(0); v = ra.get().getRealFloat(); pixels[0] = v; if (val <= v) continue;
				ra.fwd(1); v = ra.get().getRealFloat(); pixels[3] = v; if (val <= v) continue;
				ra.fwd(1); v = ra.get().getRealFloat(); pixels[6] = v; if (val <= v) continue;
				ra.fwd(0); v = ra.get().getRealFloat(); pixels[7] = v; if (val <= v) continue;
				ra.fwd(0); v = ra.get().getRealFloat(); pixels[8] = v; if (val <= v) continue;
				pixels[4] = (float) val;
				
				result.put(new XYFwLocalization(pixels, frame.getFrameNumber(), center.getIntPosition(0), center.getIntPosition(1)));
			}
		}
                return result;

	}
}
