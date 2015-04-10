package org.lemming.processors;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import net.imglib2.Cursor;
import net.imglib2.IterableInterval;
import net.imglib2.Point;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.algorithm.dog.DifferenceOfGaussian;
import net.imglib2.algorithm.gauss3.Gauss3;
import net.imglib2.algorithm.localextrema.RefinedPeak;
import net.imglib2.algorithm.localextrema.SubpixelLocalization;
import net.imglib2.exception.IncompatibleTypeException;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Intervals;
import net.imglib2.util.Util;
import net.imglib2.view.ExtendedRandomAccessibleInterval;
import net.imglib2.view.IntervalView;
import net.imglib2.view.Views;

import org.lemming.data.XYFLocalization;
import org.lemming.interfaces.Frame;
import org.lemming.interfaces.Localization;
import org.lemming.processors.LocalExtrema.LocalNeighborhoodCheck;

/**
 * @author Ronny Sczech
 *
 * @param <T> - data type
 * @param <F> - frame type
 */
public class DogDetector<T extends RealType<T>, F extends Frame<T>> extends SingleInputSingleOutput<F, Localization> {

	private double radius;
	private final double[] calibration;
	private float threshold;
	private boolean hasMoreOutputs;
	
	/**
	 * @param radius - estimated feature radius
	 * @param calibration - calibration
	 * @param threshold - threshold for subtracting background
	 */
	public DogDetector(final double radius, final double[] calibration, final float threshold){
		this.calibration = calibration;
		this.radius = radius;
		this.threshold = threshold;
		setNumThreads();
		hasMoreOutputs = true;
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
			hasMoreOutputs = false;
			stop();
			return;
		}
		if (frame.getFrameNumber() % 500 == 0)
			System.out.println("Frames finished:"+frame.getFrameNumber());
	}

	private void process1(Frame<T> frame) {
		
		final RandomAccessibleInterval< T > interval = frame.getPixels();
		final ExtendedRandomAccessibleInterval<T, RandomAccessibleInterval<T>> extended = Views.extendMirrorSingle(interval);
		
		// WE NEED TO SHIFT COORDINATES BY -MIN[] TO HAVE THE CORRECT LOCATION.
		final long[] min = new long[ interval.numDimensions() ];
		interval.min( min );
		for ( int d = 0; d < min.length; d++ )
		{
			min[ d ] = -min[ d ];
		}
		final FloatType type = new FloatType();
		final RandomAccessibleInterval< FloatType > dog = Views.offset( Util.getArrayOrCellImgFactory( interval, type ).create( interval, type ), min );
		final RandomAccessibleInterval< FloatType > dog2 = Views.offset( Util.getArrayOrCellImgFactory( interval, type ).create( interval, type ), min );
	
		final double sigma1 = radius / Math.sqrt( interval.numDimensions() ) * 0.9;
		final double sigma2 = radius / Math.sqrt( interval.numDimensions() ) * 1.1;
		final double[][] sigmas = DifferenceOfGaussian.computeSigmas( 0.5, 2, calibration, sigma1, sigma2 );
		
		try {
			Gauss3.gauss( sigmas[ 1 ], extended, dog2 );
			Gauss3.gauss( sigmas[ 0 ], extended, dog );
		} catch (IncompatibleTypeException e) {
			e.printStackTrace();
		}
		
		final IterableInterval< FloatType > dogIterable = Views.iterable( dog );
		final IterableInterval< FloatType > tmpIterable = Views.iterable( dog2 );
		final Cursor< FloatType > dogCursor = dogIterable.cursor();
		final Cursor< FloatType > tmpCursor = tmpIterable.cursor();
		while ( dogCursor.hasNext() )
			dogCursor.next().sub( tmpCursor.next() );
		
		final FloatType val = new FloatType();
		val.setReal(threshold);
		final LocalNeighborhoodCheck< Point, FloatType > localNeighborhoodCheck = new LocalExtrema.MaximumCheck< FloatType >( val );
		final IntervalView< FloatType > dogWithBorder = Views.interval( Views.extendMirrorSingle( dog ), Intervals.expand( dog, 1 ) );
		/*final ExecutorService service = Executors.newFixedThreadPool( numThreads );
		final List< Point > peaks = LocalExtrema.findLocalExtrema( dogWithBorder, localNeighborhoodCheck, service );
		service.shutdown();*/
		final List< Point > peaks = LocalExtrema.findLocalExtrema( dogWithBorder, localNeighborhoodCheck);
		
		/*final SubpixelLocalization< Point, FloatType > spl = new SubpixelLocalization< Point, FloatType >( dog.numDimensions() );
		//spl.setNumThreads( numThreads );
		spl.setReturnInvalidPeaks( true );
		spl.setCanMoveOutside( true );
		spl.setAllowMaximaTolerance( true );
		spl.setMaxNumMoves( 10 );
		final ArrayList< RefinedPeak< Point >> refined = spl.process( peaks, dogWithBorder, dog );*/
		boolean[] allowedToMoveInDim = new boolean[ dog.numDimensions() ];
		Arrays.fill( allowedToMoveInDim, true );
		final ArrayList< RefinedPeak< Point >> refined = SubpixelLocalization.refinePeaks(peaks, dogWithBorder, dog, true, 10, true, 0.01f, allowedToMoveInDim);
		
		for ( final RefinedPeak< Point > refinedPeak : refined ){
			final double x = refinedPeak.getDoublePosition( 0 ) * calibration[ 0 ];
			final double y = refinedPeak.getDoublePosition( 1 ) * calibration[ 1 ];
			output.put(new XYFLocalization(frame.getFrameNumber(), x, y));
		}
	}

	@Override
	public boolean hasMoreOutputs() {
		return hasMoreOutputs;
	}

}
