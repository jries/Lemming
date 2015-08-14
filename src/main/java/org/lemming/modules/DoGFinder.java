package org.lemming.modules;

import java.util.ArrayList;
import java.util.List;

import net.imglib2.Cursor;
import net.imglib2.IterableInterval;
import net.imglib2.Localizable;
import net.imglib2.Point;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.Sampler;
import net.imglib2.algorithm.dog.DifferenceOfGaussian;
import net.imglib2.algorithm.gauss3.Gauss3;
import net.imglib2.algorithm.neighborhood.Neighborhood;
import net.imglib2.algorithm.neighborhood.RectangleShape;
import net.imglib2.exception.IncompatibleTypeException;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Intervals;
import net.imglib2.util.Util;
import net.imglib2.view.ExtendedRandomAccessibleInterval;
import net.imglib2.view.IntervalView;
import net.imglib2.view.Views;

import org.lemming.interfaces.Element;
import org.lemming.interfaces.Frame;
import org.lemming.interfaces.Store;
import org.lemming.pipeline.Localization;
import org.lemming.pipeline.MultiRunModule;


public class DoGFinder<T extends RealType<T>, F extends Frame<T>> extends MultiRunModule {

	private double radius;
	private float threshold;
	private Store output;
	private long start;
	private double[] calibration;
	private int counter = 0;

	public DoGFinder(final double radius, final float threshold) {
		this.radius = radius;
		this.threshold = threshold;
		this.calibration = new double[]{1,1};
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
		if (frame==null) return null;
		process1(frame);
		if (frame.isLast()){
			System.out.println("Last frame finished:"+frame.getFrameNumber());
			Localization lastloc = new Localization(frame.getFrameNumber(), -1, -1);
			lastloc.setLast(true);
			output.put(lastloc);
			cancel();
			return null;
		}
		if (frame.getFrameNumber()%50==0)
			System.out.println("frames: " + frame.getFrameNumber());
		return null;
	}

	private void process1(F frame) {
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
		final MaximumCheck< FloatType > localNeighborhoodCheck = new MaximumCheck<>( val );
		final IntervalView< FloatType > dogWithBorder = Views.interval( Views.extendMirrorSingle( dog ), Intervals.expand( dog, 1 ) );
		final List< Point > peaks = findLocalExtrema( dogWithBorder, localNeighborhoodCheck,1);
		
		for (Point p :peaks){
			double x = p.getDoublePosition(0);
			double y = p.getDoublePosition(1);
			output.put(new Localization(frame.getFrameNumber(), x, y));
			counter++;
		}
	}
	
	@Override
	protected void afterRun() {
		System.out.println("DoGFinder found "
				+ counter + " peaks in "
				+ (System.currentTimeMillis() - start) + "ms.");
	}
	
	private static <T extends Comparable<T>> ArrayList<Point> findLocalExtrema( final RandomAccessibleInterval<T> img, final MaximumCheck<T> localNeighborhoodCheck, int size)
	{
		
		final RectangleShape shape = new RectangleShape( size, false );

		final ArrayList< Point > extrema = new ArrayList<>(1);
		
		final Cursor< T > center = Views.flatIterable( img ).cursor();
		
		for ( final Neighborhood< T > neighborhood : shape.neighborhoods( img ) ){
			center.fwd();
			final Point p = localNeighborhoodCheck.check( center, neighborhood );
			if ( p != null )
				extrema.add( p );
		}		
		
		return extrema ;		
	}
	
	private static class MaximumCheck< T extends Comparable< T > >
	{
		final T minPeakValue;
		
		/**
		 * @param minPeakValue - minimum PeakValue
		 */
		public MaximumCheck( final T minPeakValue )
		{
			this.minPeakValue = minPeakValue;
		}
		
		public < C extends Localizable & Sampler< T > > Point check( final C center, final Neighborhood< T > neighborhood )
		{
			final T c = center.get();
			
			if ( minPeakValue.compareTo( c ) > 0 )
				return null;
			
			for ( final T t : neighborhood )
				if ( t.compareTo( c ) > 0 )
					return null;
			
			return new Point( center );
		}
	}

	@Override
	public boolean check() {
		// TODO Auto-generated method stub
		return false;
	}

}
