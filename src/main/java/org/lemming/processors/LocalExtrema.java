package org.lemming.processors;

import java.util.ArrayList;

import net.imglib2.Cursor;
import net.imglib2.Localizable;
import net.imglib2.Point;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.Sampler;
import net.imglib2.algorithm.region.localneighborhood.HyperSphereShape;
import net.imglib2.algorithm.region.localneighborhood.Neighborhood;
import net.imglib2.view.Views;

/**
* Provides findLocalExtrema(RandomAccessibleInterval, LocalNeighborhoodCheck, int)
* to find pixels that are extrema in their local neighborhood.
*
* @author Tobias Pietzsch
* @author Ronny Sczech
*/


public class LocalExtrema {
	/**
	* A local extremum check.
	*
	* @param <P>
	* A representation of the extremum. For example, this could be
	* just a {@link Point} describing the location of the extremum.
	* It could contain additional information such as the value at
	* the extremum or an extremum type.
	* @param <T>
	* pixel type.
	*/
	
	public interface LocalNeighborhoodCheck< P, T extends Comparable< T > >
	{
	/**
	* Determine whether a pixel is a local extremum. If so, return a
	* <code>P</code> that represents the maximum. Otherwise return
	* <code>null</code>.
	*
	* @param center
	* an access located on the pixel to test
	* @param neighborhood
	* iterable neighborhood of the pixel, not containing the
	* pixel itself.
	* @param <C> - Center Type
	* @return null if the center not a local extremum, a P if it is.
	*/
	public < C extends Localizable & Sampler< T > > P check( C center, Neighborhood< T > neighborhood );
	}
	
	/**
	* A {@link LocalNeighborhoodCheck} to test whether a pixel is a local
	* maximum. A pixel is considered a maximum if its value is greater than or
	* equal to a specified minimum allowed value, and no pixel in the
	* neighborhood has a greater value. That means that maxima are non-strict.
	* Intensity plateaus may result in multiple maxima.
	*
	* @param <T>
	* pixel type.
	*
	* @author Tobias Pietzsch
	*/
	public static class MaximumCheck< T extends Comparable< T > > implements LocalNeighborhoodCheck< Point, T >
	{
		final T minPeakValue;
		
		/**
		 * @param minPeakValue - minimum PeakValue
		 */
		public MaximumCheck( final T minPeakValue )
		{
			this.minPeakValue = minPeakValue;
		}
		
		@Override
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
	
	/**
	* A {@link LocalNeighborhoodCheck} to test whether a pixel is a local
	* minimum. A pixel is considered a minimum if its value is less than or
	* equal to a specified maximum allowed value, and no pixel in the
	* neighborhood has a smaller value. That means that minima are non-strict.
	* Intensity plateaus may result in multiple minima.
	*
	* @param <T>
	* pixel type.
	*
	* @author Tobias Pietzsch
	*/
	public static class MinimumCheck< T extends Comparable< T > > implements LocalNeighborhoodCheck< Point, T >
	{
		final T maxPeakValue;
		
		/**
		 * @param maxPeakValue - maximal PeakValue
		 */
		public MinimumCheck( final T maxPeakValue )
		{
			this.maxPeakValue = maxPeakValue;
		}
		
		@Override
		public < C extends Localizable & Sampler< T > > Point check( final C center, final Neighborhood< T > neighborhood )
		{
			final T c = center.get();
			
			if ( maxPeakValue.compareTo( c ) < 0 )
				return null;
			
			for ( final T t : neighborhood )
				if ( t.compareTo( c ) < 0 )
					return null;
			
			return new Point( center );
		}
	}
	
	/**
	 * @param img - Image
	 * @param localNeighborhoodCheck - Neighborhood check
	 * @param <P> - Neighborhood Type
	 * @param <T> - Pixel Type
	 * @return local Extrema
	 */
	public static < P, T extends Comparable< T > > ArrayList< P > findLocalExtrema( final RandomAccessibleInterval< T > img, final LocalNeighborhoodCheck< P, T > localNeighborhoodCheck)
	{
		final HyperSphereShape shape = new HyperSphereShape( 1 );
		
		final ArrayList< P > extrema = new ArrayList< P >(1);
		
		final Cursor< T > center = Views.flatIterable( img ).cursor();
		
		for ( final Neighborhood< T > neighborhood : shape.neighborhoods( img ) ){
			center.fwd();
			final P p = localNeighborhoodCheck.check( center, neighborhood );
			if ( p != null )
				extrema.add( p );
		}		
		
		return extrema ;		
	}
}
