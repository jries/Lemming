package org.lemming.tests;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.lemming.processors.LocalExtrema;
import org.lemming.processors.LocalExtrema.LocalNeighborhoodCheck;
import org.lemming.processors.LocalExtrema.MaximumFinder;

import net.imglib2.Cursor;
import net.imglib2.IterableInterval;
import net.imglib2.Point;
import net.imglib2.RandomAccess;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.algorithm.dog.DifferenceOfGaussian;
import net.imglib2.algorithm.gauss3.Gauss3;
import net.imglib2.algorithm.region.hypersphere.HyperSphere;
import net.imglib2.exception.IncompatibleTypeException;
import net.imglib2.img.Img;
import net.imglib2.img.ImgFactory;
import net.imglib2.img.array.ArrayImgFactory;
import net.imglib2.img.display.imagej.ImageJFunctions;
import net.imglib2.img.sparse.NtreeImgFactory;
import net.imglib2.type.numeric.integer.ShortType;
import net.imglib2.util.Intervals;
import net.imglib2.util.Util;
import net.imglib2.view.ExtendedRandomAccessibleInterval;
import net.imglib2.view.IntervalView;
import net.imglib2.view.SubsampleIntervalView;
import net.imglib2.view.Views;
import ij.ImageJ;
import io.scif.img.ImgOpener;

@SuppressWarnings("javadoc")
public class ExampleQuadtree
{
	@SuppressWarnings("deprecation")
	final static public void main( final String[] args )
	{
		new ImageJ();

		Img< ShortType > array = null;
		final ImgFactory<ShortType> arrayFactory = new NtreeImgFactory< ShortType >();
		try
		{
			final ImgOpener io = new ImgOpener();
			array = io.openImg( "/home/ronny/Bilder/TubulinAF647-s0001.tif", arrayFactory, new ShortType() );
		}
		catch ( final Exception e )
		{
			e.printStackTrace();
			return;
		}

		//ImageJFunctions.show( array, "array" );

		final NtreeImgFactory< ShortType > ntreeFactory = new NtreeImgFactory< ShortType >();
		final Img< ShortType > quadtree = ntreeFactory.create( array, new ShortType() );
		

		// copy to sparse img
		final Cursor< ShortType > dst = quadtree.localizingCursor();
		final RandomAccess< ShortType > src = array.randomAccess();
		while( dst.hasNext() )
		{
			dst.fwd();
			src.setPosition( dst );
			dst.get().set( src.get() );
		}
		/*
		final RandomAccess< ShortType > dst = quadtree.randomAccess();
		final Cursor< ShortType > src = array.localizingCursor();
		while( src.hasNext() )
		{
			src.fwd();
			dst.setPosition( src );
			dst.get().set( src.get() );
		}
		*/
		
		//ImageJFunctions.show( quadtree, "quadtree" );
		File f = new File("/home/ronny/Bilder/resultsNMS.csv");
		NMS_Detector(array,f);
		File g = new File("/home/ronny/Bilder/resultsDOG.csv");
		DOG_Detector(array,g);
		System.out.println( "done" );
	}
	
	private static void NMS_Detector(RandomAccessibleInterval<ShortType> interval, File f){
		FileWriter w = null;
		try {
			w = new FileWriter(f);
		} catch (IOException e) {
			e.printStackTrace();
		}
				 
		
		int size = 3;
		int step = 4;
		final SubsampleIntervalView<ShortType> subview = Views.subsample(interval, step);
		final Cursor<ShortType> center = Views.flatIterable( subview ).cursor();
		
		final IntervalView<ShortType> withBorder = Views.interval( Views.extendBorder(interval), Intervals.expand( interval, size));
		
		Img<ShortType> output = new NtreeImgFactory< ShortType >().create(interval, new ShortType());
				
		final ArrayList< Point > extrema = new ArrayList< Point >(1);
		MaximumFinder<ShortType> MaxFinder = new MaximumFinder<ShortType>(700);
		
		while(center.hasNext()){
			center.fwd();
			Point p = MaxFinder.check( withBorder, center, step , size);
			//ShortType c = MaxFinder.getMax();
			
			if ( p != null ){
				extrema.add( p );
				//System.out.println("x: "+ p.getIntPosition(0) +" y: "+ p.getIntPosition(1) +" : " + c);
				HyperSphere<ShortType> hyperSphere = new HyperSphere< ShortType >( output, p, 1 );
                for ( ShortType value : hyperSphere )
                    value.setInteger(5000);
				String out = String.format("%d\t %d\n",p.getIntPosition(0),p.getIntPosition(1));
				try {
					w.write(out);
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}
		try {
			w.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		ImageJFunctions.show( interval, "NMS" );
		ImageJFunctions.show( output, "maxima" );
	}
	
	private static void DOG_Detector(RandomAccessibleInterval<ShortType> interval, File f){
		
		FileWriter w = null;
		try {
			w = new FileWriter(f);
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		double[] calibration = Util.getArrayFromValue( 1d, 2 );
		int radius = 5;
		int threshold = 20;
		final ExtendedRandomAccessibleInterval<ShortType, RandomAccessibleInterval<ShortType>> extended = Views.extendMirrorSingle(interval);
		
		// WE NEED TO SHIFT COORDINATES BY -MIN[] TO HAVE THE CORRECT LOCATION.
		final long[] min = new long[ interval.numDimensions() ];
		interval.min( min );
		for ( int d = 0; d < min.length; d++ )
		{
			min[ d ] = -min[ d ];
		}
		final ShortType type = new ShortType();
		final RandomAccessibleInterval< ShortType > dog = Views.offset( Util.getArrayOrCellImgFactory( interval, type ).create( interval, type ), min );
		final RandomAccessibleInterval< ShortType > dog2 = Views.offset( Util.getArrayOrCellImgFactory( interval, type ).create( interval, type ), min );
	
		final double sigma1 = radius / Math.sqrt( interval.numDimensions() ) * 0.9;
		final double sigma2 = radius / Math.sqrt( interval.numDimensions() ) * 1.1;
		final double[][] sigmas = DifferenceOfGaussian.computeSigmas( 0.5, 2, calibration, sigma1, sigma2 );
		
		try {
			Gauss3.gauss( sigmas[ 1 ], extended, dog2 );
			Gauss3.gauss( sigmas[ 0 ], extended, dog );
		} catch (IncompatibleTypeException e) {
			e.printStackTrace();
		}
		
		final IterableInterval< ShortType > dogIterable = Views.iterable( dog );
		final IterableInterval< ShortType > tmpIterable = Views.iterable( dog2 );
		final Cursor< ShortType > dogCursor = dogIterable.cursor();
		final Cursor< ShortType > tmpCursor = tmpIterable.cursor();
		while ( dogCursor.hasNext() )
			dogCursor.next().sub( tmpCursor.next() );
		
		final ShortType val = new ShortType();
		val.setReal(threshold);
		final LocalNeighborhoodCheck< Point, ShortType > localNeighborhoodCheck = new LocalExtrema.MaximumCheck< ShortType >( val );
		final IntervalView< ShortType > dogWithBorder = Views.interval( Views.extendMirrorSingle( dog ), Intervals.expand( dog, 1 ) );
		/*final ExecutorService service = Executors.newFixedThreadPool( numThreads );
		final List< Point > peaks = LocalExtrema.findLocalExtrema( dogWithBorder, localNeighborhoodCheck, service );
		service.shutdown();*/
		final List< Point > peaks = LocalExtrema.findLocalExtrema( dogWithBorder, localNeighborhoodCheck, 1);
		for (Point p : peaks){
			//System.out.println("x: "+ p.getIntPosition(0) +" y: "+ p.getIntPosition(1) +" : ");
			String out = String.format("%d\t %d\n",p.getIntPosition(0),p.getIntPosition(1));
			try {
				w.write(out);
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		try {
			w.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}