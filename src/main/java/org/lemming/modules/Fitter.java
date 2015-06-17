package org.lemming.modules;

import ij.gui.Roi;
import ij.process.ImageProcessor;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import net.imglib2.FinalInterval;
import net.imglib2.RandomAccessible;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.RealPoint;
import net.imglib2.histogram.Histogram1d;
import net.imglib2.histogram.Integer1dBinMapper;
import net.imglib2.img.display.imagej.ImageJFunctions;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.integer.IntType;
import net.imglib2.view.Views;

import org.lemming.math.CentroidFitter;
import org.lemming.math.FindThreshold;
import org.lemming.math.FitterType;
import org.lemming.math.GaussianFitter;
import org.lemming.math.GaussianFitterAlternative;
import org.lemming.math.SubpixelLocalization;
import org.lemming.math.ThresholdingType;
import org.lemming.pipeline.Element;
import org.lemming.pipeline.FittedLocalization;
import org.lemming.pipeline.Frame;
import org.lemming.pipeline.Localization;
import org.lemming.pipeline.Module;
import org.lemming.pipeline.Store;

public class Fitter extends Module {

	private FitterType ftype;
	private long start;
	private Store output;
	private long size;

	public Fitter(final FitterType ftype, long size) {
		this.ftype = ftype;
		this.size = size;
	}

	public void setIterator(String iterator) {
		this.iterator = iterator;
	}

	@Override
	protected void beforeRun() {
		// this module has one output
		output = outputs.values().iterator().next();
		start = System.currentTimeMillis();
	}

	@SuppressWarnings({ "rawtypes" })
	@Override
	public Element process(Element data) {
		Frame frame = (Frame) data;
		if (frame == null)
			return null;

		if (frame.isLast()) {
			if (!inputs.get(iterator).isEmpty()){
				inputs.get(iterator).put(frame);
				
				return null;
			}
			process1(frame);
			FittedLocalization lastLoc = new FittedLocalization(frame.getFrameNumber(), -1, -1, 0, -1, -1) ;
			lastLoc.setLast(true);
			output.put(lastLoc);
			cancel();
		}
		
		process1(frame);
		return null;
	}

	@SuppressWarnings({ "rawtypes", "unchecked" })
	private void process1(Frame frame) {
	final RandomAccessibleInterval pixels = frame.getPixels();

	final long max = Views.iterable(frame.getPixels()).firstElement().getClass()
			.getSimpleName() == "UnsignedByte" ? 255 : 65535;

	FindThreshold ft = new FindThreshold(ThresholdingType.OTSU,
			new IntType());
	RealType thresh = (RealType) ft.compute(new Histogram1d(Views
			.iterable(pixels), new Integer1dBinMapper(0, max, true)));

	final RandomAccessible ra = Views.extendBorder(pixels);

	List<Element> sliceLocs = inputs
			.get("locs")
			.view()
			.stream()
			.filter(loc -> ((Localization) loc).getFrame() == frame
					.getFrameNumber()).collect(Collectors.toList());

	if (sliceLocs.isEmpty())
		return;
	
	System.out.println("frame:" + frame.getFrameNumber() + " - localizations:" + sliceLocs.size() );

	if (ftype == FitterType.CENTROID) {
		for (Element el : sliceLocs) {
			final Localization loc = (Localization) el;
			final FinalInterval interval = new FinalInterval(
					new long[] { (long) (loc.getX() - size),
							(long) (loc.getY() - size) }, 
					new long[] { (long) (loc.getX() + size),
							(long) (loc.getY() + size) });

			CentroidFitter cf = new CentroidFitter(Views.interval(ra,
					interval), thresh.getRealDouble());
			double[] result = null;
			result = cf.fit();
			if (result!= null)
				output.put(new FittedLocalization(loc.getID(),loc.getFrame(), result[0], result[1], 0, result[2], result[3]));			
		}
	} else if (ftype == FitterType.QUADRATIC){
		final boolean[] allowedToMoveInDim = new boolean[ ra.numDimensions() ];
		Arrays.fill( allowedToMoveInDim, true );
		
		final List<FittedLocalization> refined = SubpixelLocalization.refinePeaks(sliceLocs, ra, pixels, true, (int) size, true, 0.01f, allowedToMoveInDim);
		for ( final FittedLocalization loc : refined )
			output.put(loc);
	} else if (ftype == FitterType.ELLIPTICALGAUSSIANALTERNATIVE){ // Jorans Fitter
		ImageProcessor ip = ImageJFunctions.wrap(pixels,"").getProcessor();
		for (Element el : sliceLocs) {
			final Localization loc = (Localization) el;
			
			final Roi origroi = new Roi(loc.getX() - size, loc.getY() - size, 2*size+1, 2*size+1);
			final Roi roi = new Roi(ip.getRoi().intersection(origroi.getBounds()));
			
			GaussianFitterAlternative gfa = new GaussianFitterAlternative(ip, roi, 3000, 1000);
			double[] result = null;
			result = gfa.fit();
			if (result!= null)
				output.put(new FittedLocalization(loc.getID(),loc.getFrame(), result[0], result[1], 0, result[2], result[3]));	
		}		
	} else if (ftype == FitterType.ELLIPTICALGAUSSIAN){ // Ronnys Fitter
		final double[] sigmas = new double[ pixels.numDimensions() ];
		Arrays.fill(sigmas, size);
		GaussianFitter gf = new GaussianFitter(pixels, sigmas);
		for (Element el : sliceLocs) {
			final Localization loc = (Localization) el;
			gf.setPoint(new RealPoint(new double[]{loc.getX(),loc.getY()}));
			double[] result = null;
			result = gf.fit();
			if (result!= null)
				output.put(new FittedLocalization(loc.getID(),loc.getFrame(), result[1], result[2], result[0], result[3], result[4]));	
		}		
	} else
		return; 
	}
	

	@Override
	protected void afterRun() {
		System.out.println("Fitting done in "
				+ (System.currentTimeMillis() - start) + "ms.");
	}

}
