package org.lemming.plugins;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.lemming.factories.DetectorFactory;
import org.lemming.gui.ConfigurationPanel;
import org.lemming.gui.PeakFinderPanel;
import org.lemming.interfaces.Element;
import org.lemming.interfaces.Frame;
import org.lemming.modules.Detector;
import org.lemming.pipeline.FrameElements;
import org.lemming.pipeline.Localization;
import org.scijava.plugin.Plugin;

import net.imglib2.Cursor;
import net.imglib2.FinalInterval;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.algorithm.gauss3.Gauss3;
import net.imglib2.algorithm.neighborhood.Neighborhood;
import net.imglib2.algorithm.neighborhood.RectangleShape;
import net.imglib2.exception.IncompatibleTypeException;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Intervals;
import net.imglib2.util.Util;
import net.imglib2.view.ExtendedRandomAccessibleInterval;
import net.imglib2.view.Views;

public class PeakFinder<T extends RealType<T>> extends Detector<T> {

	public static final String NAME = "Peak Finder";

	public static final String KEY = "PEAKFINDER";

	public static final String INFO_TEXT = "<html>" + "Peak Finder Plugin" + "</html>";
	private int size;
	private double threshold;
	private int counter;
	private int gaussian;

	/**
	 * @param threshold
	 *            - threshold for subtracting background
	 * @param size
	 *            - kernel size
	 */
	public PeakFinder(final double threshold, final int size, final int gaussian) {
		setThreshold(threshold);
		this.size = size;
		this.gaussian = gaussian;
	}

	private void setThreshold(double threshold) {
		this.threshold = threshold;
	}

	@Override
	public FrameElements<T> detect(final Frame<T> frame) {
		
		final RandomAccessibleInterval<T> pixels = frame.getPixels();
		double[] sigma = new double[ pixels.numDimensions() ];
		if(gaussian>0){
			for ( int d = 0; d < pixels.numDimensions(); ++d )
	            sigma[ d ] = gaussian;
			final ExtendedRandomAccessibleInterval<T, RandomAccessibleInterval<T>> extended = Views.extendMirrorSingle(pixels);

			// WE NEED TO SHIFT COORDINATES BY -MIN[] TO HAVE THE CORRECT LOCATION.
			final long[] min = new long[pixels.numDimensions()];
			pixels.min(min);
			for (int d = 0; d < min.length; d++) {
				min[d] = -min[d];
			}
			final FloatType type = new FloatType();
			final RandomAccessibleInterval<FloatType> dog = Views.offset(Util.getArrayOrCellImgFactory(pixels, type).create(pixels, type), min);
			try {
				Gauss3.gauss(sigma, extended, dog, 4);
			} catch (IncompatibleTypeException e) {
				e.printStackTrace();
			}
			
			final Cursor<FloatType> dogCursor = Views.iterable(dog).cursor();
			final Cursor<T> tmpCursor = Views.iterable(pixels).cursor();

			while (dogCursor.hasNext()){
				tmpCursor.fwd();
				dogCursor.fwd();
				float val = Math.abs(tmpCursor.get().getRealFloat()-dogCursor.get().getRealFloat());
				tmpCursor.get().setReal(val);
			}
		}
		
		FinalInterval interval = Intervals.expand(pixels, -size);

		RandomAccessibleInterval<T> source = Views.interval(pixels, interval);

		final Cursor<T> center = Views.iterable(source).cursor();

		final RectangleShape shape = new RectangleShape(size, true);

		List<Element> found = new ArrayList<>();

		for (final Neighborhood<T> localNeighborhood : shape.neighborhoods(source)) {
			// what is the value that we investigate?
			// (the center cursor runs over the image in the same iteration
			// order as neighborhood)
			final T centerValue = center.next();
			
			try{
				if (centerValue.getRealDouble() < getThreshold())
					continue;
			}catch(Exception e){
				continue;
			}

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

			if (isMaximum) {
				found.add(new Localization(center.getIntPosition(0) * frame.getPixelDepth(),
						center.getIntPosition(1) * frame.getPixelDepth(),
						centerValue.getRealDouble(),
						frame.getFrameNumber()));
				counter++;
			}
		}
		return new FrameElements<>(found, frame);
	}

	/**
	 * @return Threshold
	 */
	public double getThreshold() {
		return threshold;
	}

	@Plugin(type = DetectorFactory.class, visible = true)
	public static class Factory implements DetectorFactory {

		private Map<String, Object> settings;
		private PeakFinderPanel configPanel = new PeakFinderPanel();

		@Override
		public String getInfoText() {
			return INFO_TEXT;
		}

		@Override
		public String getKey() {
			return KEY;
		}

		@Override
		public String getName() {
			return NAME;
		}

		@Override
		public boolean setAndCheckSettings(Map<String, Object> settings) {
			this.settings = settings;
			return settings != null;
		}

		@Override
		public <T extends RealType<T>> Detector<T> getDetector() {
			final double threshold = (Double) settings.get(PeakFinderPanel.KEY_THRESHOLD);
			final int kernelSize = (Integer) settings.get(PeakFinderPanel.KEY_KERNEL_SIZE);
			final int gaussian = (Integer) settings.get(PeakFinderPanel.KEY_GAUSSIAN_SIZE);
			return new PeakFinder<>(threshold, kernelSize, gaussian);
		}

		@Override
		public ConfigurationPanel getConfigurationPanel() {
			configPanel.setName(KEY);
			return configPanel;
		}

	}
}
