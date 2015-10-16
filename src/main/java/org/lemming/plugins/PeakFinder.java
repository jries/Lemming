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
import org.lemming.pipeline.AbstractModule;
import org.lemming.pipeline.FrameElements;
import org.lemming.pipeline.Localization;
import org.scijava.plugin.Plugin;

import net.imglib2.Cursor;
import net.imglib2.Interval;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.algorithm.neighborhood.Neighborhood;
import net.imglib2.algorithm.neighborhood.RectangleShape;
import net.imglib2.type.numeric.RealType;
import net.imglib2.util.Intervals;
import net.imglib2.view.Views;

public class PeakFinder<T extends RealType<T>, F extends Frame<T>> extends Detector<T,F> {

	public static final String NAME = "Peak Finder";

	public static final String KEY = "PEAKFINDER";

	public static final String INFO_TEXT = "<html>"
											+ "Peak Finder Plugin"
											+ "</html>";
	private int size;
	private double threshold;
	private int counter;

	/**
	 * @param threshold
	 *            - threshold for subtracting background
	 * @param size
	 *            - kernel size
	 */
	public PeakFinder( final double threshold, final int size) {
		setThreshold(threshold);
		this.size = size;
	}

	private void setThreshold(double threshold) {
		this.threshold = threshold;
	}

	
	@Override
	public FrameElements<T> detect(final F frame) {
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
		return new FrameElements<>(found, frame);
	}

	/**
	 * @return Threshold
	 */
	public double getThreshold() {
		return threshold;
	}
	
	@Plugin( type = DetectorFactory.class, visible = true )
	public static class Factory implements DetectorFactory{

		
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
			return settings!=null;
		}

		@SuppressWarnings("rawtypes")
		@Override
		public AbstractModule getDetector() {
			final double threshold = ( Double ) settings.get( PeakFinderPanel.KEY_THRESHOLD );
			final int kernelSize = ( Integer ) settings.get( PeakFinderPanel.KEY_KERNEL_SIZE );
			return new PeakFinder(threshold, kernelSize);
		}

		@Override
		public ConfigurationPanel getConfigurationPanel() {
			configPanel.setName(KEY);
			return configPanel;
		}
		
	}
}
