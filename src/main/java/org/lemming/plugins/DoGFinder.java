package org.lemming.plugins;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import net.imglib2.Cursor;
import net.imglib2.IterableInterval;
import net.imglib2.Localizable;
import net.imglib2.Point;
import net.imglib2.RandomAccess;
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

import org.lemming.factories.DetectorFactory;
import org.lemming.gui.ConfigurationPanel;
import org.lemming.gui.DoGFinderPanel;
import org.lemming.interfaces.Element;
import org.lemming.interfaces.Frame;
import org.lemming.modules.Detector;
import org.lemming.pipeline.FrameElements;
import org.lemming.pipeline.Localization;
import org.scijava.plugin.Plugin;

public class DoGFinder<T extends RealType<T>, F extends Frame<T>> extends Detector<T, F> {

	public static final String NAME = "DoG Finder";

	public static final String KEY = "DOGFINDER";

	public static final String INFO_TEXT = "<html>" + "Difference of Gaussian Finder" + "</html>";
	private double radius;
	private float threshold;
	private double[] calibration;
	private int counter = 0;

	public DoGFinder(final double radius, final float threshold) {
		super();
		this.radius = radius;
		this.threshold = threshold;
		this.calibration = new double[] { 1, 1 };
	}

	@SuppressWarnings("unchecked")
	@Override
	public Element processData(Element data) {
		F frame = (F) data;
		if (frame == null)
			return null;

		if (frame.isLast()) {
			System.out.println("Last frame finished:" + frame.getFrameNumber());
			cancel();
			FrameElements<T> res = detect(frame);
			res.setLast(true);
			return res;
		}
		if (frame.getFrameNumber() % 100 == 0)
			System.out.println("frames: " + frame.getFrameNumber());
		return detect(frame);
	}

	@Override
	public FrameElements<T> detect(F frame) {

		final RandomAccessibleInterval<T> interval = frame.getPixels();
		final ExtendedRandomAccessibleInterval<T, RandomAccessibleInterval<T>> extended = Views.extendMirrorSingle(interval);

		// WE NEED TO SHIFT COORDINATES BY -MIN[] TO HAVE THE CORRECT LOCATION.
		final long[] min = new long[interval.numDimensions()];
		interval.min(min);
		for (int d = 0; d < min.length; d++) {
			min[d] = -min[d];
		}
		final FloatType type = new FloatType();
		final RandomAccessibleInterval<FloatType> dog = Views.offset(Util.getArrayOrCellImgFactory(interval, type).create(interval, type), min);
		final RandomAccessibleInterval<FloatType> dog2 = Views.offset(Util.getArrayOrCellImgFactory(interval, type).create(interval, type), min);

		final double sigma1 = radius / Math.sqrt(interval.numDimensions()) * 0.9;
		final double sigma2 = radius / Math.sqrt(interval.numDimensions()) * 1.1;
		final double[][] sigmas = DifferenceOfGaussian.computeSigmas(0.5, 2, calibration, sigma1, sigma2);

		try {
			Gauss3.gauss(sigmas[1], extended, dog2);
			Gauss3.gauss(sigmas[0], extended, dog);
		} catch (IncompatibleTypeException e) {
			e.printStackTrace();
		}

		final IterableInterval<FloatType> dogIterable = Views.iterable(dog);
		final IterableInterval<FloatType> tmpIterable = Views.iterable(dog2);
		final Cursor<FloatType> dogCursor = dogIterable.cursor();
		final Cursor<FloatType> tmpCursor = tmpIterable.cursor();
		while (dogCursor.hasNext())
			dogCursor.next().sub(tmpCursor.next());

		final FloatType val = new FloatType();
		val.setReal(threshold);
		final MaximumCheck<FloatType> localNeighborhoodCheck = new MaximumCheck<>(val);
		final IntervalView<FloatType> dogWithBorder = Views.interval(Views.extendMirrorSingle(dog), Intervals.expand(dog, 1));
		final List<Point> peaks = findLocalExtrema(dogWithBorder, localNeighborhoodCheck, 1);
		List<Element> found = new ArrayList<>();
		RandomAccess<FloatType> ra = dogWithBorder.randomAccess();

		for (Point p : peaks) {
			double x = p.getDoublePosition(0);
			double y = p.getDoublePosition(1);
			ra.setPosition(p);
			found.add(new Localization(x * frame.getPixelDepth(), y * frame.getPixelDepth(), ra.get().getRealDouble(), frame.getFrameNumber()));
			counter++;
		}

		return new FrameElements<>(found, frame);
	}

	private static <T extends Comparable<T>> ArrayList<Point> findLocalExtrema(final RandomAccessibleInterval<T> img,
			final MaximumCheck<T> localNeighborhoodCheck, int size) {

		final RectangleShape shape = new RectangleShape(size, false);

		final ArrayList<Point> extrema = new ArrayList<>(1);

		final Cursor<T> center = Views.flatIterable(img).cursor();

		for (final Neighborhood<T> neighborhood : shape.neighborhoods(img)) {
			center.fwd();
			final Point p = localNeighborhoodCheck.check(center, neighborhood);
			if (p != null)
				extrema.add(p);
		}

		return extrema;
	}

	private static class MaximumCheck<T extends Comparable<T>> {
		final T minPeakValue;

		/**
		 * @param minPeakValue
		 *            - minimum PeakValue
		 */
		public MaximumCheck(final T minPeakValue) {
			this.minPeakValue = minPeakValue;
		}

		public <C extends Localizable & Sampler<T>> Point check(final C center, final Neighborhood<T> neighborhood) {
			final T c = center.get();

			if (minPeakValue.compareTo(c) > 0)
				return null;

			for (final T t : neighborhood)
				if (t.compareTo(c) > 0)
					return null;

			return new Point(center);
		}
	}

	@Override
	public boolean check() {
		return inputs.size() == 1 && outputs.size() >= 1;
	}

	@Plugin(type = DetectorFactory.class, visible = true)
	public static class Factory implements DetectorFactory {

		private Map<String, Object> settings;
		private DoGFinderPanel configPanel = new DoGFinderPanel();

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
			return true;
		}

		@Override
		public <T extends RealType<T>, F extends Frame<T>> Detector<T, F> getDetector() {
			final double threshold = (Double) settings.get(DoGFinderPanel.KEY_THRESHOLD);
			final int radius = (Integer) settings.get(DoGFinderPanel.KEY_RADIUS);
			return new DoGFinder<>(threshold, radius);
		}

		@Override
		public ConfigurationPanel getConfigurationPanel() {
			configPanel.setName(KEY);
			return configPanel;
		}

	}

}
