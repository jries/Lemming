package org.lemming.plugins;

import java.util.ArrayDeque;
import java.util.List;
import java.util.Map;
import java.util.Queue;

import org.lemming.factories.DetectorFactory;
import org.lemming.gui.ConfigurationPanel;
import org.lemming.gui.FastMedianPanel;
import org.lemming.interfaces.Detector;
import org.lemming.interfaces.Element;
import org.lemming.interfaces.Frame;
import org.lemming.interfaces.PreProcessor;
import org.lemming.math.QuickSelect;
import org.lemming.pipeline.FrameElements;
import org.lemming.pipeline.ImgLib2Frame;
import org.lemming.pipeline.Localization;
import org.lemming.pipeline.SingleRunModule;
import org.lemming.tools.LemmingUtils;
import org.scijava.plugin.Plugin;

import javolution.util.FastTable;
import net.imglib2.Cursor;
import net.imglib2.FinalInterval;
import net.imglib2.RandomAccess;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.algorithm.neighborhood.Neighborhood;
import net.imglib2.algorithm.neighborhood.RectangleShape;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgFactory;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.util.Intervals;
import net.imglib2.view.IntervalView;
import net.imglib2.view.Views;

public class NMSFastMedian<T extends RealType<T> & NativeType<T>> extends SingleRunModule implements PreProcessor<T> {
	
	private static final String NAME = "NMS Fast Median Filter";
	private static final String KEY = "NMSFASTMEDIAN";
	private static final String INFO_TEXT = "<html>" + "NMS detector with Fast Median Filter using a 3x3 kernel times the given frames for calculating an approximate median."
			+ "It has an option to interpolate between blocks" + "</html>";
	private final int nFrames;
	private final boolean interpolating;
	private final Queue<Frame<T>> frameList = new ArrayDeque<>();
	private int counter = 0;
	private int lastListSize = 0;
	private Frame<T> frameA = null;
	private Frame<T> frameB = null;
	private final double threshold;
	private final int n_;

	public NMSFastMedian(final int numFrames, final boolean interpolating, final double threshold, final int size) {
		this.nFrames = numFrames;
		this.interpolating = interpolating;
		this.threshold = threshold;
		this.n_ = size;
	}

	@Override
	public boolean check() {
		return inputs.size() == 1 && outputs.size() >= 1;
	}

	@SuppressWarnings("unchecked")
	@Override
	public Element processData(Element data) {
		final Frame<T> frame = (Frame<T>) data;

		frameList.add(frame);
		counter++;

		if (frame.isLast()) {// process the rest;
			Queue<Frame<T>> transferList = new ArrayDeque<>();
			transferList.addAll(frameList);
			frameB = preProcess(transferList, true);
			if (interpolating) 
				if (frameA != null) 
					interpolate(transferList);
			running = false;
			lastListSize = frameList.size() - 1;
			lastFrames(transferList);
			frameList.clear();
			return null;
		}

		if (counter % nFrames == 0) {// make a new list for each Callable
			final Queue<Frame<T>> transferList = new ArrayDeque<>();
			transferList.addAll(frameList);
			frameB = preProcess(transferList, false);
			if (interpolating) {
				if (frameA != null) 
					interpolate(transferList);
				frameA = frameB;
			} else {
				for (int i = 0; i < nFrames; i++) {
					final Frame<T> filtered = LemmingUtils.substract(frameB, transferList.poll());
					newOutput(detect(filtered));
				}
			}
			frameList.clear();
		}
		return null;
	}
	
	public Frame<T> preProcess(final Queue<Frame<T>> list, final boolean isLast) {
		Frame<T> newFrame;
		if (!list.isEmpty()) {
			final Frame<T> firstFrame = list.peek();
			final RandomAccessibleInterval<T> firstInterval = firstFrame.getPixels(); // handle borders
			final long[] dims = new long[firstInterval.numDimensions()];
			firstInterval.dimensions(dims);
			Img<T> out = new ArrayImgFactory<T>().create(dims, Views.iterable(firstInterval).firstElement());

			final FinalInterval shrinked = Intervals.expand(out, -1); // handle borders
			final IntervalView<T> source = Views.interval(out, shrinked);
			final RectangleShape outshape = new RectangleShape(1, false); // 3x3 kernel
			Cursor<T> outcursor = Views.iterable(source).cursor();

			List<RandomAccess<T>> cursorList = new FastTable<>();

			for (Frame<T> currentFrame : list) {
				RandomAccessibleInterval<T> currentInterval = currentFrame.getPixels();
				cursorList.add(currentInterval.randomAccess()); // creating neighborhoods
			}

			for (final Neighborhood<T> localNeighborhood : outshape.neighborhoods(source)) {
				outcursor.fwd();
				final Cursor<T> localCursor = localNeighborhood.cursor();
				final List<Double> values = new FastTable<>();
				while (localCursor.hasNext()) {
					localCursor.fwd();
					for (RandomAccess<T> currentCursor : cursorList) {
						currentCursor.setPosition(localCursor);
						values.add(currentCursor.get().getRealDouble());
					}
				}
				final Double median = QuickSelect.fastmedian(values, values.size()); // find the median
				if (median != null) outcursor.get().setReal(median);
			}

			// Borders
			final Cursor<T> top = Views.interval(out, Intervals.createMinMax(0, 0, dims[0] - 1, 0)).cursor();
			while (top.hasNext()) {
				final List<Double> values = new FastTable<>();
				top.fwd();
				for (RandomAccess<T> currentCursor : cursorList) {
					currentCursor.setPosition(top);
					values.add(currentCursor.get().getRealDouble());
				}
				final Double median = QuickSelect.fastmedian(values, values.size()); // find the median
				if (median != null) top.get().setReal(median);
			}
			final Cursor<T> left = Views.interval(out, Intervals.createMinMax(0, 1, 0, dims[1] - 2)).cursor();
			while (left.hasNext()) {
				final List<Double> values = new FastTable<>();
				left.fwd();
				for (RandomAccess<T> currentCursor : cursorList) {
					currentCursor.setPosition(left);
					values.add(currentCursor.get().getRealDouble());
				}
				final Double median = QuickSelect.fastmedian(values, values.size()); // find the median
				if (median != null) left.get().setReal(median);
			}
			final Cursor<T> right = Views.interval(out, Intervals.createMinMax(dims[0] - 1, 1, dims[0] - 1, dims[1] - 2)).cursor();
			while (right.hasNext()) {
				final List<Double> values = new FastTable<>();
				right.fwd();
				for (RandomAccess<T> currentCursor : cursorList) {
					currentCursor.setPosition(right);
					values.add(currentCursor.get().getRealDouble());
				}
				final Double median = QuickSelect.fastmedian(values, values.size()); // find the median
				if (median != null) right.get().setReal(median);
			}
			final Cursor<T> bottom = Views.interval(out, Intervals.createMinMax(0, dims[1] - 1, dims[0] - 1, dims[1] - 1)).cursor();
			while (bottom.hasNext()) {
				final List<Double> values = new FastTable<>();
				bottom.fwd();
				for (RandomAccess<T> currentCursor : cursorList) {
					currentCursor.setPosition(bottom);
					values.add(currentCursor.get().getRealDouble());
				}
				final Double median = QuickSelect.fastmedian(values, values.size()); // find the median
				if (median != null) bottom.get().setReal(median);
			}

			newFrame = new ImgLib2Frame<>(
				firstFrame.getFrameNumber(), firstFrame.getWidth(), firstFrame.getHeight(), firstFrame.getPixelDepth(), out);
		} else {
			newFrame = new ImgLib2Frame<>(0, 1, 1, 1, null);
		}
		if (isLast) newFrame.setLast(true);
		return newFrame;
	}
	
	/**
	 * interpolate between blocks
	 */
	private void interpolate(Queue<Frame<T>> transferList) {
		RandomAccessibleInterval<T> intervalA = frameA.getPixels();
		RandomAccessibleInterval<T> intervalB = frameB.getPixels();

		for (int i = 0; i < nFrames; i++) {
			Img<T> outFrame = new ArrayImgFactory<T>().create(intervalA, Views.iterable(intervalA).firstElement());
			Cursor<T> outCursor = outFrame.cursor();
			Cursor<T> cursorA = Views.iterable(intervalA).cursor();
			Cursor<T> cursorB = Views.iterable(intervalB).cursor();

			while (cursorA.hasNext()) {
				cursorA.fwd();
				cursorB.fwd();
				outCursor.fwd();
				Double newValue = cursorA.get().getRealDouble()
					+ Math.round((cursorB.get().getRealDouble() - cursorA.get().getRealDouble()) * ((double) i + 1) / nFrames);
				outCursor.get().setReal(newValue);
			}

			final Frame<T> filtered = LemmingUtils.substract(
					new ImgLib2Frame<>(frameA.getFrameNumber() + i, frameA.getWidth(), frameA.getHeight(), frameA.getPixelDepth(), outFrame), transferList.poll());

			newOutput(detect(filtered));
		}
	}
	
	private void lastFrames(Queue<Frame<T>> transferList) {
		// handle the last frames
		for (int i = 0; i < lastListSize; i++) {
			final Frame<T> filtered = LemmingUtils.substract(frameB, transferList.poll());
			newOutput(detect(filtered));
		}

		// create last frame
		Frame<T> lastFrame = LemmingUtils.substract(frameB,transferList.poll() );
		lastFrame.setLast(true);
		newOutput(detect(lastFrame));
	}
	
	
	public FrameElements<T> detect(Frame<T> frame) {
		final RandomAccessibleInterval<T> interval = frame.getPixels();
		final RandomAccess<T> ra = interval.randomAccess();

		// compute max of the Image
		final T max = LemmingUtils.computeMax(Views.iterable(interval));
		double threshold_ = max.getRealDouble() / 100.0 * threshold;

		int i, j, ii, jj, ll, kk;
		int mi, mj;
		boolean failed;
		long width_ = interval.dimension(0);
		long height_ = interval.dimension(1);
		List<Element> found = new FastTable<>();
		T first,second = max,third;

		for (i = 0; i <= width_ - 1 - n_; i += n_ + 1) { // Loop over (n+1)x(n+1)
			for (j = 0; j <= height_ - 1 - n_; j += n_ + 1) {
				mi = i;
				mj = j;
				for (ii = i; ii <= i + n_; ii++) {
					for (jj = j; jj <= j + n_; jj++) {
						ra.setPosition(new int[] { ii, jj });
						first = ra.get().copy();
						ra.setPosition(new int[] { mi, mj });
						second = ra.get().copy();
						if (first.compareTo(second) > 0) {
							mi = ii;
							mj = jj;
						}
					}
				}
				failed = false;

				Outer: for (ll = mi - n_; ll <= mi + n_; ll++) {
					for (kk = mj - n_; kk <= mj + n_; kk++) {
						if ((ll < i || ll > i + n_) || (kk < j || kk > j + n_)) {
							if (ll < width_ && ll > 0 && kk < height_ && kk > 0) {
								ra.setPosition(new int[] { ll, kk });
								third = ra.get().copy();
								if (third.compareTo(second) > 0) {
									failed = true;
									break Outer;
								}
							}
						}
					}
				}
				if (!failed) {
					ra.setPosition(new int[] { mi, mj });
					final T value = ra.get();
					if (value.getRealDouble() > threshold_) {
						found.add(new Localization(mi * frame.getPixelDepth(), mj * frame.getPixelDepth(), value.getRealDouble(), frame
							.getFrameNumber()));
						counter++;
					}
				}
			}
		}

		return new FrameElements<>(found, frame);
	}	
	
	@Override
	public void afterRun(){
		System.out.println("Detector found "  + counter 
				+ " peaks after filtering in " + (System.currentTimeMillis()-start) + "ms.");
		counter=0;
	}
	
	@Override
	public int getNumberOfFrames() {
		return nFrames;
	}
	
	/**
	 * Factory for the Fast Median Filter implementation
	 * 
	 * @author Ronny Sczech
	 *
	 */
	
	@Plugin(type = DetectorFactory.class )
	public static class Factory implements DetectorFactory {

		private Map<String, Object> settings;
		private final FastMedianPanel configPanel = new FastMedianPanel();

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
		public <T extends RealType<T> & NativeType<T>> Detector<T> getDetector() {
			boolean interpolating = (boolean) settings.get(FastMedianPanel.KEY_INTERPOLATING);
			int frames = (int) settings.get(FastMedianPanel.KEY_FRAMES);
			final double threshold = ( Double ) settings.get( FastMedianPanel.KEY_THRESHOLD );
			final int stepSize = ( Integer ) settings.get( FastMedianPanel.KEY_WINDOWSIZE );
			return new NMSFastMedian<>(frames, interpolating, threshold, stepSize);
		}

		@Override
		public ConfigurationPanel getConfigurationPanel() {
			configPanel.setName(KEY);
			return configPanel;
		}

		@Override
		public boolean hasPreProcessing() {
			return true;
		}	
	}
}
