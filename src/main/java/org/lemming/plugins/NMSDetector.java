package org.lemming.plugins;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import net.imglib2.Cursor;
import net.imglib2.RandomAccess;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.algorithm.gauss3.Gauss3;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Util;
import net.imglib2.view.ExtendedRandomAccessibleInterval;
import net.imglib2.view.Views;

import org.lemming.factories.DetectorFactory;
import org.lemming.gui.ConfigurationPanel;
import org.lemming.gui.NMSDetectorPanel;
import org.lemming.interfaces.Element;
import org.lemming.interfaces.Frame;
import org.lemming.modules.Detector;
import org.lemming.pipeline.FrameElements;
import org.lemming.pipeline.Localization;
import org.scijava.plugin.Plugin;

public class NMSDetector<T extends RealType<T>, F extends Frame<T>> extends Detector<T> {

	public static final String NAME = "NMS Detector";

	public static final String KEY = "NMSDETECTOR";

	public static final String INFO_TEXT = "<html>" + "NMS Detector Plugin" + "</html>";

	private double threshold;

	private int n_;

	private int counter = 0;

	private int gaussian;

	public NMSDetector(final double threshold, final int size, final int gaussian) {
		this.threshold = threshold;
		this.n_ = size;
		this.gaussian = gaussian;
		//this.service = Executors.newSingleThreadExecutor();
	}

	@Override
	public FrameElements<T> detect(Frame<T> frame) {
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
				Gauss3.gauss(sigma, extended, dog, 1);
			} catch (Exception e) {
				return null;
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
		RandomAccess<T> ra = pixels.randomAccess();

		int i, j, ii, jj, ll, kk;
		int mi, mj;
		boolean failed = false;
		long width_ = pixels.dimension(0);
		long height_ = pixels.dimension(1);
		List<Element> found = new ArrayList<>();

		for (i = 0; i <= width_ - 1 - n_; i += n_ + 1) { // Loop over (n+1)x(n+1)
			for (j = 0; j <= height_ - 1 - n_; j += n_ + 1) {
				mi = i;
				mj = j;
				for (ii = i; ii <= i + n_; ii++) {
					for (jj = j; jj <= j + n_; jj++) {
						ra.setPosition(new int[] { ii, jj });
						final T first = ra.get().copy();
						ra.setPosition(new int[] { mi, mj });
						final T second = ra.get().copy();
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
								T first = ra.get().copy();
								ra.setPosition(new int[] { mi, mj });
								T second = ra.get().copy();
								if (first.compareTo(second) > 0) {
									failed = true;
									break Outer;
								}
							}
						}
					}
				}
				if (!failed) {
					ra.setPosition(new int[] { mi, mj });
					T value = ra.get();
					if (value.getRealDouble() > threshold) {
						found.add(new Localization(mi * frame.getPixelDepth(), mj * frame.getPixelDepth(), value.getRealDouble() ,frame.getFrameNumber()));
						counter++;
					}
				}
			}
		}

		if (found.isEmpty())
			return null;
		return new FrameElements<>(found, frame);
	}

	@Plugin(type = DetectorFactory.class, visible = true)
	public static class Factory implements DetectorFactory {

		private Map<String, Object> settings;
		private NMSDetectorPanel configPanel = new NMSDetectorPanel();

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
		public <T extends RealType<T>> Detector<T> getDetector() {
			final double threshold = (Double) settings.get(NMSDetectorPanel.KEY_NMS_THRESHOLD);
			final int stepSize = (Integer) settings.get(NMSDetectorPanel.KEY_NMS_STEPSIZE);
			final int gaussian = (Integer) settings.get(NMSDetectorPanel.KEY_NMS_GAUSSIAN_SIZE);
			return new NMSDetector<>(threshold, stepSize, gaussian);
		}

		@Override
		public ConfigurationPanel getConfigurationPanel() {
			configPanel.setName(KEY);
			return configPanel;
		}

	}

}
