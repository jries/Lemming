package org.lemming.plugins;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import net.imglib2.Cursor;
import net.imglib2.RandomAccess;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.algorithm.gauss3.Gauss3;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.util.Util;
import net.imglib2.view.ExtendedRandomAccessibleInterval;
import net.imglib2.view.Views;

import org.lemming.factories.DetectorFactory;
import org.lemming.gui.ConfigurationPanel;
import org.lemming.gui.NMSDetectorPanel;
import org.lemming.interfaces.Detector;
import org.lemming.interfaces.Element;
import org.lemming.interfaces.Frame;
import org.lemming.pipeline.FrameElements;
import org.lemming.pipeline.Localization;
import org.lemming.pipeline.MultiRunModule;
import org.lemming.tools.LemmingUtils;
import org.scijava.plugin.Plugin;

public class NMSDetector<T extends RealType<T> & NativeType<T>> extends MultiRunModule implements Detector<T> {

	private static final String NAME = "NMS Detector";
	private static final String KEY = "NMSDETECTOR";
	private static final String INFO_TEXT = "<html>" + "NMS Detector Plugin" + "</html>";
	private final double threshold;
	private final int n_;
	private final int gaussian;

	public NMSDetector(final double threshold, final int size, final int gaussian) {
		this.threshold = threshold;
		this.n_ = size;
		this.gaussian = gaussian;
	}
	
	@SuppressWarnings("unchecked")
	@Override
	public Element processData(Element data) {
		Frame<T> frame = (Frame<T>) data;
		if (frame == null)
			return null;

		if (frame.isLast()) { // make the poison pill
			cancel();
			FrameElements<T> res = detect(frame);
			if (res!=null){
				res.setLast(true);
				counterList.add(res.getList().size());
				return res;
			} else {
				res = new FrameElements<>(null, frame);
				res.setLast(true);
				return res;
			}
		}
		FrameElements<T> res = detect(frame);
		if (res != null)
			counterList.add(res.getList().size());
		return res;
	}

	@Override
	public FrameElements<T> detect(Frame<T> frame) {
		final RandomAccessibleInterval<T> pixels = frame.getPixels();
		double[] sigma = new double[ pixels.numDimensions() ];
		RandomAccess<T> ra;
		final RandomAccess<T> ro = pixels.randomAccess();
		// compute max of the Image
		final T max = LemmingUtils.computeMax(Views.iterable(pixels));
		double threshold_ = max.getRealDouble() / 100.0 * threshold;
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
			T type = Views.iterable(pixels).firstElement();
			final RandomAccessibleInterval<T> dog = Views.offset(Util.getArrayOrCellImgFactory(pixels, type).create(pixels, type), min);
			
			try {
				Gauss3.gauss(sigma, extended, dog, 2);
			} catch (Exception e) {
				return null;
			}
			
			final Cursor<T> dogCursor = Views.iterable(dog).cursor();
			final Cursor<T> tmpCursor = Views.iterable(pixels).cursor();

			while (dogCursor.hasNext()){
				tmpCursor.fwd();
				dogCursor.fwd();
				double val = Math.abs(tmpCursor.get().getRealDouble()-dogCursor.get().getRealDouble());
				dogCursor.get().setReal(val);
			}
			ra = dog.randomAccess();
		}
		else
			ra = pixels.randomAccess();

		int i, j, ii, jj, ll, kk;
		int mi, mj;
		boolean failed;
		long width_ = pixels.dimension(0);
		long height_ = pixels.dimension(1);
		List<Element> found = new ArrayList<>();
		T first, third, second = max;

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
					ro.setPosition(new int[] { mi, mj });
					T value = ro.get();
					if (value.getRealDouble() > threshold_) {
						found.add(new Localization(mi * frame.getPixelDepth(), mj * frame.getPixelDepth(), value.getRealDouble() ,frame.getFrameNumber()));
					}
				}
			}
		}

		if (found.isEmpty())
			return null;
		return new FrameElements<>(found, frame);
	}
	
	@Override
	protected void afterRun() {
		Integer cc=0;
		for (Integer i : counterList)
			cc+=i;
		System.out.println("Detector found "
				+ cc + " peaks in "
				+ (System.currentTimeMillis() - start) + "ms.");
	}

	@Override
	public boolean check() {
		return inputs.size()==1 && outputs.size()>=1;
	}
	
	@Plugin(type = DetectorFactory.class )
	public static class Factory implements DetectorFactory {

		private Map<String, Object> settings;
		private final NMSDetectorPanel configPanel = new NMSDetectorPanel();

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
			final int threshold = (Integer) settings.get(NMSDetectorPanel.KEY_NMS_THRESHOLD);
			final int stepSize = (Integer) settings.get(NMSDetectorPanel.KEY_NMS_STEPSIZE);
			final int gaussian = (Integer) settings.get(NMSDetectorPanel.KEY_NMS_GAUSSIAN_SIZE);
			return new NMSDetector<>(threshold, stepSize, gaussian);
		}

		@Override
		public ConfigurationPanel getConfigurationPanel() {
			configPanel.setName(KEY);
			return configPanel;
		}

		@Override
		public boolean hasPreProcessing() {
			return false;
		}

	}

}
