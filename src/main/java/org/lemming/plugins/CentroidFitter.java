package org.lemming.plugins;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import net.imglib2.FinalInterval;
import net.imglib2.Interval;
import net.imglib2.RandomAccessible;
import net.imglib2.type.numeric.RealType;
import net.imglib2.view.IntervalView;
import net.imglib2.view.Views;

import org.lemming.factories.FitterFactory;
import org.lemming.gui.ConfigurationPanel;
import org.lemming.gui.FitterPanel;
import org.lemming.interfaces.Element;
import org.lemming.interfaces.Frame;
import org.lemming.math.CentroidFitterRA;
import org.lemming.modules.CPU_Fitter;
import org.lemming.modules.Fitter;
import org.lemming.pipeline.Localization;
import org.lemming.pipeline.LocalizationPrecision3D;
import org.scijava.plugin.Plugin;

public class CentroidFitter<T extends RealType<T>> extends CPU_Fitter<T> {

	private static final String NAME = "Centroid Fitter";

	private static final String KEY = "CENTROIDFITTER";

	private static final String INFO_TEXT = "<html>" + "Centroid Fitter Plugin" + "</html>";

	private final double thresh;

	public CentroidFitter(int halfkernel, double threshold_) {
		super(halfkernel);
		thresh = threshold_;
	}

	@Override
	public List<Element> fit(List<Element> sliceLocs, Frame<T> frame, long windowSize) {
		final double pixelDepth = frame.getPixelDepth();
		final RandomAccessible<T> source = Views.extendMirrorSingle(frame.getPixels());
		final List<Element> found = new ArrayList<>();
		final int halfKernel = size;

		for (Element el : sliceLocs) {
			final Localization loc = (Localization) el;
			
			double x = loc.getX().doubleValue()/pixelDepth;
			double y = loc.getY().doubleValue()/pixelDepth;

			final Interval roi = new FinalInterval(new long[] { (long) StrictMath.floor(x - halfKernel),
					(long) StrictMath.floor(y - halfKernel) }, new long[] { (long) StrictMath.ceil(x + halfKernel),
					(long) StrictMath.ceil(y + halfKernel) });
			IntervalView<T> interval = Views.interval(source, roi);

			CentroidFitterRA<T> cf = new CentroidFitterRA<>(interval, thresh);
			double[] result = cf.fit();
			if (result != null){
				for (int i = 0; i < 4; i++)
					result[i] *= pixelDepth;
				found.add(new LocalizationPrecision3D(result[0], result[1], 0, result[2], result[3], 0, result[4], loc.getFrame()));
			}
		}

		return found;
	}

	@Plugin(type = FitterFactory.class)
	public static class Factory implements FitterFactory {

		private Map<String, Object> settings;
		private final FitterPanel configPanel = new FitterPanel();

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

		@Override
		public <T extends RealType<T>> Fitter<T> getFitter() {
			final int windowSize = (Integer) settings.get(FitterPanel.KEY_WINDOW_SIZE);
			final double threshold = (Double) settings.get(FitterPanel.KEY_CENTROID_THRESHOLD);
			return new CentroidFitter<>(windowSize, threshold);
		}

		@Override
		public ConfigurationPanel getConfigurationPanel() {
			configPanel.setName(KEY);
			return configPanel;
		}

		@Override
		public int getHalfKernel() {
			return size;
		}
		
		@Override
		public boolean hasGPU() {
			return false;
		}
	}

}
