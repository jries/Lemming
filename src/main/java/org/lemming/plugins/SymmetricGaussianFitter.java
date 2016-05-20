package org.lemming.plugins;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import net.imglib2.Interval;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.numeric.RealType;
import net.imglib2.view.Views;

import org.lemming.factories.FitterFactory;
import org.lemming.gui.FitterPanel;
import org.lemming.gui.ConfigurationPanel;
import org.lemming.interfaces.Element;
import org.lemming.interfaces.Frame;
import org.lemming.math.Symmetric2DFitter;
import org.lemming.modules.CPU_Fitter;
import org.lemming.modules.Fitter;
import org.lemming.pipeline.LocalizationPrecision3D;
import org.lemming.pipeline.Localization;
import org.scijava.plugin.Plugin;

public class SymmetricGaussianFitter<T extends RealType<T>> extends CPU_Fitter<T> {

	private static final String NAME = "Symmetric Gaussian";

	private static final String KEY = "SYMMETRICFITTER";

	private static final String INFO_TEXT = "<html>" + "2D symmetric Gaussian" + "</html>";


	public SymmetricGaussianFitter(final int halfkernel) {
		super(halfkernel);
	}

	@Override
	public List<Element> fit(final List<Element> sliceLocs, Frame<T> frame, final long halfKernel) {
		final double pixelDepth = frame.getPixelDepth();
		final RandomAccessibleInterval<T> pixels = frame.getPixels();
		final List<Element> found = new ArrayList<>();
		long[] imageMin = new long[2];
		long[] imageMax = new long[2];
		for (Element el : sliceLocs) {
			final Localization loc = (Localization) el;
			long x = Math.round(loc.getX().doubleValue()/pixelDepth);
			long y = Math.round(loc.getY().doubleValue()/pixelDepth);
			pixels.min(imageMin);
			pixels.max(imageMax);
			Interval roi = cropInterval(imageMin,imageMax,new long[]{x - halfKernel,y - halfKernel},new long[]{x + halfKernel,y + halfKernel});
			Symmetric2DFitter<T> gf = new Symmetric2DFitter<>(Views.interval(pixels, roi), 200, 200);
			double[] result;
			result = gf.fit();
			
			if (result != null){
				for (int i = 0; i < 6; i++)
					result[i] *= pixelDepth;
				found.add(new LocalizationPrecision3D(result[0], result[1], 0, result[5], result[5], result[6], result[3]/pixelDepth, loc.getFrame()));
			}
		}
		return found;
	}

	@Plugin(type = FitterFactory.class )
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
			final int windowSize = (int) settings.get(FitterPanel.KEY_WINDOW_SIZE);
			return new SymmetricGaussianFitter<>(windowSize);
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
