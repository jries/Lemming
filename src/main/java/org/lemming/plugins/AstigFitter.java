package org.lemming.plugins;

import ij.IJ;
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
import org.lemming.math.GaussianFitterZ;
import org.lemming.modules.CPU_Fitter;
import org.lemming.modules.Fitter;
import org.lemming.pipeline.LocalizationPrecision3D;
import org.lemming.tools.LemmingUtils;
import org.lemming.pipeline.Localization;
import org.scijava.plugin.Plugin;

public class AstigFitter<T extends RealType<T>> extends CPU_Fitter<T> {

	private static final String NAME = "Astigmatism";

	private static final String KEY = "ASTIGFITTER";

	private static final String INFO_TEXT = "<html>" + "Astigmatism Fitter with Z" + "</html>";

	private final Map<String, Object> params;

	public AstigFitter(final int windowSize, final Map<String,Object> params) {
		super(windowSize);
		this.params=params;
	}

	@Override
	public List<Element> fit(final List<Element> sliceLocs, Frame<T> frame, final long halfKernel) {
		final double pixelDepth = frame.getPixelDepth();
		final RandomAccessibleInterval<T> pixels = frame.getPixels();
		final List<Element> found = new ArrayList<>();
		for (Element el : sliceLocs) {
			final Localization loc = (Localization) el; 
			long x = Math.round(loc.getX().doubleValue()/pixelDepth);
			long y = Math.round(loc.getY().doubleValue()/pixelDepth);
			long[] imageMin = new long[2];
			pixels.min(imageMin);
			long[] imageMax = new long[2];
			pixels.max(imageMax);
			Interval roi = cropInterval(imageMin,imageMax,new long[]{x - halfKernel,y - halfKernel},new long[]{x + halfKernel,y + halfKernel});
			
			GaussianFitterZ<T> gf = new GaussianFitterZ<>(Views.interval(pixels, roi), 1000, 1000, pixelDepth, params);
			double[] result;
			result = gf.fit();
			
			if (result != null){
				result[0] *= pixelDepth;
				result[1] *= pixelDepth;
				result[2] *= (double)params.get("zStep");
				result[3] *= pixelDepth;
				result[4] *= pixelDepth;
				result[5] *= (double)params.get("zStep");
				found.add(new LocalizationPrecision3D(result[0], result[1], result[2], result[3], result[4], result[5], result[6], loc.getFrame()));
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
			return settings.get(FitterPanel.KEY_CALIBRATION_FILENAME) != null;
		}

		@Override
		public <T extends RealType<T>> Fitter<T> getFitter() {
			final int windowSize = (int) settings.get(FitterPanel.KEY_WINDOW_SIZE);
			final String calibFileName = (String) settings.get(FitterPanel.KEY_CALIBRATION_FILENAME);
			if (calibFileName == null) {
				IJ.error("No Calibration File!");
				return null;
			}
			return new AstigFitter<>(windowSize, LemmingUtils.readCSV(calibFileName));
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
