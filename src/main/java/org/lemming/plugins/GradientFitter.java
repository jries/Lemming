package org.lemming.plugins;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.apache.commons.math3.analysis.polynomials.PolynomialFunction;
import org.lemming.factories.FitterFactory;
import org.lemming.gui.ConfigurationPanel;
import org.lemming.gui.FitterPanel;
import org.lemming.gui.GradientFitterPanel;
import org.lemming.interfaces.Element;
import org.lemming.interfaces.Frame;
import org.lemming.math.Gradient;
import org.lemming.modules.CPU_Fitter;
import org.lemming.modules.Fitter;
import org.lemming.pipeline.Localization;
import org.lemming.pipeline.LocalizationPrecision3D;
import org.lemming.tools.LemmingUtils;
import org.scijava.plugin.Plugin;

import ij.IJ;
import net.imglib2.Interval;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.numeric.RealType;
import net.imglib2.view.Views;

public class GradientFitter<T extends RealType<T>> extends CPU_Fitter<T> {

	public static final String NAME = "Gradient";

	public static final String KEY = "GRADIENTFITTER";

	public static final String INFO_TEXT = "<html>" + "Gradient Fitter Plugin" + "</html>";

	private PolynomialFunction zFunction;

	//private double zStep;

	public GradientFitter(int halfkernel, final Map<String,Object> params) {
		super(halfkernel);
		//zStep=(double) params.get("zStep");
		zFunction = (PolynomialFunction) params.get("ellipticity");
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
			long x = Math.round(loc.getX().doubleValue() / pixelDepth);
			long y = Math.round(loc.getY().doubleValue() / pixelDepth);
			pixels.min(imageMin);
			pixels.max(imageMax);
			final Interval roi = cropInterval(imageMin, imageMax, new long[] { x - halfKernel, y - halfKernel },
					new long[] { x + halfKernel, y + halfKernel });
			final Gradient<T> gf = new Gradient<T>(Views.interval(pixels, roi), 0, (int) halfKernel);
			double[] result = null;
			result = gf.fit();
			if (result != null) {
				result[0] *= pixelDepth;
				result[1] *= pixelDepth;

				found.add(new LocalizationPrecision3D(result[0], result[1], zFunction.value(result[2]), 0, 0, 0,
						result[3], loc.getFrame()));
			}
		}
		return found;
	}

	@Plugin(type = FitterFactory.class, visible = true)
	public static class Factory implements FitterFactory {

		private Map<String, Object> settings;
		private GradientFitterPanel configPanel = new GradientFitterPanel();

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
		public <T extends RealType<T>> Fitter<T> getFitter() {
			final int windowSize = (int) settings.get(GradientFitterPanel.KEY_GRADIENT_WINDOW_SIZE);
			final String calibFileName = (String) settings.get(FitterPanel.KEY_CALIBRATION_FILENAME);
			if (calibFileName == null) {
				IJ.error("No Calibration File!");
				return null;
			}
			return new GradientFitter<>(windowSize, LemmingUtils.readCSV(calibFileName));
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

	}

}
