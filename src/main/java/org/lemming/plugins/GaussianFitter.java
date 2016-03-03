package org.lemming.plugins;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.lemming.factories.FitterFactory;
import org.lemming.gui.FitterPanel;
import org.lemming.gui.ConfigurationPanel;
import org.lemming.interfaces.Element;
import org.lemming.interfaces.Frame;
import org.lemming.math.Gaussian2DFitter;
import org.lemming.modules.CPU_Fitter;
import org.lemming.modules.Fitter;
import org.lemming.pipeline.LocalizationPrecision3D;
import org.lemming.pipeline.Localization;
import org.scijava.plugin.Plugin;

import net.imglib2.Interval;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.numeric.RealType;
import net.imglib2.view.Views;

public class GaussianFitter<T extends RealType<T>> extends CPU_Fitter<T> {

	public static final String NAME = "Gaussian";
	public static final String KEY = "GAUSSIANFITTER";
	public static final String INFO_TEXT = "<html>" + "Gaussian Fitter Plugin (with sx and sy)" + "</html>";

	private List<Double> param;
	private List<Double> zgrid;
	private List<Double> Calibcurve;

	private static int INDEX_WX = 0;
	private static int INDEX_WY = 1;
	private static int INDEX_AX = 2;
	private static int INDEX_AY = 3;
	private static int INDEX_BX = 4;
	private static int INDEX_BY = 5;
	private static int INDEX_C = 6;
	private static int INDEX_D = 7;
	private static int INDEX_Mp = 8;

	public GaussianFitter(int windowSize) {
		super(windowSize);
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
			Gaussian2DFitter<T> gf = new Gaussian2DFitter<T>(Views.interval(pixels, roi), 200, 200);
			double[] result = null;
			result = gf.fit();
			if (result != null) {
				//double SxSy = result[2] * result[2] - result[3] * result[3];
				result[0] *= pixelDepth;
				result[1] *= pixelDepth;
				result[6] *= pixelDepth;
				result[7] *= pixelDepth;
				found.add(new LocalizationPrecision3D( result[0], result[1], 0//calculateZ(SxSy) 
					,result[6], result[7], result[8], result[4], loc.getFrame()));
			}
		}
		return found;
	}

	@SuppressWarnings("unused")
	private double calculateZ(final double SxSy) {
		final int end = Calibcurve.size() - 1;

		if (end < 1)
			return 0;
		if (Calibcurve.size() != zgrid.size())
			return 0;

		// reuse calibration curve -- we can use this as starting point
		if (SxSy < Math.min(Calibcurve.get(0), Calibcurve.get(end)))
			return Math.min(Calibcurve.get(0), Calibcurve.get(end));
		if (SxSy > Math.max(Calibcurve.get(0), Calibcurve.get(end)))
			return Math.max(Calibcurve.get(0), Calibcurve.get(end));

		return calcIterZ(SxSy, Math.min(zgrid.get(0), zgrid.get(end)), Math.max(zgrid.get(0), zgrid.get(end)), 1e-4);
	}

	private double calcIterZ(double SxSy, double start_, double end, double precision) {
		double zStep = Math.abs(end - start_) / 10;
		double curveWx = valuesWith(start_)[0];
		double curveWy = valuesWith(start_)[1];
		double calib = curveWx * curveWx - curveWy * curveWy;
		double distance = Math.abs(calib - SxSy);
		double idx = start_;
		for (double c = start_ + zStep; c <= end; c += zStep) {
			curveWx = valuesWith(c)[0];
			curveWy = valuesWith(c)[1];
			calib = curveWx * curveWx - curveWy * curveWy;
			double cdistance = Math.abs(calib - SxSy);
			if (cdistance < distance) {
				idx = c;
				distance = cdistance;
			}
		}
		if (zStep <= precision) {
			return idx;
		}
		return calcIterZ(SxSy, idx - zStep, idx + zStep, precision);
	}

	// with 9 Parameters
	private double[] valuesWith(double z) {
		double[] values = new double[2];
		double b;

		b = (z - param.get(INDEX_C) - param.get(INDEX_Mp)) / param.get(INDEX_D);
		values[0] = param.get(INDEX_WX) * Math.sqrt(1 + b * b + param.get(INDEX_AX) * b * b * b + param.get(INDEX_BX) * b * b * b * b);

		b = (z - param.get(INDEX_C) - param.get(INDEX_Mp)) / param.get(INDEX_D);
		values[1] = param.get(INDEX_WY) * Math.sqrt(1 + b * b + param.get(INDEX_AY) * b * b * b + param.get(INDEX_BY) * b * b * b * b);

		return values;
	}

	@Plugin(type = FitterFactory.class, visible = true)
	public static class Factory implements FitterFactory {

		private Map<String, Object> settings;
		private FitterPanel configPanel = new FitterPanel();

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
			if (settings.get(FitterPanel.KEY_CALIBRATION_FILENAME) != null)
				return true;
			return false;
		}

		@Override
		public <T extends RealType<T>> Fitter<T> getFitter() {
			final int windowSize = (int) settings.get(FitterPanel.KEY_WINDOW_SIZE);
			/*final String calibFileName = (String) settings.get(FitterPanel.KEY_CALIBRATION_FILENAME);
			if (calibFileName == null) {
				IJ.error("No Calibration File!");
				return null;
			}
			Map<String, List<Double>> cal = LemmingUtils.readCSV(calibFileName);*/
			return new GaussianFitter<>(windowSize/*, cal*/);
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
