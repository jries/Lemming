package org.lemming.plugins;

import ij.IJ;
import ij.gui.Roi;
import ij.process.ImageProcessor;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import net.imglib2.img.display.imagej.ImageJFunctions;
import net.imglib2.type.numeric.RealType;

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

	public static final String NAME = "Astigmatism";

	public static final String KEY = "ASTIGFITTER";

	public static final String INFO_TEXT = "<html>" + "Astigmatism Fitter with Z" + "</html>";

	private double[] params;

	public AstigFitter(final int windowSize, double stepSize, final List<Double> list) {
		super(windowSize, stepSize);
		this.params = new double[list.size()];
		for (int i = 0; i < list.size(); i++)
			params[i] = list.get(i);
	}

	@Override
	public List<Element> fit(final List<Element> sliceLocs, Frame<T> frame, final long windowSize, double stepSize) {
		final double pixelDepth = frame.getPixelDepth();
		final ImageProcessor ip = ImageJFunctions.wrap(frame.getPixels(), "").getProcessor();
		final List<Element> found = new ArrayList<>();
		for (Element el : sliceLocs) {
			final Localization loc = (Localization) el;
			double x = loc.getX().longValue() / pixelDepth;
			double y = loc.getY().longValue() / pixelDepth;
			final Roi origroi = new Roi(x - size, y - size, 2 * size + 1, 2 * size + 1);
			final Roi roi = cropRoi(ip.getRoi(), origroi.getBounds());
			GaussianFitterZ gf = new GaussianFitterZ(ip, roi, 100, 100, pixelDepth, params);
			double[] result = null;
			result = gf.fit();
			
			if (result != null){
				result[0] *= pixelDepth;
				result[1] *= pixelDepth;
				result[2] *= stepSize;
				result[3] *= pixelDepth;
				result[4] *= pixelDepth;
				result[5] *= stepSize;
				found.add(new LocalizationPrecision3D(result[0], result[1], result[2], result[3], result[4], result[5], result[6], loc.getFrame()));
			}
		}
		return found;
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
			final String calibFileName = (String) settings.get(FitterPanel.KEY_CALIBRATION_FILENAME);
			if (calibFileName == null) {
				IJ.error("No Calibration File!");
				return null;
			}
			return new AstigFitter<>(windowSize, stepSize, LemmingUtils.readCSV(calibFileName).get("param"));
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
