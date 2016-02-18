package org.lemming.plugins;

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
import org.lemming.math.Symmetric2DFitter;
import org.lemming.modules.CPU_Fitter;
import org.lemming.modules.Fitter;
import org.lemming.pipeline.LocalizationPrecision3D;
import org.lemming.pipeline.Localization;
import org.scijava.plugin.Plugin;

public class SymmetricGaussianFitter<T extends RealType<T>> extends CPU_Fitter<T> {

	public static final String NAME = "Symmetric Gaussian";

	public static final String KEY = "SYMMETRICFITTER";

	public static final String INFO_TEXT = "<html>" + "2D symmetric Gaussian" + "</html>";


	public SymmetricGaussianFitter(final int halfkernel) {
		super(halfkernel, stepSize);
	}

	@Override
	public List<Element> fit(final List<Element> sliceLocs, Frame<T> frame, final long windowSize, double stepSize) {
		final double pixelDepth = frame.getPixelDepth();
		final ImageProcessor ip = ImageJFunctions.wrap(frame.getPixels(), "").getProcessor();
		final List<Element> found = new ArrayList<>();
		for (Element el : sliceLocs) {
			final Localization loc = (Localization) el;
			double x = loc.getX().doubleValue() / pixelDepth;
			double y = loc.getY().doubleValue() / pixelDepth;
			final Roi origroi = new Roi(x - size, y - size, 2 * size + 1, 2 * size + 1);
			final Roi roi = cropRoi(ip.getRoi(), origroi.getBounds());
			Symmetric2DFitter gf = new Symmetric2DFitter(ip, roi, 100, 100);
			double[] result = null;
			result = gf.fit();
			
			if (result != null){
				for (int i = 0; i < 6; i++)
					result[i] *= pixelDepth;
				found.add(new LocalizationPrecision3D(result[0], result[1], 0, result[5], result[5], result[6], result[3]/pixelDepth, loc.getFrame()));
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

	}

}
