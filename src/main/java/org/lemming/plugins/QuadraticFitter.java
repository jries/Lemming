package org.lemming.plugins;

import java.util.Arrays;
import java.util.List;
import java.util.Map;

import org.lemming.factories.FitterFactory;
import org.lemming.gui.ConfigurationPanel;
import org.lemming.gui.QuadraticFitterPanel;
import org.lemming.interfaces.Element;
import org.lemming.interfaces.Frame;
import org.lemming.math.SubpixelLocalization;
import org.lemming.modules.CPU_Fitter;
import org.lemming.modules.Fitter;
import org.scijava.plugin.Plugin;

import net.imglib2.RandomAccessible;
import net.imglib2.type.numeric.RealType;
import net.imglib2.view.Views;

public class QuadraticFitter<T extends RealType<T>> extends CPU_Fitter<T> {

	public static final String NAME = "Quadratic";

	public static final String KEY = "QUADRATICFITTER";

	public static final String INFO_TEXT = "<html>" + "Quadratic Fitter Plugin (without z-direction)" + "</html>";

	public QuadraticFitter(int halfkernel) {
		super(halfkernel, stepSize);
	}

	@Override
	public List<Element> fit(final List<Element> sliceLocs, Frame<T> frame, final long windowSize, double stepSize) {
		
		final RandomAccessible<T> ra = Views.extendBorder(frame.getPixels());
		final boolean[] allowedToMoveInDim = new boolean[ra.numDimensions()];
		Arrays.fill(allowedToMoveInDim, true);

		final List<Element> refined = SubpixelLocalization.refinePeaks(sliceLocs, ra, frame.getPixels(), true, size, true, 0.01f, allowedToMoveInDim,
				frame.getPixelDepth());

		return refined;
	}

	@Plugin(type = FitterFactory.class, visible = true)
	public static class Factory implements FitterFactory {

		private Map<String, Object> settings;
		private QuadraticFitterPanel configPanel = new QuadraticFitterPanel();

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
			final int windowSize = (int) settings.get(QuadraticFitterPanel.KEY_QUAD_WINDOW_SIZE);
			return new QuadraticFitter<>(windowSize);
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
