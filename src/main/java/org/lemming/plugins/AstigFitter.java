package org.lemming.plugins;

import ij.IJ;
import ij.gui.Roi;
import ij.process.ImageProcessor;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.display.imagej.ImageJFunctions;
import net.imglib2.type.numeric.RealType;

import org.lemming.factories.FitterFactory;
import org.lemming.gui.FitterPanel;
import org.lemming.gui.ConfigurationPanel;
import org.lemming.interfaces.Element;
import org.lemming.interfaces.Frame;
import org.lemming.math.GaussianFitterZ;
import org.lemming.modules.Fitter;
import org.lemming.pipeline.FittedLocalization;
import org.lemming.pipeline.Localization;
import org.lemming.pipeline.Settings;
import org.scijava.plugin.Plugin;

public class AstigFitter<T extends RealType<T>, F extends Frame<T>> extends Fitter<T, F> {
	
	public static final String NAME = "Astigmatism Fitter";

	public static final String KEY = "ASTIGFITTER";

	public static final String INFO_TEXT = "<html>"
			+ "Astigmatism Fitter Plugin"
			+ "</html>";
	
	private final double[] params;
	
	public AstigFitter(final int windowSize, final List<Double> list) {
		super(windowSize);
		this.params = new double[list.size()];
		for (int i =0 ; i<list.size(); i++)
			params[i]=list.get(i);
	}
	
	@Override
	public List<Element> fit(List<Element> sliceLocs,RandomAccessibleInterval<T> pixels, long windowSize, long frameNumber) {
		ImageProcessor ip = ImageJFunctions.wrap(pixels,"").getProcessor();
		List<Element> found = new ArrayList<>();
		for (Element el : sliceLocs) {
			final Localization loc = (Localization) el;
			final Roi origroi = new Roi(loc.getX() - size, loc.getY() - size, 2*size+1, 2*size+1);
			final Roi roi = cropRoi(ip.getRoi(),origroi.getBounds());
			GaussianFitterZ gf = new GaussianFitterZ(ip, roi, 3000, 1000, params);
			double[] result = null;
			result = gf.fit();
			if (result!= null)
				found.add(new FittedLocalization(loc.getID(),loc.getFrame(), result[0], result[1], result[2], result[3], result[4]));			
		}
		return found;
	}

	@Plugin( type = FitterFactory.class, visible = true )
	public static class Factory implements FitterFactory{

		
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
			if(settings.get(FitterPanel.KEY_CALIBRATION_FILENAME) != null)
				return true;
			return false;
		}

		@SuppressWarnings({ "rawtypes", "unchecked" })
		@Override
		public Fitter getFitter() {
			final int windowSize = (int) settings.get( FitterPanel.KEY_WINDOW_SIZE );
			final String calibFileName = (String) settings.get( FitterPanel.KEY_CALIBRATION_FILENAME );
			if (calibFileName == null){ 
				IJ.error("No Calibration File!");
				return null;
			}
			return new AstigFitter(windowSize, Settings.readCSV(calibFileName).get("param"));
		}

		@Override
		public ConfigurationPanel getConfigurationPanel() {
			return configPanel;
		}
		
	}


}
