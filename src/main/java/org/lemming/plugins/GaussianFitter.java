package org.lemming.plugins;

import java.awt.Rectangle;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.lemming.factories.FitterFactory;
import org.lemming.gui.FitterPanel;
import org.lemming.gui.ConfigurationPanel;
import org.lemming.interfaces.Element;
import org.lemming.interfaces.Frame;
import org.lemming.math.GaussianFitterLM;
import org.lemming.modules.Fitter;
import org.lemming.pipeline.FittedLocalization;
import org.lemming.pipeline.Localization;
import org.lemming.pipeline.Settings;
import org.scijava.plugin.Plugin;

import ij.IJ;
import ij.gui.Roi;
import ij.process.ImageProcessor;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.display.imagej.ImageJFunctions;
import net.imglib2.type.numeric.RealType;

public class GaussianFitter<T extends RealType<T>, F extends Frame<T>> extends Fitter<T, F> {
	
	public static final String NAME = "Gaussian Fitter";

	public static final String KEY = "GAUSSIANFITTER";

	public static final String INFO_TEXT = "<html>"
			+ "Gaussian Fitter Plugin (with sx and sy)"
			+ "</html>";

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

	public GaussianFitter(int queueSize, int windowSize, final Map<String,List<Double>> cal) {
		super(queueSize, windowSize);
		param = cal.get("param");
		zgrid = cal.get("zgrid");
		Calibcurve = cal.get("Calibcurve");
		
	}

	@Override
	public List<Element> fit(List<Element> sliceLocs, RandomAccessibleInterval<T> pixels, long windowSize, long frameNumber) {
		ImageProcessor ip = ImageJFunctions.wrap(pixels,"").getProcessor();
		List<Element> found = new ArrayList<>();
		final Rectangle imageRoi = ip.getRoi();
		for (Element el : sliceLocs) {
			final Localization loc = (Localization) el;
			final Roi origroi = new Roi(loc.getX() - size, loc.getY() - size, 2*size+1, 2*size+1);
			final Roi roi = cropRoi(imageRoi,origroi.getBounds());
			GaussianFitterLM gf = new GaussianFitterLM(ip, roi, 3000, 1000);
			double[] result = null;
			result = gf.fit();
			if (result!= null){
				double SxSy = result[2]*result[2] - result[3]*result[3];
				found.add(new FittedLocalization(loc.getID(),loc.getFrame(), result[0], result[1], calculateZ(SxSy), result[2], result[3]));	
			}
		}
		return found;
	}
	
	private double calculateZ(final double SxSy){
		final int end = Calibcurve.size()-1;
		
		if(end < 1) return 0;
		if(Calibcurve.size() != zgrid.size()) return 0;
		
		// reuse calibration curve -- we can use this as starting point
		if (SxSy < Math.min(Calibcurve.get(0),Calibcurve.get(end)) )
			return Math.min(Calibcurve.get(0),Calibcurve.get(end));
		if (SxSy > Math.max(Calibcurve.get(0),Calibcurve.get(end)) )
			return Math.max(Calibcurve.get(0),Calibcurve.get(end));

		return calcIterZ(SxSy, Math.min(zgrid.get(0),zgrid.get(end)), Math.max(zgrid.get(0),zgrid.get(end)), 1e-4);
		
	}
	
	private double calcIterZ(double SxSy, double start, double end, double precision) {
		double zStep = Math.abs(end-start)/10;
		double curveWx = valuesWith(start)[0];
		double curveWy = valuesWith(start)[1];
		double calib = curveWx*curveWx-curveWy*curveWy;
		double distance = Math.abs(calib - SxSy);
		double idx = start;
		for ( double c = start+zStep ; c<=end; c += zStep){
			curveWx = valuesWith(c)[0];
			curveWy = valuesWith(c)[1];
			calib = curveWx*curveWx-curveWy*curveWy;
		    double cdistance = Math.abs(calib - SxSy);
		    if(cdistance < distance){
		        idx = c;
		        distance = cdistance;
		    }
		}
		if (zStep<=precision){ 
			return idx;
		}
		return calcIterZ(SxSy,idx - zStep, idx + zStep, precision);
	}
	
	// with 9 Parameters
	private double[] valuesWith(double z) {
		double[] values = new double[2];
		double b;
		
		b = (z-param.get(INDEX_C)-param.get(INDEX_Mp))/param.get(INDEX_D);
		values[0] = param.get(INDEX_WX)*Math.sqrt(1+b*b+param.get(INDEX_AX)* b*b*b + param.get(INDEX_BX) * b*b*b*b);
	
		b = (z-param.get(INDEX_C)-param.get(INDEX_Mp))/param.get(INDEX_D);
		values[1] = param.get(INDEX_WY) * Math.sqrt(1 + b*b + param.get(INDEX_AY) * b*b*b + param.get(INDEX_BY) * b*b*b*b);
		
		return values;
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
			final int queueSize = (int) settings.get( FitterPanel.KEY_QUEUE_SIZE );
			final int windowSize = (int) settings.get( FitterPanel.KEY_WINDOW_SIZE );
			final String calibFileName = (String) settings.get( FitterPanel.KEY_CALIBRATION_FILENAME );
			if (calibFileName == null){ 
				IJ.error("No Calibration File!");
				return null;
			}
			Map<String, List<Double>> cal = Settings.readCSV(calibFileName);
			return new GaussianFitter(queueSize, windowSize, cal);
		}

		@Override
		public ConfigurationPanel getConfigurationPanel() {
			return configPanel;
		}
		
	}

}
