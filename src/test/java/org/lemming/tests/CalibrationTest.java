package org.lemming.tests;

import org.lemming.math.Calibrator;
import org.lemming.tools.LemmingUtils;

import ij.IJ;
import ij.ImagePlus;
import ij.gui.Roi;
import ij.gui.StackWindow;

public class CalibrationTest {
	
	private StackWindow calibWindow;
	private Calibrator calibrator;
	
	public CalibrationTest(){
		ImagePlus calibImage = new ImagePlus("D:/ownCloud/z-stack_1.tif");
		calibWindow = new StackWindow(calibImage);
		calibImage.setRoi(45, 95, 21, 21);
	}
	
	private boolean fitbeads() {
		final Roi roitemp = calibWindow.getImagePlus().getRoi();
		Roi calibRoi = null;
		try {
			final double w = roitemp.getFloatWidth();
			final double h = roitemp.getFloatHeight();
			if (w != h) {
				IJ.showMessage("Needs a quadratic ROI /n(hint: press Shift).");
				return false;
			}
			calibRoi = roitemp;
		} catch (NullPointerException e) {
			calibRoi = new Roi(0, 0, calibWindow.getImagePlus().getWidth(), calibWindow.getImagePlus().getHeight());
		}

		final int zstep = 10; // set
		calibrator = new Calibrator(calibWindow.getImagePlus(), LemmingUtils.readCameraSettings(System.getProperty("user.home")+"/camera.props"), zstep, calibRoi);
		calibrator.fitStack();
		calibWindow.close();
		return true;
	}

	private boolean fitCurve() {
		final int rangeMin = 0; //set
		final int rangeMax = 400; //set
		calibrator.fitBSplines(rangeMin, rangeMax);
		return true;
	}

	private void saveCalibration() {
		calibrator.saveCalib("D:/ownCloud/set1-calb.csv");
		//calibrator.readCalib("/media/backup/ownCloud/set1-calb.csv");
		//calibrator.getCalibration().closePlotWindows();
	}

	public static void main(String[] args) {
		CalibrationTest ct = new CalibrationTest();
		ct.fitbeads();
		ct.fitCurve();
		ct.saveCalibration();
	}
}
