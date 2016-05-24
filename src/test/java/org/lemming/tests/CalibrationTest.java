package org.lemming.tests;

import org.lemming.math.Calibrator;

import ij.IJ;
import ij.ImagePlus;
import ij.gui.Roi;
import ij.gui.StackWindow;

class CalibrationTest {
	
	private final StackWindow calibWindow;
	private Calibrator calibrator;
	
	private CalibrationTest(){
	//final ImagePlus calibImage = new ImagePlus(System.getProperty("user.home")+"/ownCloud/set1.tif");
		final ImagePlus calibImage = new ImagePlus("H:/Images/stack-beads-100nm-AS-Exp-100x100x10-as-stack.tif");
		calibWindow = new StackWindow(calibImage);
		calibImage.setRoi(13, 19, 24, 24);
	}
	
	private void fitbeads() {
		final Roi roitemp = calibWindow.getImagePlus().getRoi();
		Roi calibRoi = null;
		try {
			final double w = roitemp.getFloatWidth();
			final double h = roitemp.getFloatHeight();
			if (w != h) {
				IJ.showMessage("Needs a quadratic ROI /n(hint: press Shift).");
				return;
			}
			calibRoi = roitemp;
		} catch (NullPointerException e) {
			calibRoi = new Roi(0, 0, calibWindow.getImagePlus().getWidth(), calibWindow.getImagePlus().getHeight());
		}

		final int zstep = 10; // set
		calibrator = new Calibrator(calibWindow.getImagePlus(), zstep, calibRoi);
		calibrator.fitStack();
		calibWindow.close();
	}

	private void fitCurve() {
		final int rangeMin = 30; //set
		final int rangeMax = 1500; //set
		calibrator.fitBSplines(rangeMin, rangeMax);
	}

	private void saveCalibration() {
		calibrator.saveCalib("H:/Images/stack-beads-100nm-AS-Exp-cal.csv");
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
