package org.lemming.tests;

import static org.junit.Assert.*;

import org.junit.Before;
import org.junit.Test;
import org.lemming.utils.FitFunction;
import org.lemming.utils.Gaussian2DFunction;
import org.lemming.utils.LevMarBev;

public class LevMarBevTest {
	
	@Before
	public void setUp() throws Exception {
	}

	@Test
	public void testGaussian2DOneMolecule() {
		
		// create the fit function
		FitFunction fcn = new Gaussian2DFunction();
		
		// this is what we want to fit to converge to
		double[] expect = {0.25533, 9.312, 20.4634, 53.5352, 1.17583, 2.3132, 3.124};
		
		// create the x and y data
		int npts = 40;
		double[][] x = new double[2][npts];
		for (int i=0; i<npts; i++) {		
			for (int j=0; j<npts; j++) {
				x[0][i] = i;
				x[1][j] = j;
			}
		}
		double[] y = new double[npts*npts];
		fcn.fcn(x, expect, y);

		// pick a reasonable guess
		double[] guess = {0.0, 8.0, 22.0, 40.0, 1.0, 1.0, 2.0};
		
		// prepare the fitting routine
		LevMarBev fit = new LevMarBev(fcn, x, y, guess);
		// just to make sure that there are no warnings
		fit.setVerbose(true);
		
		// fit it
		fit.run();
		
		// test that we get out what we put in
		double[] best = fit.getBestParameters();
		for (int i=0; i<best.length; i++)
			assertEquals(0.0, Math.abs(best[i]-expect[i]), 1e-12);
	}

	@Test
	public void testGaussian2DTwoMolecules() {
		
		// create the fit function
		FitFunction fcn = new Gaussian2DFunction();
		
		// this is what we want to fit to converge to
		double[] expect = {0.25533, 9.3132, 20.4634, 53.5352, 1.17583, 2.3132, 3.124,
								   32.6221, 24.63232, 102.426532, 0.0123, 1.3245, 0.5325};
		
		// create the x and y data
		int npts = 40;
		double[][] x = new double[2][npts];
		for (int i=0; i<npts; i++) {		
			for (int j=0; j<npts; j++) {
				x[0][i] = i;
				x[1][j] = j;
			}
		}
		double[] y = new double[npts*npts];
		fcn.fcn(x, expect, y);

		// pick a reasonable guess
		double[] guess = {0.0, 8.0, 22.0, 40.0, 1.0, 1.0, 2.0, 33.0, 26.0, 150.0, 0.2, 1.0, 0.4};
		
		// prepare the fitting routine
		LevMarBev fit = new LevMarBev(fcn, x, y, guess);
		fit.setCalculateResiduals(true);
		// just to make sure that there are no warnings
		fit.setVerbose(true);
		
		// fit it
		fit.run();
		
		// test that we get out what we put in
		double[] best = fit.getBestParameters();
		for (int i=0; i<best.length; i++) {
			assertEquals(0.0, Math.abs(best[i]-expect[i]), 1e-12);
		}
		
		// test that the residuals are zero (since we did not add any noise
		// to the data, the residuals should be zero)
		double[] residuals = fit.getResuiduals();
		for (int i=0; i<residuals.length; i++) {
			assertEquals(0.0, Math.abs(residuals[i]), 1e-12);
		}
		
	}

}
