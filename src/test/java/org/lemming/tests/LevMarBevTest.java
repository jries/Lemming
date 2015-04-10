package org.lemming.tests;

import org.junit.Before;
import org.junit.Test;
import org.lemming.utils.FitFunction;
import org.lemming.utils.LevMarBev;

public class LevMarBevTest {

	LevMarBev fit;
	FitFunction fcn;
	double[][] x;
	double[] y;
	double[] p;
	byte[] pFloat;
	
	@Before
	public void setUp() throws Exception {
		
		p = new double[13];
		p[0] = 0.23;
		p[1] = 9.3;
		p[2] = 20.4;
		p[3] = 53.5;
		p[4] = 1.1;
		p[5] = 2.3;
		p[6] = 3.1;
		p[7] = 18.4;
		p[8] = 17.1;
		p[9] = 50.31;
		p[10] = 2.2;
		p[11] = 1.4;
		p[12] = 2.3;
		
		int npts = 41;
		x = new double[2][npts];
		y = new double[npts];
		for (int i=0; i<npts; i++) {			
			for (int j=0; j<npts; j++) {
				x[0][i] = i;
				x[1][j] = j;
			}
		}
		
	}

	@Test
	public void test() {
	}

}
