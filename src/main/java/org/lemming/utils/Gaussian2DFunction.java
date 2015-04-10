package org.lemming.utils;

/**
 * @author Ronny Sczech
 *
 */
public class Gaussian2DFunction implements FitFunction {
	
	// 'temporary' values,
	private int n;
	private double t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10;
	private double t11, t12, t13, t14, t15, t16, t17, t18, t19, t20;
	private double t21, t22, t23, t24, t25, t26, dx, dy, f1, f2;
	
	/**
	 * A normalized, 2D, symmetrical or elliptical, Gaussian point-spread function (PSF). 
	 * This function can model multiple overlapping PSFs by using the appropriate 
	 * {@code p} array.
	 * 
	 * @param x - a 2 by N list of x,y pixel coordinates, e.g. [ [x1, x2, ..., xN], [y1, y2, ..., yN] ] 
	 * @param y - the list containing the function values to be evaluated using
	 * the values in {@code p} for each value in {@code x}
	 * @param p - the values of the function parameters
	 * <ul>
	 * <li> p[0] = background signal </li>
	 * <li> p[1] = the localization in the x dimension </li>
	 * <li> p[2] = the localization in the y dimension </li>
	 * <li> p[3] = the total intensity (area under the curve) (the background signal
	 * is already subtracted) </li>
	 * <li> p[4] = rotation angle, in radians, between the fluorescing molecule
	 *  and the image canvas </li>
	 * <li> p[5] = sigma of the 2D Gaussian in the x-dimension, sigmaX </li>
	 * <li> p[6] = aspect ratio, sigmaY/sigmaX (or equivalently fwhmY/fwhmX) </li> 
	 * </ul>
	 * <p> You can use p[5] and p[6] to model a symmetrical or elliptical Gaussian 
	 * point-spread function in different ways.<br>
	 * (see: {@link org.lemming.utils.LevMarBev LevMarBev} for a description of 
	 * pFloat[] below, that is of course, only if you are using the {@code LevMarBev}
	 *  algorithm) </p>
	 * (fixed-symmetrical) To force sigmaX=sigmaY=sigma and keep the value of 
	 * sigma constant during the fit:<br>
	 * set p[5]=sigma, p[6]=1.0, pFloat[5]=0, pFloat[6]=0 <br><br>
	 * (float-symmetrical) To force sigmaX=sigmaY=sigma and let the fit determine
	 * the best value for sigma:<br>
	 * set p[5]=sigma, p[6]=1.0, pFloat[5]=1, pFloat[6]=0 <br><br>
	 * (fixed-fixed-elliptical) To force the values of sigmaX and sigmaY to be a
	 * constant during the fit:<br>
	 * set p[5]=sigmaX, p[6]=aspect_ratio, pFloat[5]=0, pFloat[6]=0 <br><br>
	 * (float-fixed-elliptical) To determine the best values for sigmaX from the 
	 * fit and force sigmaY to be a constant during the fit:<br>
	 * set p[5]=sigmaX, p[6]=aspect_ratio, pFloat[5]=1, pFloat[6]=0 <br><br>
	 * (fixed-float-elliptical) To determine the best values for sigmaY from the 
	 * fit and force sigmaX to be a constant during the fit:<br>
	 * set p[5]=sigmaX, p[6]=aspect_ratio, pFloat[5]=0, pFloat[6]=1, and then 
	 * determine sigmaY=p[5]*p[6] when the fitting routine is done<br><br>
	 * (float-float-elliptical) To determine the best values for sigmaX and sigmaY
	 * from the fit:<br>
	 * set p[5]=sigmaX, p[6]=aspect_ratio, pFloat[5]=1, pFloat[6]=1, and then determine
	 * sigmaY=p[5]*p[6] when the fitting routine is done<br>
	 * 
	 * <p>If modeling multiple overlapping 2D Gaussians, then {@code p} must have a very 
	 * specific structure. For example, consider 3 overlapping Gaussian PSFs, let's called them A, B, C. 
	 * Then {@code p} must have the following structure:<p>
	 * p = [<br>
	 * &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;background,<br>
	 * &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;A_x, A_y, A_intensity, A_angle, A_sigmaX, A_aspect,<br>
	 * &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;B_x, B_y, B_intensity, B_angle, B_sigmaX, B_aspect,<br>
	 * &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;C_x, C_y, C_intensity, C_angle, C_sigmaX, C_aspect<br>
	 * &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;]
	 */
	@Override
	public void fcn(double[][] x, double[] p, double[] y) {
		// avoid getting the parameter values and performing unnecessary method calls in the inner loop
		n = x[0].length;
		t0 = p[0];
		for (int i=0; i<n; i++)
			y[i] = t0; // add the background signal
		for (int j=0, mx=p.length-1; j<mx; j+=6) {
			t1 = p[1+j];
			t2 = p[2+j];
			t3 = p[3+j] / (2.0 * Math.PI * p[5+j] * (p[6+j]*p[5+j]));
			t4 = Math.cos(p[4+j]);
			t5 = Math.sin(p[4+j]);
			t6 = 1.0 / (p[5+j]*p[5+j]);
			t7 = 1.0 / ((p[6+j]*p[5+j])*(p[6+j]*p[5+j]));
			for (int i=0; i<n; i++) {
				dx = x[0][i] - t1;
				dy = x[1][i] - t2;
				f1 = dx*t4 - dy*t5;
				f2 = dx*t5 + dy*t4;
				y[i] += t3 * Math.exp(-0.5*( f1*f1*t6 + f2*f2*t7 ) ); // add the contribution from each PSF
			}
		}
	}
	
	/**
	 * The partial derivatives (i.e. the derivatives with respect to each parameter). 
	 * Used the <i>CodeGeneration[Fortran]</i> package from <b>Maplesoft</b> to 
	 * generate this 'optimized' code. 'Optimized' in the sense that it tries to reduce 
	 * the number of times that a calculation needs to be performed by storing the 
	 * calculation in a temporary number and using this number for future calculations.
	 * 
	 * @param x - a 2 by N list of x,y pixel coordinates, e.g. [ [x1, x2, ..., xN], [y1, y2, ..., yN] ]
	 * @param p - an array of the function parameters
	 * @param der - a ({@code p.length}) x ({@code x[0].length}) array of the 
	 * partial derivative values for each parameter
	 */
	@Override
	public void deriv(double[][] x, double[] p, double[][] der) {
		// avoid getting the parameter values and performing unnecessary method calls in the inner loop
		n = x[0].length;
		for (int i=0; i<n; i++)
			der[0][i] = 1.0; // the partial derivative for the background signal
		for (int j=0, mx=p.length-1; j<mx; j+=6) {
	  	    t0 = p[1+j];
	  	    t1 = p[2+j];
	  	    t2 = p[3+j];
		    t3 = 1.0 / Math.PI;
	  	    t4 = p[5+j]*p[5+j];
	  	    t5 = 1.0 / t4;
	  	    t6 = 1.0 / p[6+j]; 	    
	  	    t7 = Math.cos(p[4+j]);
	  	    t8 = Math.sin(p[4+j]);
	  	    t9 = p[6+j]*p[6+j];
	  	    t10 = 1.0 / t9;
	  	    t11 = p[3+j] * t3;
	  	    t12 = 1.0 / t4 / p[5+j];
	  	    t13 = t4 * t4;
	  	    t14 = t9 * t9;
	  	    for (int i=0; i<n; i++) {
	  	    	t15 = x[0][i] - t0;
	  	    	t16 = x[1][i] - t1;
	  	    	t17 = t15 * t7 - t16 * t8;
	  	    	t18 = t17 * t5;
	  	    	t19 = t15 * t8 + t16 * t7;
	  	    	t20 = t19 * t10;
	  	    	t21 = t17 * t17;
	  	    	t22 = t19 * t19;
	  	    	t23 = t22 * t10;
	  	    	t24 = Math.exp( (-t21 - t23) * t5 * 0.5 );
	  	    	t25 = t3 * t5 * t6 * t24 * 0.5;
	  	    	t26 = t2 * t25;
	  	    	der[1+j][i] = t26 * (t18 * t7 + t20 * t5 * t8);
	  	    	der[2+j][i] = t26 * (-t18 * t8 + t20 * t5 * t7);
	  	    	der[3+j][i] = t25;
	  	    	der[4+j][i] = t26 * t18 *(t19 - t20);
	  	    	der[5+j][i] = t11 * t24 * t6 * t12 * (t5 * (t21 + t23) * 0.5 - 1.0);
	  	    	der[6+j][i] = t11 * t24 * (t22 / (t13 * t14) - t5 * t10) * 0.5;
	  	    }
		}
  	}

	@Override
	public void finalCheck(double[][] x, double[] y, double[] p) {
	    // make sure the phase is between -pi and pi
		for (int j=0, mx=p.length-1; j<mx; j+=6)
			p[4+j] = 2.0*Math.PI*(Math.round(p[4+j]/(2.0*Math.PI)));
	}

	@Override
	public void pCheck(double[] p, double[] pInitial) {}

	/** Not used since the x values are 2D */
	@Override
	public void fcn(double[] x, double[] p, double[] y) {}

	/** Not used since the x values are 2D */
	@Override
	public void deriv(double[] x, double[] p, double[][] der) {}

	/** Not used since the x values are 2D */
	@Override
	public void finalCheck(double[] x, double[] y, double[] p) {}

}
