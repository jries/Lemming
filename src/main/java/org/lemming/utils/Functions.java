package org.lemming.utils;

/**
 * @author Ronny Sczech
 *
 */
public class Functions {

	/** A 2-dimensional, elliptical, Gaussian function.
	 * 
	 * @param X is a list of (x,y) localizations e.g. [ [x0,y0], [x1,y1], ..., [xn, yn] ] 
	 * @param P has the values
	 * @return
	 * <ul>
	 * <li> P[0] = background signal
	 * <li> P[1] = xLocalization
	 * <li> P[2] = yLocalization
	 * <li> P[3] = area under the curve (the total intensity)
	 * <li> P[4] = rotation angle, in radians, between the fluorescing molecule and the image canvas
	 * <li> P[5] = sigma of the 2D Gaussian in the x-direction, sigmaX
	 * <li> P[6] = aspect ratio, sigmaY/sigmaX (or equivalently fwhmY/fwhmX) 
	 * </ul> */
	public static double[] gaussian2D(int[][] X, double[] P){
		double[] fcn = new double[X.length];
		double t12 = Math.cos(P[4]);
		double t16 = Math.sin(P[4]);
		double t20 = Math.pow(1.0 / P[5], 2);
		double t27 = Math.pow(1.0 / (P[6]*P[5]), 2);
		double t2 = P[3] / (2.0 * Math.PI * P[5] * (P[6]*P[5]));
		double dx, dy;
		for(int i = 0; i < X.length; i++){
			dx = (double)X[i][0] - P[1];
			dy = (double)X[i][1] - P[2];
			fcn[i] = P[0] + t2*Math.exp(-0.5*( Math.pow(dx*t12 - dy*t16, 2)*t20 + Math.pow(dx*t16 + dy*t12, 2)*t27 ) );	
		}
		return fcn;
	}

}
