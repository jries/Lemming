package org.lemming.math;

import org.apache.commons.math3.analysis.MultivariateMatrixFunction;
import org.apache.commons.math3.analysis.MultivariateVectorFunction;
import org.apache.commons.math3.optim.OptimizationData;
import org.apache.commons.math3.util.FastMath;

import ij.gui.Roi;
import ij.process.ImageProcessor;

/** ThunderSTORM algorithm for a symmetric Gaussian*/
public class SimpleGaussian implements OptimizationData {
	private int[] xgrid, ygrid;
	double[] params;
	double[] initialGuess;

	private static final int INDEX_X0 = 0;
	private static final int INDEX_Y0 = 1;
	private static final int INDEX_S = 2;
	private static final int INDEX_SX = 2;
	private static final int INDEX_SY = 3;
	private static final int INDEX_I0 = 3;
	private static final int INDEX_Bg = 4;
	private static final int PARAM_LENGTH = 5;

	public SimpleGaussian(int[] xgrid, int[] ygrid) {
		this.xgrid = xgrid;
		this.ygrid = ygrid;
	}

	public static double getValue(double[] params, double x, double y) {
		double twoSigmaSquared = params[INDEX_S] * params[INDEX_S] * 2;
        return params[INDEX_Bg] + params[INDEX_I0] / (twoSigmaSquared * FastMath.PI)
                * FastMath.exp(-((x - params[INDEX_X0]) * (x - params[INDEX_X0]) + (y - params[INDEX_Y0]) * (y - params[INDEX_Y0])) / twoSigmaSquared);
	}

	public MultivariateVectorFunction getModelFunction() {
		return new MultivariateVectorFunction() {
			@Override
			public double[] value(double[] params_) throws IllegalArgumentException {
				double[] retVal = new double[xgrid.length];
				for (int i = 0; i < xgrid.length; i++) {
					retVal[i] = getValue(params_, xgrid[i], ygrid[i]);
				}
				return retVal;
			}
		};
	}
	
	 public MultivariateMatrixFunction getModelFunctionJacobian() {
	        return new MultivariateMatrixFunction() {
	            @Override
	            public double[][] value(double[] point) throws IllegalArgumentException {
	            	
	            	final double sigma = point[INDEX_S];
	            	final double sigmaSquared = sigma * sigma;

	            	final double[][] jacobian = new double[xgrid.length][PARAM_LENGTH];
	            	 
	        	     for (int i = 0; i < xgrid.length; ++i) {      
	        	    	 final double xd = (xgrid[i] - point[INDEX_X0]);
	        	    	 final double yd = (ygrid[i] - point[INDEX_Y0]);
	        	    	 final double upper = -(xd * xd + yd * yd) / (2 * sigmaSquared);
	        	    	 final double expVal = FastMath.exp(upper);
	        	    	 final double expValDivPISigmaSquared = expVal / (sigmaSquared * FastMath.PI);
	        	    	 final double expValDivPISigmaPowEight = expValDivPISigmaSquared / sigmaSquared;
	  	        	    	 
	        	    	 jacobian[i][INDEX_X0] = point[INDEX_I0] * xd * expValDivPISigmaPowEight * 0.5;
	        	    	 jacobian[i][INDEX_Y0] = point[INDEX_I0] * yd * expValDivPISigmaPowEight * 0.5;
	        	    	 jacobian[i][INDEX_S]  = point[INDEX_I0] * expValDivPISigmaPowEight / point[INDEX_S] * (xd * xd + yd * yd - 2 * sigmaSquared);
	        	    	 jacobian[i][INDEX_I0] = point[INDEX_I0] * expValDivPISigmaSquared;
	        	    	 jacobian[i][INDEX_Bg] = 2 * point[INDEX_Bg];
	        	     }
	        	     
					return jacobian;
	            }
	        };
	    }

	public double[] getInitialGuess(ImageProcessor ip, Roi roi) {
		initialGuess = new double[PARAM_LENGTH];
		
		double[] centroid = CentroidFitterIP.fitCentroidandWidth(ip, roi, ip.getAutoThreshold());

		initialGuess[INDEX_X0] = centroid[INDEX_X0];
		initialGuess[INDEX_Y0] = centroid[INDEX_Y0];
		initialGuess[INDEX_S]  = 0.25*(centroid[INDEX_SY] * centroid[INDEX_SY] + centroid[INDEX_SX] * centroid[INDEX_SX]);
		initialGuess[INDEX_I0] = ip.getMax() - ip.getMin();
		initialGuess[INDEX_Bg] = ip.getMin();

		return initialGuess;
	}

}
