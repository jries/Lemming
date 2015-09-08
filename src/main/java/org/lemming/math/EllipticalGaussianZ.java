package org.lemming.math;

import ij.gui.Roi;
import ij.process.ImageProcessor;

import java.util.Arrays;

import org.apache.commons.math3.analysis.MultivariateMatrixFunction;
import org.apache.commons.math3.analysis.MultivariateVectorFunction;
import org.apache.commons.math3.optim.OptimizationData;
import org.apache.commons.math3.special.Erf;
import org.apache.commons.math3.util.FastMath;

public class EllipticalGaussianZ implements OptimizationData {
	
	int[] xgrid, ygrid;
	double[] params;
	double[] initialGuess;
		
	public static int INDEX_WX = 0;
	public static int INDEX_WY = 1;
	public static int INDEX_AX = 2;
	public static int INDEX_AY = 3;
	public static int INDEX_BX = 4;
	public static int INDEX_BY = 5;
	public static int INDEX_C = 6;
	public static int INDEX_D = 7;
	public static int INDEX_Mp = 8;
	public static int PARAM_1D_LENGTH = 9;
	
	public static int INDEX_X0 = 0;
	public static int INDEX_Y0 = 1;
	public static int INDEX_SX = 2;
	public static int INDEX_Z0 = 2;
	public static int INDEX_SY = 3;
	public static int INDEX_I0 = 3;
	public static int INDEX_Bg = 4;
	public static int PARAM_LENGTH = 5;
	
	static double defaultSigma = 1.5;
	private static double sqrt2 = Math.sqrt(2);
	
	public EllipticalGaussianZ(int[] xgrid, int[] ygrid, double[] params){
		this.xgrid = xgrid;
		this.ygrid = ygrid;
		this.params = params;
	}
	
    public double getValue(double[] parameter, double x, double y) {
        return parameter[INDEX_I0]*Ex(x,parameter)*Ey(y,parameter)+parameter[INDEX_Bg];
    }
    
    public MultivariateVectorFunction getModelFunction() {
        return new MultivariateVectorFunction() {
            @Override
            public double[] value(double[] parameter) throws IllegalArgumentException {
                double[] retVal = new double[xgrid.length];
                for(int i = 0; i < xgrid.length; i++) {
                    retVal[i] = getValue(parameter, xgrid[i], ygrid[i]);
                }
                return retVal;
            }
        };
    }
    
    public MultivariateMatrixFunction getModelFunctionJacobian() {
        return new MultivariateMatrixFunction() {
            @Override
            public double[][] value(double[] point) throws IllegalArgumentException {

            	 double[][] jacobian = new double[xgrid.length][PARAM_LENGTH];
            	 
        	     for (int i = 0; i < xgrid.length; ++i) {	 
        	    	 jacobian[i][INDEX_X0] = point[INDEX_I0]*Ey(ygrid[i], point)*dEx(xgrid[i],point);
        	    	 jacobian[i][INDEX_Y0] = point[INDEX_I0]*Ex(xgrid[i], point)*dEy(ygrid[i],point); 
        	    	 jacobian[i][INDEX_Z0] = point[INDEX_I0]*
        	    			 (dEsx(xgrid[i],point)*Ey(ygrid[i], point)*dSx(point[INDEX_Z0])+
	    					 Ex(xgrid[i],point)*dEsy(ygrid[i], point)*dSy(point[INDEX_Z0]));
        	    	 jacobian[i][INDEX_I0] = Ex(xgrid[i], point)*Ey(ygrid[i],point);
        	    	 jacobian[i][INDEX_Bg] = 1;
        	     }
        	     return jacobian;
            }
        };
    }
 
	public double[] getInitialGuess(ImageProcessor ip, Roi roi) {
		initialGuess = new double[PARAM_LENGTH];
	    Arrays.fill(initialGuess, 0);
	    
	    double[] centroid = CentroidFitterIP.fitCentroidandWidth(ip,roi, ip.getAutoThreshold());
	    	    
	    initialGuess[INDEX_X0] = centroid[INDEX_X0];
	    initialGuess[INDEX_Y0] = centroid[INDEX_Y0];

		double w0 = (params[INDEX_WX]+params[INDEX_WY])/2;
		double c = params[INDEX_C];
		double d = params[INDEX_D];
	    initialGuess[INDEX_Z0] = d*d*(centroid[INDEX_SY]*centroid[INDEX_SY]-centroid[INDEX_SX]*centroid[INDEX_SX])/(4*w0*w0*c)+params[INDEX_Mp];

	    initialGuess[INDEX_I0] = ip.getMax()-ip.getMin();
	    initialGuess[INDEX_Bg] = ip.getMin();
	    
		return initialGuess;
	}

	// /////////////////////////////////////////////////////////////
	// Math functions
	private static double erf(double x) {
		return Erf.erf(x);
	}

	private static double dErf(double x) {
		return 2 * FastMath.exp(-x * x) / FastMath.sqrt(FastMath.PI);
	}

	public double Ex(double x, double[] variables) {
		double tsx = sqrt2 * Sx(variables[INDEX_Z0]);
		double xm = x - variables[INDEX_X0] - 0.5;
		double xp = x - variables[INDEX_X0] + 0.5;
		return 0.5 * erf(xp / tsx) - 0.5 * erf(xm / tsx);
	}

	public double Ey(double y, double[] variables) {
		double tsy = sqrt2 * Sy(variables[INDEX_Z0]);
		double ym = y - variables[INDEX_Y0] - 0.5;
		double yp = y - variables[INDEX_Y0] + 0.5;
		return 0.5 * erf(yp / tsy) - 0.5 * erf(ym / tsy);
	}

	public double dEx(double x, double[] variables) {
		double xm = x - variables[INDEX_X0] - 0.5;
		double xp = x - variables[INDEX_X0] + 0.5;
		double tsx = sqrt2 * Sx(variables[INDEX_Z0]);
		return 0.5 * (dErf(xm / tsx) - dErf(xp / tsx)) / tsx;
	}

	public double dEy(double y, double[] variables) {
		double ym = y - variables[INDEX_Y0] - 0.5;
		double yp = y - variables[INDEX_Y0] + 0.5;
		double tsy = sqrt2 * Sy(variables[INDEX_Z0]);
		return 0.5 * (dErf(ym / tsy) - dErf(yp / tsy)) / tsy;
	}

	public double dEsx(double x, double[] variables) {
		double tsx = sqrt2 * Sx(variables[INDEX_Z0]);
		double xm = x - variables[INDEX_X0] - 0.5;
		double xp = x - variables[INDEX_X0] + 0.5;
		return 0.5 * (xm * dErf(xm / tsx) - xp * dErf(xp / tsx))
				/ Sx(variables[INDEX_Z0]) / tsx;
	}

	public double dEsy(double y, double[] variables) {
		double tsy = sqrt2 * Sy(variables[INDEX_Z0]);
		double ym = y - variables[INDEX_Y0] - 0.5;
		double yp = y - variables[INDEX_Y0] + 0.5;
		return 0.5 * (ym * dErf(ym / tsy) - yp * dErf(yp / tsy))
				/ Sy(variables[INDEX_Z0]) / tsy;
	}

	public double Sx(double z) {
		double b = (z - params[INDEX_C] - params[INDEX_Mp]) / params[INDEX_D];
		return params[INDEX_WX]
				* Math.sqrt(1 + b * b + params[INDEX_AX] * b * b * b
						+ params[INDEX_BX] * b * b * b * b);
	}

	public double Sy(double z) {
		double b = (z + params[INDEX_C] - params[INDEX_Mp]) / params[INDEX_D];
		return params[INDEX_WY]
				* Math.sqrt(1 + b * b + params[INDEX_AY] * b * b * b
						+ params[INDEX_BY] * b * b * b * b);
	}

	public double dSx(double z) {
		double value;

		double A = params[INDEX_AX];
		double B = params[INDEX_BX];
		double d = params[INDEX_D];
		double b = (z - params[INDEX_C] - params[INDEX_Mp]) / d;
		value = 0.5 * params[INDEX_WX] * params[INDEX_WX]
				* (2 * b / d + 3 * A * b * b / d + 4 * B * b * b * b / d)
				/ Sx(z);
		return value;
	}

	public double dSy(double z) {
		double value;
		double A = params[INDEX_AY];
		double B = params[INDEX_BY];
		double d = params[INDEX_D];
		double b = (z + params[INDEX_C] - params[INDEX_Mp]) / d;
		value = 0.5 * params[INDEX_WY] * params[INDEX_WY]
				* (2 * b / d + 3 * A * b * b / d + 4 * B * b * b * b / d)
				/ Sy(z);
		return value;
	}
	
	
}
