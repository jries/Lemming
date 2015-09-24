package org.lemming.math;

import java.util.Arrays;

import ij.gui.Roi;
import ij.process.ImageProcessor;
import ij.process.ImageStatistics;

import org.apache.commons.math3.analysis.MultivariateMatrixFunction;
import org.apache.commons.math3.analysis.MultivariateVectorFunction;
import org.apache.commons.math3.optim.OptimizationData;
import org.apache.commons.math3.special.Erf;
import org.apache.commons.math3.util.FastMath;

public class Gaussian implements OptimizationData {
	private int[] xgrid, ygrid;
	double[] params;
	double[] initialGuess;
	
	boolean calibration = false;

	private static int INDEX_X0 = 0;
	private static int INDEX_Y0 = 1;
	private static int INDEX_S = 2;
	private static int INDEX_I0 = 3;
	private static int INDEX_Bg = 4;
	private static int PARAM_LENGTH = 5;
	private static double sqrt2 = Math.sqrt(2);
	
	//private static double defaultSigma = 1.5;

	public Gaussian(){
	}
	
	public Gaussian(int[] xgrid, int[] ygrid){
		this.xgrid = xgrid;
		this.ygrid = ygrid;
	}
	
    public static double getValue(double[] params, double x, double y) {
    	//System.out.println("val : "+(params[INDEX_I0]*Ex(x,params)*Ey(y,params)+params[INDEX_Bg]));		
        return params[INDEX_I0]*Ex(x,params)*Ey(y,params)+params[INDEX_Bg];
    }

	
    public MultivariateVectorFunction getModelFunction() {
        return new MultivariateVectorFunction() {
            @Override
            public double[] value(double[] params_) throws IllegalArgumentException {
                double[] retVal = new double[xgrid.length];
                for(int i = 0; i < xgrid.length; i++) {
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

            	 double[][] jacobian = new double[xgrid.length][PARAM_LENGTH];
            	 
        	     for (int i = 0; i < xgrid.length; ++i) {        	    	 
        	    	 jacobian[i][INDEX_X0] = point[INDEX_I0]*Ey(ygrid[i], point)*dEx(xgrid[i],point);
        	    	 jacobian[i][INDEX_Y0] = point[INDEX_I0]*Ex(xgrid[i], point)*dEy(ygrid[i],point);
        	    	 jacobian[i][INDEX_S] = point[INDEX_I0]*(Ey(ygrid[i], point)*dEsx(xgrid[i],point)+Ex(xgrid[i], point)*dEsy(ygrid[i],point));
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
	    //ImageProcessor ip = ip_.duplicate();
	    ip.setRoi(roi);
	    //double[] centroid = CentroidFitterAlternative.fitCentroidandWidth(ip,roi, ip.getAutoThreshold());
	    ImageStatistics stat = ip.getStatistics();
	    
	    initialGuess[INDEX_X0] = stat.xCenterOfMass;
	    initialGuess[INDEX_Y0] = stat.yCenterOfMass;

	    initialGuess[INDEX_S] = Math.abs(stat.skewness);					///// skewness does not seem relevant here std in x and y are...
	        
	    initialGuess[INDEX_I0] = ip.getMax()-ip.getMin(); 
	    initialGuess[INDEX_Bg] = stat.median;
		
		return initialGuess;
	}

	///////////////////////////////////////////////////////////////
	// Math functions
	private static double erf(double x) {
		return Erf.erf(x);
	}
	
	private static double dErf(double x){
		return 2*FastMath.exp(-x*x)/FastMath.sqrt(FastMath.PI);
	}

	private static double Ex(double x, double[] variables){
		double tsx = 1/(sqrt2*variables[INDEX_S]);
		return 0.5*erf(tsx*(x-variables[INDEX_X0]+0.5))-0.5*erf(tsx*(x-variables[INDEX_X0]-0.5));
	}
	
	private static double Ey(double y, double[] variables){
		double tsy = 1/(sqrt2*variables[INDEX_S]);
		return 0.5*erf(tsy*(y-variables[INDEX_Y0]+0.5))-0.5*erf(tsy*(y-variables[INDEX_Y0]-0.5));
	}	
	
	private static double dEx(double x, double[] variables){
		double tsx = 1/(sqrt2*variables[INDEX_S]);
		return 0.5*tsx*(dErf(tsx*(x-variables[INDEX_X0]-0.5))-dErf(tsx*(x-variables[INDEX_X0]+0.5)));
	}
	
	private static double dEy(double y, double[] variables){
		double tsy = 1/(sqrt2*variables[INDEX_S]);
		return 0.5*tsy*(dErf(tsy*(y-variables[INDEX_Y0]-0.5))-dErf(tsy*(y-variables[INDEX_Y0]+0.5)));
	}
	
	private static double dEsx(double x, double[] variables){
		double tsx = 1/(sqrt2*variables[INDEX_S]);
		return 0.5*tsx*((x-variables[INDEX_X0]-0.5)*dErf(tsx*(x-variables[INDEX_X0]-0.5))-(x-variables[INDEX_X0]+0.5)*dErf(tsx*(x-variables[INDEX_X0]+0.5)))/variables[INDEX_S];
	}
	
	private static double dEsy(double y, double[] variables){
		double tsy = 1/(sqrt2*variables[INDEX_S]);
		return 0.5*tsy*((y-variables[INDEX_Y0]-0.5)*dErf(tsy*(y-variables[INDEX_Y0]-0.5))-(y-variables[INDEX_Y0]+0.5)*dErf(tsy*(y-variables[INDEX_Y0]+0.5)))/variables[INDEX_S];
	}
}
