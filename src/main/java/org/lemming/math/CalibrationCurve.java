package org.lemming.math;

import org.apache.commons.math3.analysis.MultivariateMatrixFunction;
import org.apache.commons.math3.analysis.MultivariateVectorFunction;

/**
 * Fitter for the calibration curve
 * 
 * @author Ronny Sczech
 *
 */
public class CalibrationCurve {

	public static int INDEX_WX = 0;
	public static int INDEX_WY = 1;
	public static int INDEX_AX = 2;
	public static int INDEX_AY = 3;
	public static int INDEX_BX = 4;
	public static int INDEX_BY = 5;
	public static int INDEX_C = 6;
	public static int INDEX_D = 7;
	public static int INDEX_Mp = 8;
	public static int PARAM_LENGTH = 9;
	
    private double[] zgrid, w;
    private int length;
    private int minIndexx, minIndexy;
    private double minx, miny; 

    public CalibrationCurve(double[] z, double[] wx, double[] wy) {
    	length = z.length;
    	zgrid = new double[2*length];
    	java.lang.System.arraycopy(z, 0, zgrid, 0, length);
    	java.lang.System.arraycopy(z, 0, zgrid, length, length);
    	
    	w = new double[2*length];
    	java.lang.System.arraycopy(wx, 0, w, 0, length);
    	java.lang.System.arraycopy(wy, 0, w, length, length);

        minIndexx = findMinIndex(wx);
        minx = wx[minIndexx];
        minIndexy = findMinIndex(wy);
        miny = wy[minIndexy];
    }
    
    public static int findMinIndex(double[] A){
    	double min = A[0];
    	int index = 0;
    	for(int i=0;i<A.length;i++){
    		if(A[i] < min){
    			index = i;
    			min = A[i];
    		}
    	}
    	return index;
    }

	public double[] getInitialGuess() {
		double[] initialGuess = new double[PARAM_LENGTH];

	    initialGuess[INDEX_WX] = minx;
	    initialGuess[INDEX_WY] = miny;
	    initialGuess[INDEX_AX] = 0;
	    initialGuess[INDEX_AY] = 0;
	    initialGuess[INDEX_BX] = 0;
	    initialGuess[INDEX_BY] = 0;
	    initialGuess[INDEX_C] = 0.5*Math.abs(zgrid[minIndexx])-Math.abs(zgrid[minIndexy]);
	    initialGuess[INDEX_D] = Math.max(zgrid[minIndexx],zgrid[minIndexy])-Math.min(zgrid[minIndexx],zgrid[minIndexy]);		
	    initialGuess[INDEX_Mp] = Math.abs(zgrid[minIndexx])-Math.abs(zgrid[minIndexy])+Math.min(zgrid[minIndexx], zgrid[minIndexy]);	

		return initialGuess;
	}
    
	public double[] getTarget(){
		return w;
	}

    public double[] valuesWith(double[] params) {
        double[] values = new double[2*length];
        double b;
        for (int i = 0; i < length; ++i) {
    		b = (zgrid[i]-params[INDEX_C]-params[INDEX_Mp])/params[INDEX_D];
    		values[i] = params[INDEX_WX]*Math.sqrt(1+b*b+params[INDEX_AX]*b*b*b+params[INDEX_BX]*b*b*b*b);
        }
        for (int i = length; i < 2*length; ++i) {
    		b = (zgrid[i]+params[INDEX_C]-params[INDEX_Mp])/params[INDEX_D];
    		values[i] = params[INDEX_WY]*Math.sqrt(1+b*b+params[INDEX_AY]*b*b*b+params[INDEX_BY]*b*b*b*b);
        }
        return values;
    }
    
    public static double[] valuesWith(double z[], double[] params) {
        double[] values = new double[2*z.length];
        double b;
        
        for(int i=0;i<PARAM_LENGTH;i++){
        	System.out.println(params[i]);
        }
        
        for (int i = 0; i < z.length; ++i) {
    		b = (z[i]-params[INDEX_C]-params[INDEX_Mp])/params[INDEX_D];
    		values[i] = params[INDEX_WX]*Math.sqrt(1+b*b+params[INDEX_AX]*b*b*b+params[INDEX_BX]*b*b*b*b);
    		//System.out.println(values[i]);
        }
        for (int i = z.length; i < 2*z.length; ++i) {
    		b = (z[i-z.length]+params[INDEX_C]-params[INDEX_Mp])/params[INDEX_D];
    		values[i] = params[INDEX_WY]*Math.sqrt(1+b*b+params[INDEX_AY]*b*b*b+params[INDEX_BY]*b*b*b*b);
    		//System.out.println(values[i]);
        }

        return values;
    }
    
    public MultivariateVectorFunction getModelFunction() {
        return new MultivariateVectorFunction() {
            public double[] value(double[] params) {
                return valuesWith(params);
            }
        };
    }

    public MultivariateMatrixFunction getModelFunctionJacobian() {
        return new MultivariateMatrixFunction() {
            public double[][] value(double[] params) {
                double[][] jacobian = new double[2*length][PARAM_LENGTH];
                double b, denom ;
                for (int i = 0; i < length; ++i) {

           	   		 b = (zgrid[i]-params[INDEX_C]-params[INDEX_Mp])/params[INDEX_D];
        	   		 denom = Math.sqrt(1+b*b+params[INDEX_AX]*b*b*b+params[INDEX_BX]*b*b*b*b);

        	   		 jacobian[i][INDEX_WX] = denom;
        	   		 jacobian[i][INDEX_WY] = 0;
        	   		 jacobian[i][INDEX_AX] = 0.5*params[INDEX_WX]*b*b*b/denom;
        	   		 jacobian[i][INDEX_BX] = 0.5*params[INDEX_WX]*b*b*b*b/denom;
                	 jacobian[i][INDEX_AY] = 0;
                     jacobian[i][INDEX_BY] = 0;
        	   		 jacobian[i][INDEX_C] = -0.5*params[INDEX_WX]*(2*b+3*params[INDEX_AX]*b*b+4*params[INDEX_BX]*b*b*b)/(params[INDEX_D]*denom);
        	   		 jacobian[i][INDEX_D] = b*jacobian[i][INDEX_C];
        	   		 jacobian[i][INDEX_Mp] = jacobian[i][INDEX_C];
                }                
                for (int i = length; i < 2*length; ++i) {
          	   		 b = (zgrid[i]+params[INDEX_C]-params[INDEX_Mp])/params[INDEX_D];
	       	   		 denom = Math.sqrt(1+b*b+params[INDEX_AY]*b*b*b+params[INDEX_BY]*b*b*b*b);

	       	   		 jacobian[i][INDEX_WX] = 0;
	       	   		 jacobian[i][INDEX_WY] = denom;
        	   		 jacobian[i][INDEX_AX] = 0;
                     jacobian[i][INDEX_BX] = 0;
	       	   		 jacobian[i][INDEX_AY] = 0.5*params[INDEX_WY]*b*b*b/denom;
	       	   		 jacobian[i][INDEX_BY] = 0.5*params[INDEX_WY]*b*b*b*b/denom;
	       	   		 jacobian[i][INDEX_C] = 0.5*params[INDEX_WY]*(2*b+3*params[INDEX_AY]*b*b+4*params[INDEX_BY]*b*b*b)/(params[INDEX_D]*denom);
	       	   		 jacobian[i][INDEX_D] = -b*jacobian[i][INDEX_C];
        	   		 jacobian[i][INDEX_Mp] = -jacobian[i][INDEX_C];
               }

                return jacobian;
            }
        };
    }	    	
 }