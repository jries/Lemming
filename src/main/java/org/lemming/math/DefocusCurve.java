package org.lemming.math;

import org.apache.commons.math3.analysis.MultivariateMatrixFunction;
import org.apache.commons.math3.analysis.MultivariateVectorFunction;
import org.apache.commons.math3.optim.OptimizationData;

public class DefocusCurve implements OptimizationData {
	private static int INDEX_W0 = 0;
	private static int INDEX_A = 1;
	private static int INDEX_B = 2;
	private static int INDEX_C = 3;
	private static int INDEX_D = 4;
	private static int PARAM_LENGTH = 5;
	
    private double[] zgrid, w;
    private int minIndex;
    private double w0; 

    public DefocusCurve(double[] z, double[] w) {
        this.zgrid = z;
        this.w = w;
        
        minIndex = findMinIndex(w);
        w0 = w[minIndex];
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
	    
	    initialGuess[INDEX_W0] = w0;
	    initialGuess[INDEX_A] = 0;
	    initialGuess[INDEX_B] = 0;
	    initialGuess[INDEX_C] = zgrid[minIndex];
	    initialGuess[INDEX_D] = 10;
		
		return initialGuess;
	}
    
	public double[] getTarget(){
		return w;
	}

    public static double[] valuesWith(double[] z, double[] params) {
        double[] values = new double[z.length];
        double b;
        for (int i = 0; i < values.length; ++i) {
    		b = (z[i]-params[INDEX_C])/params[INDEX_D];
    		values[i] = params[INDEX_W0]*Math.sqrt(1+b*b+params[INDEX_A]*b*b*b+params[INDEX_B]*b*b*b*b);
        }
        return values;
    }

    public double[] valuesWith(double[] params) {
        double[] values = new double[zgrid.length];
        double b;
        for (int i = 0; i < values.length; ++i) {
    		b = (zgrid[i]-params[INDEX_C])/params[INDEX_D];
    		values[i] = params[INDEX_W0]*Math.sqrt(1+b*b+params[INDEX_A]*b*b*b+params[INDEX_B]*b*b*b*b);
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
                double[][] jacobian = new double[zgrid.length][PARAM_LENGTH];
                for (int i = 0; i < jacobian.length; ++i) {
           	   		 final double b = (zgrid[i]-params[INDEX_C])/params[INDEX_D];
        	   		 final double denom = Math.sqrt(1+b*b+params[INDEX_A]*b*b*b+params[INDEX_B]*b*b*b*b);
        	
        	   		 jacobian[i][INDEX_W0] = denom;
        	   		 jacobian[i][INDEX_A] = 0.5*params[INDEX_W0]*b*b*b/denom;
        	   		 jacobian[i][INDEX_B] = 0.5*params[INDEX_W0]*b*b*b*b/denom;
        	   		 jacobian[i][INDEX_C] = -0.5*params[INDEX_W0]*(2*b+3*params[INDEX_A]*b*b+4*params[INDEX_B]*b*b*b)/(params[INDEX_D]*denom);
        	   		 jacobian[i][INDEX_D] = b*jacobian[i][INDEX_C];
                }
                return jacobian;
            }
        };
    }	    	
}
