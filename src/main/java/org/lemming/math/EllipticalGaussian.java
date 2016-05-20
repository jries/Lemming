package org.lemming.math;

import org.apache.commons.math3.analysis.MultivariateMatrixFunction;
import org.apache.commons.math3.analysis.MultivariateVectorFunction;
import org.apache.commons.math3.optim.OptimizationData;
import org.apache.commons.math3.special.Erf;
import org.apache.commons.math3.util.FastMath;

/**
 * EllipticalGaussian Fitter using sigmax and sigmay intermediates to calculate Z from the calibration curve
 * 
 * @author Ronny Sczech
 *
 */
class EllipticalGaussian implements OptimizationData {
	private final int[] xgrid;
	private final int[] ygrid;

	private static final int INDEX_X0 = 0;
	private static final int INDEX_Y0 = 1;
	private static final int INDEX_SX = 2;
	private static final int INDEX_SY = 3;
	private static final int INDEX_I0 = 4;
	private static final int INDEX_Bg = 5;
	private static final int PARAM_LENGTH = 6;
	private static final double sqrt2 = Math.sqrt(2);
	
	public EllipticalGaussian(int[] xgrid, int[] ygrid){
		this.xgrid = xgrid;
		this.ygrid = ygrid;
	}
	
    private static double getValue(double[] params, double x, double y) {

        return params[INDEX_I0]*Ex(x,params)*Ey(y,params)+params[INDEX_Bg];
    }

	
    public MultivariateVectorFunction getModelFunction() {
        return new MultivariateVectorFunction() {
            @Override
            public double[] value(double[] params_) throws IllegalArgumentException {
            	final double[] retVal = new double[xgrid.length];
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

            	final double[][] jacobian = new double[xgrid.length][PARAM_LENGTH];
            	 
        	     for (int i = 0; i < xgrid.length; ++i) {
        	    	 final double ex = Ex(xgrid[i], point);
        	    	 final double ey = Ey(ygrid[i], point);
        	    	 jacobian[i][INDEX_X0] = point[INDEX_I0]*ey*dEx(xgrid[i],point);
        	    	 jacobian[i][INDEX_Y0] = point[INDEX_I0]*ex*dEy(ygrid[i],point);
        	    	 jacobian[i][INDEX_SX] = point[INDEX_I0]*ey*dEsx(xgrid[i],point);
        	    	 jacobian[i][INDEX_SY] = point[INDEX_I0]*ex*dEsy(ygrid[i],point);
        	    	 jacobian[i][INDEX_I0] = ex*ey;
        	    	 jacobian[i][INDEX_Bg] = 1;
        	     }
        	     
				return jacobian;
            }
        };
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
		double tsx = 1/(sqrt2*variables[INDEX_SX]);
		return 0.5*erf(tsx*(x-variables[INDEX_X0]+0.5))-0.5*erf(tsx*(x-variables[INDEX_X0]-0.5));
	}
	
	private static double Ey(double y, double[] variables){
		double tsy = 1/(sqrt2*variables[INDEX_SY]);
		return 0.5*erf(tsy*(y-variables[INDEX_Y0]+0.5))-0.5*erf(tsy*(y-variables[INDEX_Y0]-0.5));
	}	
	
	private static double dEx(double x, double[] variables){
		double tsx = 1/(sqrt2*variables[INDEX_SX]);
		return 0.5*tsx*(dErf(tsx*(x-variables[INDEX_X0]-0.5))-dErf(tsx*(x-variables[INDEX_X0]+0.5)));
	}
	
	private static double dEy(double y, double[] variables){
		double tsy = 1/(sqrt2*variables[INDEX_SY]);
		return 0.5*tsy*(dErf(tsy*(y-variables[INDEX_Y0]-0.5))-dErf(tsy*(y-variables[INDEX_Y0]+0.5)));
	}
	
	private static double dEsx(double x, double[] variables){
		double tsx = 1/(sqrt2*variables[INDEX_SX]);
		return 0.5*tsx*((x-variables[INDEX_X0]-0.5)*dErf(tsx*(x-variables[INDEX_X0]-0.5))-(x-variables[INDEX_X0]+0.5)*dErf(tsx*(x-variables[INDEX_X0]+0.5)))/variables[INDEX_SX];
	}
	
	private static double dEsy(double y, double[] variables){
		double tsy = 1/(sqrt2*variables[INDEX_SY]);
		return 0.5*tsy*((y-variables[INDEX_Y0]-0.5)*dErf(tsy*(y-variables[INDEX_Y0]-0.5))-(y-variables[INDEX_Y0]+0.5)*dErf(tsy*(y-variables[INDEX_Y0]+0.5)))/variables[INDEX_SY];
	}
}
