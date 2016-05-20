package org.lemming.math;

import org.apache.commons.math3.analysis.MultivariateMatrixFunction;
import org.apache.commons.math3.analysis.MultivariateVectorFunction;
import org.apache.commons.math3.optim.OptimizationData;
import org.apache.commons.math3.special.Erf;
import org.apache.commons.math3.util.FastMath;

/** ThunderSTORM algorithm for a symmetric Gaussian*/
class SymmetricGaussian implements OptimizationData {
	private final int[] xgrid;
	private final int[] ygrid;

	private static final int INDEX_X0 = 0;
	private static final int INDEX_Y0 = 1;
	private static final int INDEX_S = 2;
	private static final int INDEX_I0 = 3;
	private static final int INDEX_Bg = 4;
	private static final int PARAM_LENGTH = 5;
	private static final double sqrt2 = Math.sqrt(2);

	public SymmetricGaussian(int[] xgrid, int[] ygrid) {
		this.xgrid = xgrid;
		this.ygrid = ygrid;
	}

	private static double getValue(double[] params, double x, double y) {
		double ts = 1/(sqrt2*params[INDEX_S]);
		return params[INDEX_I0] * E(x, params[INDEX_X0], ts) * E(y, params[INDEX_Y0], ts) + params[INDEX_Bg];
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

	                final double[][] jacobian = new double[xgrid.length][PARAM_LENGTH];
	
	                final double ts = 1/(sqrt2*point[INDEX_S]);
	
	                 for (int i = 0; i < xgrid.length; ++i) {
	                     final double ex = E(xgrid[i], point[INDEX_X0], ts);
	                     final double ey = E(ygrid[i], point[INDEX_Y0], ts);
	                     jacobian[i][INDEX_X0] = point[INDEX_I0]*ey*dE(xgrid[i],point[INDEX_X0],ts);
	                     jacobian[i][INDEX_Y0] = point[INDEX_I0]*ex*dE(ygrid[i],point[INDEX_Y0],ts);
	                     jacobian[i][INDEX_S]  = point[INDEX_I0]*(ey*dEs(xgrid[i],point[INDEX_X0],ts,point[INDEX_S]) + ex*dEs(ygrid[i],point[INDEX_Y0],ts,point[INDEX_S]));
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
	
	private static double E(double x, double x0, double ts){
		return 0.5*erf(ts*(x-x0+0.5))-0.5*erf(ts*(x-x0-0.5));
	}
	
	private static double dE(double x,  double x0, double ts){
		return 0.5*ts*(dErf(ts*(x-x0-0.5))-dErf(ts*(x-x0+0.5)));
	}
	
	private static double dEs(double x, double x0, double ts, double s){
		return 0.5*ts*((x-x0-0.5)*dErf(ts*(x-x0-0.5))-(x-x0+0.5)*dErf(ts*(x-x0+0.5)))/s;
	}
}
