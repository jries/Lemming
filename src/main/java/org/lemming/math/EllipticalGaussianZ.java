package org.lemming.math;

import java.util.Map;

import org.apache.commons.math3.analysis.MultivariateMatrixFunction;
import org.apache.commons.math3.analysis.MultivariateVectorFunction;
import org.apache.commons.math3.analysis.polynomials.PolynomialSplineFunction;
import org.apache.commons.math3.optim.OptimizationData;
import org.apache.commons.math3.special.Erf;
import org.apache.commons.math3.util.FastMath;

class EllipticalGaussianZ implements OptimizationData {

	private final int[] xgrid;
	private final int[] ygrid;
	private final PolynomialSplineFunction psx;
	private final PolynomialSplineFunction psy;

	private static final int INDEX_X0 = 0;
	private static final int INDEX_Y0 = 1;
	private static final int INDEX_Z0 = 2;
	private static final int INDEX_I0 = 3;
	private static final int INDEX_Bg = 4;
	private static final int PARAM_LENGTH = 5;
	private static final double sqrt2 = FastMath.sqrt(2);

	public EllipticalGaussianZ(int[] xgrid, int[] ygrid, Map<String,Object> params){

		this.xgrid = xgrid;
		this.ygrid = ygrid;
		psx = (PolynomialSplineFunction) params.get("psx");
		psy = (PolynomialSplineFunction) params.get("psy");
	}

	private double getValue(double[] parameter, double x, double y) {
		return parameter[INDEX_I0] * Ex(x, parameter) * Ey(y, parameter) + parameter[INDEX_Bg];
	}

	public MultivariateVectorFunction getModelFunction() {
        return new MultivariateVectorFunction() {
            @Override
            public double[] value(double[] parameter) throws IllegalArgumentException {
                final double[] retVal = new double[xgrid.length];
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

            	 final double[][] jacobian = new double[xgrid.length][PARAM_LENGTH];
            	 final double dsx = dSx(point[INDEX_Z0]);
            	 final double dsy = dSy(point[INDEX_Z0]);
            	 
        	     for (int i = 0; i < xgrid.length; ++i) {
        	    	 final double ex = Ex(xgrid[i], point);
        	    	 final double ey = Ey(ygrid[i], point);
        	    	 jacobian[i][INDEX_X0] = point[INDEX_I0]*ey*dEx(xgrid[i],point);
        	    	 jacobian[i][INDEX_Y0] = point[INDEX_I0]*ex*dEy(ygrid[i],point); 
        	    	 jacobian[i][INDEX_Z0] = point[INDEX_I0]*(dEsx(xgrid[i],point)*ey*dsx + ex*dEsy(ygrid[i], point)*dsy);
        	    	 jacobian[i][INDEX_I0] = ex*ey;
        	    	 jacobian[i][INDEX_Bg] = 1;
        	     }
        	     return jacobian;
            }
        };
    }

	// /////////////////////////////////////////////////////////////
	// Math functions
	private static double erf(double x) {
		return Erf.erf(x);
	}

	private static double dErf(double x) {
		return 2 * FastMath.exp(-x * x) / FastMath.sqrt(FastMath.PI);
	}

	private double Ex(double x, double[] variables) {
		double tsx = sqrt2 * Sx(variables[INDEX_Z0]);
		double xm = x - variables[INDEX_X0] - 0.5;
		double xp = x - variables[INDEX_X0] + 0.5;
		return 0.5 * erf(xp / tsx) - 0.5 * erf(xm / tsx);
	}

	private double Ey(double y, double[] variables) {
		double tsy = sqrt2 * Sy(variables[INDEX_Z0]);
		double ym = y - variables[INDEX_Y0] - 0.5;
		double yp = y - variables[INDEX_Y0] + 0.5;
		return 0.5 * erf(yp / tsy) - 0.5 * erf(ym / tsy);
	}

	private double dEx(double x, double[] variables) {
		double xm = x - variables[INDEX_X0] - 0.5;
		double xp = x - variables[INDEX_X0] + 0.5;
		double tsx = sqrt2 * Sx(variables[INDEX_Z0]);
		return 0.5 * (dErf(xm / tsx) - dErf(xp / tsx)) / tsx;
	}

	private double dEy(double y, double[] variables) {
		double ym = y - variables[INDEX_Y0] - 0.5;
		double yp = y - variables[INDEX_Y0] + 0.5;
		double tsy = sqrt2 * Sy(variables[INDEX_Z0]);
		return 0.5 * (dErf(ym / tsy) - dErf(yp / tsy)) / tsy;
	}

	private double dEsx(double x, double[] variables) {
		double tsx = sqrt2 * Sx(variables[INDEX_Z0]);
		double xm = x - variables[INDEX_X0] - 0.5;
		double xp = x - variables[INDEX_X0] + 0.5;
		return 0.5 * (xm * dErf(xm / tsx) - xp * dErf(xp / tsx))
				/ Sx(variables[INDEX_Z0]) / tsx;
	}

	private double dEsy(double y, double[] variables) {
		double tsy = sqrt2 * Sy(variables[INDEX_Z0]);
		double ym = y - variables[INDEX_Y0] - 0.5;
		double yp = y - variables[INDEX_Y0] + 0.5;
		return 0.5 * (ym * dErf(ym / tsy) - yp * dErf(yp / tsy))
				/ Sy(variables[INDEX_Z0]) / tsy;
	}

	public double Sx(double z) {
		double valuex = 1.;
		if(psx.isValidPoint(z))
			valuex = psx.value(z);
		return valuex;
	}

	public double Sy(double z) {
		double valuey = 1.;
		if(psy.isValidPoint(z))
			valuey = psy.value(z);
		return valuey;
	}

	private double dSx(double z) {
		double value = 1.;
		if(psx.isValidPoint(z))
			value = psx.derivative().value(z);
		return value;
	}

	private double dSy(double z) {
		double value = 1.;
		if(psy.isValidPoint(z))
			value = psy.derivative().value(z);
		return value;
	}
}
