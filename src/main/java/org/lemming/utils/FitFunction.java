package org.lemming.utils;

/**
 * Interface for a fitting function
 * 
 * @author jborbely
 */
public interface FitFunction {
	
	/**
	 * The fitting function, for nD data.
	 * @param x - the x values (the independent variable)
	 * @param p - the values of the function parameters
	 * @param y - a list that is holding the y values (the value of the function
	 * evaluated at each {@code x} value). Does an in-place replacement of the
	 * value in {@code y} evaluated at each {@code x} value in order to avoid 
	 * creating a new double[] {@code y} array on every method call from the 
	 * fitting routine.
	 */
	public void fcn(double[][] x, double[] p, double[] y);

	/**
	 * The fitting function, for 1D data.
	 * @param x - the x values (the independent variable)
	 * @param p - the values of the function parameters
	 * @param y - a list that is holding the y values (the value of the function
	 * evaluated at each {@code x} value). Does an in-place replacement of the
	 * value in {@code y} evaluated at each {@code x} value in order to avoid 
	 * creating a new double[] {@code y} array on every method call from the 
	 * fitting routine.
	 */
	public void fcn(double[] x, double[] p, double[] y);

	/**
	 * The analytical form of the partial derivatives of the fitting function, for nD data.
	 * @param x - the x values (the independent variable)
	 * @param p - the values of the function parameters
	 * @param der - a {@code p.length} by {@code y.length} array that is holding 
	 * the values of the partial derivatives for each parameter that is evaluated 
	 * at each {@code x} value. Does an in-place replacement of the value in 
	 * {@code der} evaluated for each parameter and at each {@code x} value in 
	 * order to avoid creating a new double[][] {@code der} array on every method 
	 * call from the fitting routine.
	 */
	public void deriv(double[][] x, double[] p, double[][] der);

	/**
	 * The analytical form of the partial derivatives of the fitting function, for 1D data.
	 * @param x - the x values (the independent variable)
	 * @param p - the values of the function parameters
	 * @param der - a {@code p.length} by {@code y.length} array that is holding 
	 * the values of the partial derivatives for each parameter that is evaluated 
	 * at each {@code x} value. Does an in-place replacement of the value in 
	 * {@code der} evaluated for each parameter and at each {@code x} value in 
	 * order to avoid creating a new double[][] {@code der} array on every method 
	 * call from the fitting routine.
	 */
	public void deriv(double[] x, double[] p, double[][] der);

	/**
	 * Final check (for nD data) to ensure that the best-guess parameters from
	 * the fit make sense or if you want to force a parameter to be within a 
	 * certain range (e.g., if one of the fitting parameters is an angle and 
	 * its final best-guess value is not between -&#960; and &#960; you can 
	 * correct for its value to be within this range by adding/subtracting 
	 * 2&#960;n).<br><br>You can specify code in this method to modify the 
	 * values in {@code p}. If you don't want to modify {@code p} in any way
	 * or if your data set has dimension = 1 then you can just ignore this method.
	 * @param x - the x values (the independent variable)
	 * @param y - the y values (the dependent variable) 
	 * @param p - the values of the function parameters.
	 */	
	public void finalCheck(double[][] x, double[] y, double[] p);

	/**
	 * Final check (for 1D data) to ensure that the best-guess parameters from 
	 * the fit make sense or if you want to force a parameter to be within a 
	 * certain range (e.g., if one of the fitting parameters is an angle and 
	 * its final best-guess value is not between -&#960; and &#960; you can 
	 * correct for its value to be within this range by adding/subtracting 
	 * 2&#960;n).<br><br>You can specify code in this method to modify the 
	 * values in {@code p}. If you don't want to modify {@code p} in any way 
	 * or if your data set has dimensions greater than 1 then you can just ignore this method.
	 * @param x - the x values (the independent variable)
	 * @param y - the y values (the dependent variable) 
	 * @param p - the values of the function parameters.
	 */	
	public void finalCheck(double[] x, double[] y, double[] p);

	/**
	 * Checks to ensure that none of the fit parameters get an unrealistic 
	 * (unwanted) value. If a parameter value does start to converge to an 
	 * unrealistic (unwanted) value then it can either be set back to its 
	 * initial-guess value or modified in another way (e.g., if one of the 
	 * fitting parameters is the amplitude of a cosine function and you want
	 * its value to always be positive so that you don't have to worry about
	 * applying &#960; phase shifts to compensate for a negative amplitude)

	 * @param p - the current values of the function parameters. You can specify
	 * code in this method to replace unrealistic(unwanted) values by the 
	 * appropriate values specified by you. If you don't want to modify {@code p}
	 * in any way then you can just ignore this method.
	 * @param pInitial - the initial guess of the parameters
	 */
	public void pCheck(double[] p, double[] pInitial);
	
}
