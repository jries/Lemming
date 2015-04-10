package org.lemming.utils;

/**
 * <p>Program to do a least-squares fit to a non-linear function with a
 * linearization of the fitting function. This gradient-expansion algorithm
 * (a.k.a Levenberg-Marquardt method) is taken from the book <i>Data 
 * Reduction and Error Analysis for Physical Sciences, by P.R. Bevington</i></p>  
 * Allows for
 * <ul> 
 * <li>each function parameter to be fixed/floated independently</li>
 * <li>specifying a weight for each data point</li>
 * </ul>
 * <p>The original FORTRAN version of this code was written by EAH</p>
 * 
 * pFloat - specifies which parameters in {@code p} are allowed to vary (float) or are fixed during the fitting process, 0=fixed
 * or 1=float. For example, if there are 4 parameters in the {@code function} 
 * and pFloat=[1,1,0,1] then parameters 0, 1 and 3 are allowed to vary (float) 
 * while parameter 2 is fixed at the value specified in {@code p[2]} - specifies which parameters in {@code p} are 
 * allowed to vary (float) or are fixed during the fitting process, 0=fixed
 * or 1=float. For example, if there are 4 parameters in the {@code function} 
 * and pFloat=[1,1,0,1] then parameters 0, 1 and 3 are allowed to vary (float) 
 * while parameter 2 is fixed at the value specified in {@code p[2]}<br><br>
 * 
 * xmin, xmax (for 1D) or double[] <b>xxmin, xxmax</b> (for nD) -
 * if these values are specified then the {@code function} is evaluated using 
 * the best-fit parameters for each value in {@code fitX} and the function-evaluated values
 * are stored within the {@code double[] fitY} variable. For nD data, the values would be
 * {@code xxmin=[xmin1, ..., xminN]} and {@code xxmax=[xmax1, ..., xmaxN]} <br><br>
 * 
 * <p>This class also produces the following variables</p>
 * <ul>
 * <li> double[] <b>pBest</b> - the parameters that best fit the data to minimize
 * the chi square</li>
 * <li> double[] <b>dpBest</b> - the uncertainty for each parameter, if the parameter 
 * was fixed in the fit then its uncertainty will be 0.0</li>
 * <li> double[][] <b>der</b> - a {@code p.length} by {@code y.length} array 
 * containing the values of the partial derivatives</li>
 * </ul>
 * 
 * <p>This class may also produce the following variables if requested for in the constructor</p>
 * <ul>
 * <li> double[] <b>fitX</b> (for 1D) or double[][] <b>fitXX</b> (for nD) - 
 * for 1D data - values from {@code xmin} to {@code xmax} (inclusive) using {@code nStep} steps
 * for nD data - values from {@code xmin=[xmin1, ..., xminN]} to {@code xmax=[xmax1, ..., xmaxN]} 
 * (inclusive) using {@code nStep} steps</li>
 * <li> double[] <b>fitY</b> - for each value in {@code fitX} the corresponding 
 * function-evaluated values are calculated using the best-fit parameters</li>
 * <li> double[] <b>currentY</b> - the function evaluated at each x value using the 
 * latest best-fit parameters</li>
 * <li> double[] <b>residuals</b> - the values of the residuals 
 * ({@code y - currentY})</li>
 * </ul>
 * @author Joe Borbely
 */
public class LevMarBev implements Runnable {

	private boolean is1D; // is true if x is a double[], is false if x is a double[][]
	private boolean goto_21; // this boolean value is used to replace a FORTRAN 'goto' statement in the original code
	private boolean[] precisionAcheived; // determines if the requested precision was achieved for each fit parameter
	private int iter; // the current iteration number
	private int npts; // the length of the x and y arrays
	private int nTerms; // the number of fit parameters in the function
	private int nfl; // the number of floating parameters in the fit
	private int[] ifl, ik, jk; // used for specifying the parameters indices that are allowed to vary (are floating) (used in the matrix inversion method)
	private double nFree; // the number of free parameters, i.e., npts - nfl (it's of type double because it used to calculate the reduced chisqr)
	private double det; // the determinant of the fit
	private double chiSqr; // the reduced chi-square of the latest fitting iteration
	private double chiSqrOld; // the reduced chi-square of the previous fitting iteration
	private double amax; // the
	private double temp; // holds temporary values during the matrix inversion
	private double lambda = 0.001; // the damping parameter
	private double[] weights; // the weights, i.e., 1/dy^2
	private double[] currentParamPrecision; // the precision of each fitting parameter is calculated in each fitting iteration
	private double[] currentY; // the evaluation of the function for the current parameter values
	private double[] beta; // beta is the "curvature" matrix of chi squared, see p.224 of Bevington
	private double[] b; // holds temporary parameter values
	private double[][] array; // the inverted modified curvature matrix
	private double[][] alpha; // alpha is the "curvature" matrix of chi squared, see p.224 of Bevington

	// these variables are defined in the javadoc
	private FitFunction function;
	private boolean calcRes, verbose;
	private int nSteps, maxIter;
	private double precision, xmin, xmax;
	private byte[] pFloat;
	private double[] x, y, dy, p, pBest, dpBest, pInitial, xxmin, xxmax, residuals, fitY, fitX; 
	private double[][] xx, fitXX, der;	
	
	/** see {@link org.lemming.utils.LevMarBev LevMarBev} 
	 * @param function - the fitting function to use 
	 * @param x  - the x values 
	 *  @param y - the y values (the dependent variable) 
	 * @param p - the initial guess 
	 * @param pFloat - specifies which parameters in {@code p} are allowed to vary (float) or are fixed during the fitting process, 0=fixed
 * or 1=float. For example, if there are 4 parameters in the {@code function} 
 * and pFloat=[1,1,0,1] then parameters 0, 1 and 3 are allowed to vary (float) 
 * while parameter 2 is fixed at the value specified in {@code p[2]} 
	 * @param dy - the uncertainty for each value in {@code y}, used as the weights (1/dy)^2 
	 * @param maxIter - the maximum number of fitting iterations that are allowed 
	 * @param p - the initial guessrecision 
	 * @param x  - the x valuesmin 
	 * @param x  - the x valuesmax 
	 * @param precision - keep iterating the fit until this relative precision is reached
	 * @param xmin - {@code xmin=[xmin1, ..., xminN]}
	 * @param xmax - {@code xmax=[xmax1, ..., xmaxN]}
	 * @param nSteps - the number of evenly-spaced numbers to use to generate the 
 * {@code double[] fitX} variable (i.e., {@code fitX} goes from {@code xmin} to {@code xmax} 
 * (inclusive) using {@code nSteps} steps) 
	 * @param calcRes - specify whether to calculate residuals
	 * @param verbose - specify whether to display warning messages. */
	public LevMarBev(FitFunction function, double[][] x, double[] y, double[] p, byte[] pFloat, double[] dy, int maxIter, double precision, double[] xmin, double[] xmax, int nSteps, boolean calcRes, boolean verbose) {
		is1D = false;
		this.function = function;
		xx = x;
		this.y = y;
		this.p = p;
		this.pFloat = pFloat;
		this.dy = dy;
		this.maxIter = maxIter;
		this.precision = precision;
		xxmin = xmin;
		xxmax = xmax;
		this.nSteps = nSteps;
		this.calcRes = calcRes;
		this.verbose = verbose;
		initialize();
	}

	/** see {@link org.lemming.utils.LevMarBev LevMarBev} 
	 * @param function - the fitting function to use 
	 * @param x  - the x values 
	 *  @param y - the y values (the dependent variable) 
	 * @param p - the initial guess 
	 * @param pFloat - specifies which parameters in {@code p} are allowed to vary (float) or are fixed during the fitting process, 0=fixed
 * or 1=float. For example, if there are 4 parameters in the {@code function} 
 * and pFloat=[1,1,0,1] then parameters 0, 1 and 3 are allowed to vary (float) 
 * while parameter 2 is fixed at the value specified in {@code p[2]} 
	 * @param dy - the uncertainty for each value in {@code y}, used as the weights (1/dy)^2 
	 * @param maxIter - the maximum number of fitting iterations that are allowed 
	 * @param p - the initial guessrecision 
	 * @param x  - the x valuesmin 
	 * @param x  - the x valuesmax 
	 * @param precision - keep iterating the fit until this relative precision is reached
	 * @param xmin - {@code xmin=[xmin1, ..., xminN]}
	 * @param xmax - {@code xmax=[xmax1, ..., xmaxN]}
	 * @param nSteps - the number of evenly-spaced numbers to use to generate the 
 * {@code double[] fitX} variable (i.e., {@code fitX} goes from {@code xmin} to {@code xmax} 
 * (inclusive) using {@code nSteps} steps) 
	 * @param calcRes - specify whether to calculate residuals.
	 * @param verbose - specify whether to display warning messages. */
	public LevMarBev(FitFunction function, double[] x, double[] y, double[] p, byte[] pFloat, double[] dy, int maxIter, double precision, double xmin, double xmax, int nSteps, boolean calcRes, boolean verbose) {
		is1D = true;
		this.function = function;
		this.x = x;
		this.y = y;
		this.p = p;
		this.pFloat = pFloat;
		this.dy = dy;
		this.maxIter = maxIter;
		this.precision = precision;
		this.xmin = xmin;
		this.xmax = xmax;
		this.nSteps = nSteps;
		this.calcRes = calcRes;
		this.verbose = verbose;
		initialize();
	}	

	/** see {@link org.lemming.utils.LevMarBev LevMarBev} 
	 * @param function - the fitting function to use 
	 * @param x  - the x values 
	 *  @param y - the y values (the dependent variable) 
	 * @param p - the initial guess */
	public LevMarBev(FitFunction function, double[][] x, double[] y, double[] p) {
		this(function, x, y, p, new byte[0], new double[0], 100, 1.0e-6, new double[0], new double[0], 1, false, false);		
	}

	/** see {@link org.lemming.utils.LevMarBev LevMarBev} 
	 * @param function - the fitting function to use 
	 * @param x  - the x values 
	 *  @param y - the y values (the dependent variable) 
	 * @param p - the initial guess 
	 * @param pFloat - specifies which parameters in {@code p} are allowed to vary (float) or are fixed during the fitting process, 0=fixed
 * or 1=float. For example, if there are 4 parameters in the {@code function} 
 * and pFloat=[1,1,0,1] then parameters 0, 1 and 3 are allowed to vary (float) 
 * while parameter 2 is fixed at the value specified in {@code p[2]} */
	public LevMarBev(FitFunction function, double[][] x, double[] y, double[] p, byte[] pFloat) {
		this(function, x, y, p, pFloat, new double[0], 100, 1.0e-6, new double[0], new double[0], 1, false, false);		
	}

	/** see {@link org.lemming.utils.LevMarBev LevMarBev} 
	 * @param function - the fitting function to use 
	 * @param x  - the x values 
	 *  @param y - the y values (the dependent variable) 
	 * @param p - the initial guess 
	 * @param pFloat - specifies which parameters in {@code p} are allowed to vary (float) or are fixed during the fitting process, 0=fixed
 * or 1=float. For example, if there are 4 parameters in the {@code function} 
 * and pFloat=[1,1,0,1] then parameters 0, 1 and 3 are allowed to vary (float) 
 * while parameter 2 is fixed at the value specified in {@code p[2]} 
	 * @param dy - the uncertainty for each value in {@code y}, used as the weights (1/dy)^2 */
	public LevMarBev(FitFunction function, double[][] x, double[] y, double[] p, byte[] pFloat, double[] dy) {
		this(function, x, y, p, pFloat, dy, 100, 1.0e-6, new double[0], new double[0], 1, false, false);		
	}

	/** see {@link org.lemming.utils.LevMarBev LevMarBev} 
	 * @param function - the fitting function to use 
	 * @param x  - the x values 
	 *  @param y - the y values (the dependent variable) 
	 * @param p - the initial guess 
	 * @param pFloat - specifies which parameters in {@code p} are allowed to vary (float) or are fixed during the fitting process, 0=fixed
 * or 1=float. For example, if there are 4 parameters in the {@code function} 
 * and pFloat=[1,1,0,1] then parameters 0, 1 and 3 are allowed to vary (float) 
 * while parameter 2 is fixed at the value specified in {@code p[2]} 
	 * @param dy - the uncertainty for each value in {@code y}, used as the weights (1/dy)^2 
	 * @param maxIter - the maximum number of fitting iterations that are allowed 
	 * @param p - the initial guessrecision 
	 * @param precision - keep iterating the fit until this relative precision is reached */
	public LevMarBev(FitFunction function, double[][] x, double[] y, double[] p, byte[] pFloat, double[] dy, int maxIter, double precision) {
		this(function, x, y, p, pFloat, dy, maxIter, precision, new double[0], new double[0], 1, false, false);		
	}

	/** see {@link org.lemming.utils.LevMarBev LevMarBev} 
	 * @param function - the fitting function to use 
	 * @param x  - the x values 
	 *  @param y - the y values (the dependent variable) 
	 * @param p - the initial guess */
	public LevMarBev(FitFunction function, double[] x, double[] y, double[] p) {
		this(function, x, y, p, new byte[0], new double[0], 100, 1.0e-6, 0.0, 0.0, 1, false, false);		
	}

	/** see {@link org.lemming.utils.LevMarBev LevMarBev} 
	 * @param function - the fitting function to use 
	 * @param x  - the x values 
	 *  @param y - the y values (the dependent variable) 
	 * @param p - the initial guess 
	 * @param pFloat - specifies which parameters in {@code p} are allowed to vary (float) or are fixed during the fitting process, 0=fixed
 * or 1=float. For example, if there are 4 parameters in the {@code function} 
 * and pFloat=[1,1,0,1] then parameters 0, 1 and 3 are allowed to vary (float) 
 * while parameter 2 is fixed at the value specified in {@code p[2]} */
	public LevMarBev(FitFunction function, double[] x, double[] y, double[] p, byte[] pFloat) {
		this(function, x, y, p, pFloat, new double[0], 100, 1.0e-6, 0.0, 0.0, 1, false, false);		
	}

	/** see {@link org.lemming.utils.LevMarBev LevMarBev} 
	 * @param function - the fitting function to use 
	 * @param x  - the x values 
	 *  @param y - the y values (the dependent variable) 
	 * @param p - the initial guess 
	 * @param pFloat - specifies which parameters in {@code p} are allowed to vary (float) or are fixed during the fitting process, 0=fixed
 * or 1=float. For example, if there are 4 parameters in the {@code function} 
 * and pFloat=[1,1,0,1] then parameters 0, 1 and 3 are allowed to vary (float) 
 * while parameter 2 is fixed at the value specified in {@code p[2]} 
	 * @param dy - the uncertainty for each value in {@code y}, used as the weights (1/dy)^2 */
	public LevMarBev(FitFunction function, double[] x, double[] y, double[] p, byte[] pFloat, double[] dy) {
		this(function, x, y, p, pFloat, dy, 100, 1.0e-6, 0.0, 0.0, 1, false, false);		
	}

	/** see {@link org.lemming.utils.LevMarBev LevMarBev} 
	 * @param function - the fitting function to use 
	 * @param x  - the x values 
	 *  @param y - the y values (the dependent variable) 
	 * @param p - the initial guess 
	 * @param pFloat - specifies which parameters in {@code p} are allowed to vary (float) or are fixed during the fitting process, 0=fixed
 * or 1=float. For example, if there are 4 parameters in the {@code function} 
 * and pFloat=[1,1,0,1] then parameters 0, 1 and 3 are allowed to vary (float) 
 * while parameter 2 is fixed at the value specified in {@code p[2]} 
	 * @param dy - the uncertainty for each value in {@code y}, used as the weights (1/dy)^2 
	 * @param maxIter - the maximum number of fitting iterations that are allowed 
	 * @param precision - keep iterating the fit until this relative precision is reached */
	public LevMarBev(FitFunction function, double[] x, double[] y, double[] p, byte[] pFloat, double[] dy, int maxIter, double precision) {
		this(function, x, y, p, pFloat, dy, maxIter, precision, 0.0, 0.0, 1, false, false);		
	}

	/**  
	 * @return Returns the best fit parameters*/
	public double[] getBestParameters(){
		return pBest;
	}

	/**  
	 * @return Returns the uncertainty of the best fit parameters*/
	public double[] getBestParametersUncertainty(){
		return dpBest;
	}

	/**  
	 * @return Returns the value of the reduced chi square*/
	public double getReducedChiSquare(){
		return chiSqr;
	}

	/**  
	 * @return Returns the residuals (y - currentY)*/
	public double[] getResuiduals(){
		for (int i=0; i<npts; i++)
			residuals[i] = (y[i]-currentY[i]);
		return residuals;
	}
	
	/** Returns the function evaluated for each value in {@code fitX} using
	 * the parameters of best fit 
	 * @return function*/
	public double[] getFitY(){
		return fitY;
	}

	/** Returns the user specified x range from {@code xmin} to {@code xmax} 
	 * (inclusive) using {@code nStep} steps 
	 * @return user specified x*/
	public double[] getFitX(){
		return fitX;
	}

	/** Returns the user specified x range from values from values from 
	 * {@code xmin=[xmin1, ..., xminN]} to {@code xmax=[xmax1, ..., xmaxN]}
	 * (inclusive) using {@code nStep} steps 
	 * @return the user specified x range
	 */
	public double[][] getFitXX(){
		return fitXX;
	}

	/** set whether each parameter should be fixed (0) or allowed to float (1) during the fit 
	 * @param pFloat - specifies which parameters in {@code p} are allowed to vary (float) or are fixed during the fitting process, 0=fixed
 * or 1=float. For example, if there are 4 parameters in the {@code function} 
 * and pFloat=[1,1,0,1] then parameters 0, 1 and 3 are allowed to vary (float) 
 * while parameter 2 is fixed at the value specified in {@code p[2]} */
	public void setPfloat(byte[] pFloat) {
		if (pFloat == null) {
			LemMING.error("LevMarBev error :: the pFloat-array is null");
		} else if (pFloat.length == 0) {
			nfl = nTerms;
			this.pFloat = new byte[nfl];
			ifl = new int[nfl];
			ik = new int[nfl];
			jk = new int[nfl];
			beta = new double[nfl];
			alpha = new double[nfl][nfl];
			array = new double[nfl][nfl];
			for (int i=0; i<nfl; i++) {
				this.pFloat[i] = 1; // allow all function parameters to float
				ifl[i] = i;
			}
		} else if (nTerms != pFloat.length) { 
			LemMING.error(String.format("LevMarBev error :: the length of p (%d) and pFloat (%d) are not equal", nTerms, pFloat.length));
		} else {
			this.pFloat = pFloat;
			nfl = 0;
			ifl = new int[nTerms];
			for (int i=0; i<nTerms; i++) {
				if (pFloat[i] != 0){
					ifl[nfl] = i;
					nfl += 1;
				}
			}
			ik = new int[nfl-1];
			jk = new int[nfl-1];
			beta = new double[nfl-1];
			alpha = new double[nfl-1][nfl-1];
			array = new double[nfl-1][nfl-1];
		}
		nFree = (double)(npts - nfl);
	    if (nFree <= 0.0) LemMING.error("LevMarBev error :: the number of free parameters is <= 0 (i.e., there are more floating fitting variables than data points)");
	}
	
	/** set the values of {@code dy} array 
	 * @param dy - the uncertainty for each value in {@code y}, used as the weights (1/dy)^2 */
	public void setDy(double[] dy) {
		if (dy==null) {
			LemMING.error("LevMarBev error :: the dy-array is null");
		} else if (dy.length == 0) {
			this.dy = new double[npts];
			for (int i=0; i<npts ; i++)
				// pick a y-uncertainty that would stand out as being strange so that you know it was manually entered.
				// using a value of 1.0 or 9999.9999 does not change the final uncertainty of each fit parameter 
				// it only makes the reduced chisq value extremely small, implying that the error bars are way too big
				dy[i] = 9999.9999;
		} else if (npts != dy.length) {
			LemMING.error(String.format("LevMarBev error :: the length of y (%d) and dy (%d) are not equal", npts, dy.length));
		} else {
			this.dy=dy;
		}
		weights = new double[npts];
		double val;
		for (int i=0; i<npts; i++) {
			val = dy[i];
			weights[i] = 1.0/(val*val);
		}
	}
	
	/**  
	 * @param maxIter - the maximum number of fitting iterations that are allowed */
	public void setMaxIter(int maxIter) {
		this.maxIter=maxIter;
	}

	/** set the precision to achieve for each parameter that is floating 
	 * @param precision - keep iterating the fit until this relative precision is reached */
	public void setPrecision(double precision) {
		this.precision=precision;
	}

	/** 
	 * @param calcRes - specify whether to calculate the residuals*/
	public void setCalcRes(boolean calcRes) {
		this.calcRes=calcRes;
		if (calcRes) {
			residuals = new double[npts];
		} else {
			residuals = new double[0];
		}
	}

	/** specify whether to display warning messages, if any occur 
	 * @param verbose - specify whether to display warning messages. - switch */
	public void setVerbose(boolean verbose) {
		this.verbose=verbose;
	}

	/** set the values of {@code fitX} to the values from {@code xmin} 
	 * to {@code xmax} (inclusive) using {@code nStep} steps 
	 * @param nSteps - the number of evenly-spaced numbers to use to generate the 
 * {@code double[] fitX} variable (i.e., {@code fitX} goes from {@code xmin} to {@code xmax} 
 * (inclusive) using {@code nSteps} steps) 
	 * @param xmin - {@code xmin=[xmin1, ..., xminN]}
	 * @param xmax - {@code xmax=[xmax1, ..., xmaxN]} */
	public void setFitX(int nSteps, double xmin, double xmax) {
		this.nSteps = nSteps;
		this.xmin = xmin;
		this.xmax = xmax;
		if (xmin!=xmax)	{
			fitX = new double[nSteps];
			fitY = new double[nSteps];
			double delta = (xmax-xmin)/(nSteps-1.0);
			for (int i=0; i<nSteps; i++)
				fitX[i] = i*delta + xmin;
		} else {
			fitX = new double[0];
			fitY = new double[0];
		}
	}

	/** set the values of {@code fitX} to values from 
	 * {@code xmin=[xmin1, ..., xminN]} to {@code xmax=[xmax1, ..., xmaxN]}
	 * (inclusive) using {@code nStep} steps 
	 * @param nSteps - the number of evenly-spaced numbers to use to generate the 
 * {@code double[] fitX} variable (i.e., {@code fitX} goes from {@code xmin} to {@code xmax} 
 * (inclusive) using {@code nSteps} steps) 
	 * @param xmin - {@code xmin=[xmin1, ..., xminN]}
	 * @param xmax - {@code xmax=[xmax1, ..., xmaxN]} */
	public void setFitX(int nSteps, double[] xmin, double[] xmax) {
		this.nSteps = nSteps;
		xxmin = xmin;
		xxmax = xmax;
		if ( (xmin!=null) && (xmax!=null) ) {
			if (xmin.length!=xmax.length)	LemMING.error(String.format("LevMarBev error :: the length of xmin (%d) and xmax (%d) are not equal", xmin.length, xmax.length));
			fitXX = new double[xmin.length][nSteps];
			fitY = new double[nSteps];
			double delta, mn;
			for (int i=0; i<xmin.length; i++) {
				mn = xmin[i];
				delta = (xmax[i]-mn)/(nSteps-1.0);
				for (int j=0; j<nSteps; j++)
					fitXX[i][j] = j*delta + mn;
			}
		} else {
			fitXX = new double[0][0];
			fitY = new double[0];
		}
	}	


	
	/** Start the fitting routine */
	@Override
	public void run() {
		iter = 0;
		while (!isPrecisionAcheived() && (iter < maxIter) ) {
			marquardt();
			
	        // check relative precision of the fitting parameters
	        for (int i=0; i<nfl; i++) {
	            if (b[ifl[i]] == 0.0) {
	            	currentParamPrecision[i] = Double.MAX_VALUE;
	            	precisionAcheived[i] = false;
	            } else {
	            	currentParamPrecision[i] = Math.abs(1.0-p[ifl[i]]/b[ifl[i]]);
	                if ( (iter > 0) && ( (Math.abs(p[ifl[i]]) < 1.e-10) || (currentParamPrecision[i] < precision) ) ) {
	                	precisionAcheived[i] = true;
	                } else {
	                	precisionAcheived[i] = false;
	                }
	            }
	        }
	        
	        // set the best-fit parameters and evaluate the parameter uncertainties
	        for (int j=0; j<nfl; j++) {
	        	pBest[ifl[j]]=b[ifl[j]];
                dpBest[ifl[j]]=Math.sqrt(array[j][j]/alpha[j][j] * chiSqr); // java converts to Infinity if division by zero occurs
	        }

			// check for NaN or infinities in the fitting parameters
			if ( isNaNInf()) {
				if (verbose) LemMING.warning("LevMarBev warning :: the parameter array contains NaN or infinity");
				return;
			}
			
			iter++;
		}
		
	    // display a warning if the fitting routine exceeded the specified number 
		// of fitting iterations before reaching the specified precision
	    if ((verbose) && (iter == maxIter) && !isPrecisionAcheived()) {
	    	String s = "";
	    	String v = "";
	    	for (int i=0; i<nfl; i++) {
	    		s += String.format("%11s", String.valueOf(ifl[i]));
	    		v += String.format("%11.3e", currentParamPrecision[i]);
	    	}	    	
	    	LemMING.warning(String.format("LevMarBev warning :: Maximum number of fitting iterations reached\nParameter:%s\nPrecision:%s", s, v));
	    }

	    // determine the residuals
	    if (calcRes) getResuiduals();
	    
	    // determine the resultant fit using the users specified range
	    if (is1D){	    	
	    	function.fcn(fitX, pBest, fitY);
	    } else {
	    	function.fcn(fitXX, pBest, fitY);
	    }

	    // do some final value adjustments/checks on the best-fit parameters
	    // (e.g., restricting a cosine phase to be in the 0 to 360 degree range)
	    function.finalCheck(x, y, pBest);
	    
	    // if the final fit parameters are equal to the initial guess then
	    // we must have been caught in a function.pCheck() loop that kept 
	    // re-assigning the initial guess to the best fit. If this is the case
	    // then we should make the values of pBest to be Double.MAX_VALUE
	    // to flag that something went wrong and it is not a good fit.
	    boolean pEqual = true;
	    for (int i=0; i<nTerms; i++) {
	    	if (p[i]!=pBest[i])
	    		pEqual = false;
	    }
	    if (pEqual) {
	    	for (int i=0; i<nTerms; i++) {
	    		pBest[i] = Double.MAX_VALUE;
	    		dpBest[i] = Double.MAX_VALUE;
	    	}
	    }
	}

	/** If NaN or infinity is found in the parameter array then return
	 *  {@code true}, otherwise return {@code false} */
	private boolean isNaNInf() {
		for (double d : p)
			if ( Double.isNaN(d) || Double.isInfinite(d) )
				return true;
		return false;
	}
	
	/** The Levenberg-Marquardt method */
	private void marquardt() {

		// calculate the derivatives
	    function.deriv(x, p, der);

	    // evaluate the alpha and beta "curvature" matrices of chi squared, see p.224 Bevington
	    for (int j=0; j<nfl; j++) {
	    	beta[j]=0.0;
	    	for (int k=0; k<=j; k++) {
	    		alpha[j][k] = 0.0;
	    	}	
	    }	    
	    for (int i=0; i<npts; i++) {
	    	for (int j=0; j<nfl; j++) {
	            beta[j] += weights[i]*(y[i]-currentY[i])*der[ifl[j]][i];
	            for (int k=0; k<=j; k++) {
	                alpha[j][k] += weights[i]*der[ifl[j]][i]*der[ifl[k]][i];
	                alpha[k][j] = alpha[j][k];
	            }
	    	}
	    }
	    
	    boolean changeLambda = true;
	    int cnt = 0; // used to make sure we don't get caught in an infinite loop if the value of chisqr keeps increasing
	    while (changeLambda) {        
	        
	        cnt += 1;
	        if (cnt > 100) {
	            if (verbose) LemMING.warning("LevMarBev warning :: lambda counter to big");
	            for (int i=0; i<nTerms; i++) 
					b[i] = Double.NaN;
	            return;
	        }
	        
	        // invert the modified curvature matrix to find new parameters
	        for (int j=0; j<nfl; j++) {
	            for (int k=0; k<nfl; k++) {
	            	array[j][k] = alpha[j][k]/Math.sqrt(alpha[j][j]*alpha[k][k]);
	            }
	            array[j][j] = 1.0 + lambda;
	        }
	        matinv(array); 
	    
	        // use the new parameters
	        for (int i=0; i<nTerms; i++) 
				b[i] = p[i];
	        for (int j=0; j<nfl; j++) {
	            for (int k=0; k<nfl; k++) {
	            	b[ifl[j]] += beta[k]*array[j][k]/Math.sqrt(alpha[j][j]*alpha[k][k]);
	            }
	        }
	    
	        // If chiSqr increases then increase lambda and try again
	        function.fcn(x, b, currentY);
	        chiSqr = calcRedChiSq(y, currentY, weights); // determine the new chiSq    
    
	        if (chiSqr > chiSqrOld) {
	            lambda *= 10.0; // increase by a factor of 10
	            changeLambda = true;
	        } else {
	            changeLambda = false;
	            lambda *= lambda*0.1; // decrease lambda by a factor of 10, but don't let lambda get too small
	            if (lambda < 1e-7) lambda = 1.0e-7;
	        }
	    
	    }
	    
	    chiSqrOld = chiSqr;
	    
	    // use this routine to force constraints on parameters in case they get too large/small/unrealistic
	    function.pCheck(b, pInitial);
	}
	
	/** Inverts the curvature matrix */
	private void matinv(double[][] array) {
	    det = 1.0;
	    int i=0, j=0, k=0; 
		for (k=0; k<nfl; k++) {    
	        amax = 0.0;
	        goto_21 = true; // this boolean replaces a FORTRAN 'goto' statement in the original code
	        while (goto_21) {
	        	for (i=k; i<nfl; i++) {
	                for (j=k; j<nfl; j++) {
	                    if ( Math.abs(amax)-Math.abs(array[i][j]) <= 0.0 ) {
	                        amax = array[i][j];
	                        ik[k] = i;
	                        jk[k] = j;
	                    }
	                }
	            }
	            if (amax == 0.0) {
	                det = 0.0;
	                return;
	            } else {
	                i = ik[k];
	            }
	            if (i-k < 0) {
	                goto_21 = true;
	            } else if (i-k > 0) {
	                goto_21 = false;
	                for (j=0; j<nfl; j++) {
	                    temp = array[k][j];
	                    array[k][j] = array[i][j];
	                    array[i][j] = -temp;
	                }
	                j = jk[k];
	            } else {
	                goto_21 = false;
	                j = jk[k];
	            }
	            if (j-k < 0) {
	                goto_21 = true;
	            } else if (j-k > 0) {
	                goto_21 = false;
	                for (i=0; i<nfl; i++) {
	                    temp = array[i][k];
	                    array[i][k] = array[i][j];
	                    array[i][j] = -temp;
	                }
	            } else {
	                goto_21 = false;
	            }
	        }
	        for (i=0; i<nfl; i++) {
	            if (i-k != 0) {
	                array[i][k] = -array[i][k]/amax;
	            }
	        }
	        for (i=0; i<nfl; i++) {
	            for (j=0; j<nfl; j++) {
	                if (i-k != 0) {   
	                    if (j-k != 0) {
	                        array[i][j] += array[i][k]*array[k][j];
	                    }
	                }
	            }
	        }
	        for (j=0; j<nfl; j++) {
	            if (j-k != 0) {
	                array[k][j] = array[k][j]/amax;
	            }
	        }
	        array[k][k] = 1.0/amax;
	        det = det*amax;
		}
	    for (int m=0; m<nfl; m++) {
	        k = nfl - m - 1;
	        j = ik[k];
	        if (j-k > 0) {
	            for (i=0; i<nfl; i++) {
	                temp = array[i][k];
	                array[i][k] = -array[i][j];
	                array[i][j] = temp;
	            }
	            i = jk[k];
	        } else {
	            i = jk[k];
	        }
	        if (i-k > 0) {
	            for (j=0; j<nfl; j++) {
	                temp = array[k][j];
	                array[k][j] = -array[i][j];
	                array[i][j] = temp;
	            }
	        }
	    }
	}
	
	/** Returns the logical AND of the values in the precisionAcheived */
	private boolean isPrecisionAcheived() {
		for (boolean b : precisionAcheived)
			if (!b)
				return false;
		return true;
	}
	
	/** Initialize the fitting algorithm: do parameter checks and create the necessary arrays */
	private void initialize() {

		if (is1D) {
			if (x==null) {
				LemMING.error("LevMarBev error :: the x-array is null");
			} else if (x.length == 0) {
				LemMING.error("LevMarBev error :: the length of the x-array is zero");
			} else if (x.length != y.length) {
				LemMING.error(String.format("LevMarBev error :: the length of x (%d) and y (%d) are not equal", x.length, y.length));
			}
		} else {
			if (xx==null) {
				LemMING.error("LevMarBev error :: the x-array is null");
			} else if (xx.length == 0) {
				LemMING.error("LevMarBev error :: the length of the x-array is zero");
			} else if (xx.length != y.length) {
				LemMING.error(String.format("LevMarBev error :: the length of x (%d) and y (%d) are not equal", xx.length, y.length));
			}
		}

		if (y==null) {
			LemMING.error("LevMarBev error :: the y-array is null");
		} else if (y.length == 0) {
			LemMING.error("LevMarBev error :: the length of the y-array is zero");
		} else {
			npts = y.length;
		}
		
		if (p==null) {
			LemMING.error("LevMarBev error :: the parameter-array is null");
		} else {
			nTerms = p.length;
			if (nTerms == 0) {
				LemMING.error(String.format("LevMarBev error :: the length of the parameter-array is zero"));
			} else {
				dpBest = new double[nTerms];
				pBest = new double[nTerms];
				for (int i=0; i<nTerms; i++)
					pBest[i] = p[i];
			}
		}
		
		der = new double[nTerms][npts];
		
		if (maxIter < 1) this.maxIter=1;
		
		b = new double[nTerms];

		pInitial = new double[nTerms];
		for (int i=0; i<nTerms; i++)
			pInitial[i] = p[i];
		
		setPfloat(this.pFloat);
		setDy(this.dy);		
		setCalcRes(calcRes);

		if (is1D) {
			setFitX(nSteps, xmin, xmax);
		} else {
			setFitX(nSteps, xxmin, xxmax);
		}
		
		precisionAcheived = new boolean[nfl];
		currentParamPrecision = new double[nfl];
		for (int i=0; i<nfl; i++) {
			precisionAcheived[i] = false;
			currentParamPrecision[i] = Double.MAX_VALUE;
		}

	    currentY = new double[npts];
	    function.fcn(x, p, currentY);
   		chiSqrOld = calcRedChiSq(y, currentY, weights);   		
	}

	/** Calculate the reduced chi-squared value */
	private double calcRedChiSq(double[] y, double[] currentY, double[] weights) {
		double chisq = 0.0;
		double v1, v2;
		for (int i=0; i<npts; i++) {
			v1 = y[i];
			v2 = currentY[i];
			chisq += (v1 - v2) * (v1 - v2) * weights[i];
		}
		return chisq/nFree;
	}


}
