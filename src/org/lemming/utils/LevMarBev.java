package org.lemming.utils;

import org.lemming.data.GenericLocalization;
import org.lemming.processors.SISO;

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
 * @param {@link org.lemming.utils.FitFunction FitFunction} <b>function</b> - the fitting
 * function to use</br></br>
 * @param double[] <b>x</b> (for 1D) or double[][] <b>x</b> (for nD) - the x values
 * (the independent variable). For 1D, the array has length M. For n>1 dimension, {@code x}
 * is a N by M array, where N is the number of dimensions, M is the number of data points</br></br>
 * @param double[] <b>y</b> - array of length M^N specifying the y values (the 
 * dependent variable)</br></br>
 * @param double[] <b>p</b> - the initial guess</br></br>
 * @param byte[] <b>pFloat</b> - specifies which parameters in {@code p} are 
 * allowed to vary (float) or are fixed during the fitting process, 0=fixed
 * or 1=float. For example, if there are 4 parameters in the {@code function} 
 * and {@code pFloat=[1,1,0,1]} then parameters 0, 1 and 3 are allowed to vary (float) 
 * while parameter 2 is fixed at the value specified in {@code p[2]}</br></br>
 * @param double[] <b>dy</b> - array of length M^N specifying the uncertainty 
 * for each {@code y} value, the {@code dy} values are used as the weights (1/dy)^2</br></br>
 * @param int <b>nSteps</b> - the number of evenly-spaced numbers to use to generate the 
 * {@code double[] fitX} variable (i.e., for 1D data, {@code fitX} goes from 
 * {@code xmin} to {@code xmax} (inclusive) using {@code nSteps} steps)</br></br>
 * @param double <b>xmin, xmax</b> (for 1D) or double[] <b>xmin, xmax</b> (for nD) -
 * if these values are specified then the {@code function} is evaluated using 
 * the best-fit parameters for each value in {@code fitX} and the function-evaluated values
 * are stored within the {@code double[] fitY} variable. For nD data, the values would be
 * {@code xmin=[xmin1, ..., xminN]} and {@code xmax=[xmax1, ..., xmaxN]} </br></br>
 * @param int <b>maxIter</b> - the maximum number of fitting iterations that are allowed</br></br>
 * @param double <b>precision</b> - keep iterating the fit until this relative precision 
 * is reached for every parameter that is allowed to vary (i.e., the parameters where 
 * {@code pFloat[i]=1}). If the iteration number becomes greater than {@code maxIter}
 * than the fitting routine will also stop.</br></br>
 * @param boolean <b>calcRes</b> - specify whether to calculate the (@code residuals} 
 * ({@code y - currentY})</br></br>
 * @param boolean <b>verbose</b> - specify whether to display a warning message
 * (if there are any). For example, if the {@code precision} condition was not met after 
 * {@code maxIter} iterations then the fit did not converge to a <i>good-enough</i>
 * result for you.</br></br>
 * 
 * <p>This class also produces the following variables</p>
 * <ul>
 * <li> double[] <b>pBest</b> - the parameters that best fit the data which minimize
 * the chi square</li></br>
 * <li> double[] <b>dpBest</b> - the uncertainty for each {@code pBest}, if 
 * the parameter was fixed in the fit then its uncertainty will be 0.0</li></br>
 * <li> double[][] <b>der</b> - a {@code p.length} by M^N array 
 * containing the values of the partial derivatives of the function</li></br>
 * <li> double[] <b>currentY</b> - array of length M^N of the function evaluated 
 * at each {@code x} value using the latest best-fit parameters, {@code pBest}</li></br> 
 * </ul>
 * 
 * <p>This class may also produce the following variables if requested for in the constructor</p>
 * <ul>
 * <li> double[] <b>fitX</b> (for 1D) or double[][] <b>fitXX</b> (for nD) - 
 * for 1D data -> values from {@code xmin} to {@code xmax} (inclusive) using {@code nStep} steps<br> 
 * for nD data -> values from {@code xmin=[xmin1, ..., xminN]} to {@code xmax=[xmax1, ..., xmaxN]} 
 * (inclusive) using {@code nStep} steps</li></br>
 * <li> double[] <b>fitY</b> - for each value in {@code fitX} the corresponding 
 * function-evaluated values are calculated using the values from {@code pBest}</li></br>
 * <li> double[] <b>residuals</b> - the values of the residuals 
 * ({@code y - currentY})</li></br> 
 * </ul>
 * 
 * @author jborbely
 */
public class LevMarBev extends SISO<GenericLocalization,GenericLocalization> {//implements Runnable {

	boolean is1D; // is true if x is a double[], is false if x is a double[][]
	boolean goto_21; // this boolean value is used to replace a FORTRAN 'goto' statement in the original code
	boolean[] precisionAcheived; // determines if the requested precision was achieved for each fit parameter
	int iter; // the current iteration number
	int npts; // the number of data points
	int nTerms; // the number of fit parameters in the function
	int nfl; // the number of floating parameters in the fit
	int[] ifl, ik, jk; // used for specifying the parameters indices that are allowed to vary (are floating) (used in the matrix inversion method)
	double nFree; // the number of free parameters, i.e., npts - nfl (it's of type double because it used to calculate the reduced chisqr)
	double det; // the determinant of the fit
	double chiSqr; // the reduced chi-square of the latest fitting iteration
	double chiSqrOld; // the reduced chi-square of the previous fitting iteration
	double amax; // the
	double temp; // holds temporary values during the matrix inversion
	double lambda = 0.001; // the damping parameter
	double[] weights; // the weights, i.e., 1/dy^2
	double[] currentParamPrecision; // the precision of each fitting parameter is calculated in each fitting iteration
	double[] currentY; // the evaluation of the function for the current parameter values
	double[] beta; // beta is the "curvature" matrix of chi squared, see p.224 of Bevington
	double[] pNew; // holds the new parameter values
	double[][] array; // the inverted modified curvature matrix
	double[][] alpha; // alpha is the "curvature" matrix of chi squared, see p.224 of Bevington

	// these variables are defined in the documentation above
	FitFunction function;
	boolean calcRes, verbose;
	int nSteps, maxIter;
	double precision, xmin, xmax;
	byte[] pFloat;
	double[] x, y, dy, p, pBest, dpBest, pInitial, xxmin, xxmax, residuals, fitY, fitX; 
	double[][] xx, fitXX, der;	
	
	/** see {@link org.lemming.utils.LevMarBev LevMarBev} */
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

	/** see {@link org.lemming.utils.LevMarBev LevMarBev} */
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

	/** see {@link org.lemming.utils.LevMarBev LevMarBev} */
	public LevMarBev(FitFunction function, double[][] x, double[] y, double[] p) {
		this(function, x, y, p, new byte[0], new double[0], 100, 1.0e-6, new double[0], new double[0], 1, false, false);		
	}

	/** see {@link org.lemming.utils.LevMarBev LevMarBev} */
	public LevMarBev(FitFunction function, double[][] x, double[] y, double[] p, byte[] pFloat) {
		this(function, x, y, p, pFloat, new double[0], 100, 1.0e-6, new double[0], new double[0], 1, false, false);		
	}

	/** see {@link org.lemming.utils.LevMarBev LevMarBev} */
	public LevMarBev(FitFunction function, double[][] x, double[] y, double[] p, byte[] pFloat, double[] dy) {
		this(function, x, y, p, pFloat, dy, 100, 1.0e-6, new double[0], new double[0], 1, false, false);		
	}

	/** see {@link org.lemming.utils.LevMarBev LevMarBev} */
	public LevMarBev(FitFunction function, double[][] x, double[] y, double[] p, byte[] pFloat, double[] dy, int maxIter, double precision) {
		this(function, x, y, p, pFloat, dy, maxIter, precision, new double[0], new double[0], 1, false, false);		
	}

	/** see {@link org.lemming.utils.LevMarBev LevMarBev} */
	public LevMarBev(FitFunction function, double[] x, double[] y, double[] p) {
		this(function, x, y, p, new byte[0], new double[0], 100, 1.0e-6, 0.0, 0.0, 1, false, false);		
	}

	/** see {@link org.lemming.utils.LevMarBev LevMarBev} */
	public LevMarBev(FitFunction function, double[] x, double[] y, double[] p, byte[] pFloat) {
		this(function, x, y, p, pFloat, new double[0], 100, 1.0e-6, 0.0, 0.0, 1, false, false);		
	}

	/** see {@link org.lemming.utils.LevMarBev LevMarBev} */
	public LevMarBev(FitFunction function, double[] x, double[] y, double[] p, byte[] pFloat, double[] dy) {
		this(function, x, y, p, pFloat, dy, 100, 1.0e-6, 0.0, 0.0, 1, false, false);		
	}

	/** see {@link org.lemming.utils.LevMarBev LevMarBev} */
	public LevMarBev(FitFunction function, double[] x, double[] y, double[] p, byte[] pFloat, double[] dy, int maxIter, double precision) {
		this(function, x, y, p, pFloat, dy, maxIter, precision, 0.0, 0.0, 1, false, false);		
	}

	/** Calculate the reduced chi-squared value */
	private double calcRedChiSq(double[] y, double[] currentY, double[] weights) {
		double chisq = 0.0, v1, v2;
		for (int i=0; i<npts; i++) {
			v1 = y[i];
			v2 = currentY[i];
			chisq += (v1 - v2) * (v1 - v2) * weights[i];
		}
		return chisq/nFree;
	}

	/** Returns the best fit parameters */
	public double[] getBestParameters(){
		return pBest;
	}

	/** Returns the uncertainty of the best fit parameters */
	public double[] getBestParametersUncertainty(){
		return dpBest;
	}

	/** Returns the user specified x range from {@code xmin} to {@code xmax} 
	 * (inclusive) using {@code nStep} steps */
	public double[] getFitX(){
		return fitX;
	}

	/** Returns the user specified x range from values from values from 
	 * {@code xmin=[xmin1, ..., xminN]} to {@code xmax=[xmax1, ..., xmaxN]}
	 * (inclusive) using {@code nStep} steps 
	 */
	public double[][] getFitXX(){
		return fitXX;
	}

	/** Returns the function evaluated for each value in {@code fitX} using
	 * the parameters of best fit */
	public double[] getFitY(){
		return fitY;
	}

	/** Returns the number of fitting iterations */
	public int getNumberOfFittingIterations(){
		return iter;
	}

	/** Returns the value of the reduced chi square */
	public double getReducedChiSquare(){
		return chiSqr;
	}

	/** Returns the residuals (y - currentY) */
	public double[] getResuiduals(){
		for (int i=0; i<npts; i++)
			residuals[i] = (y[i]-currentY[i]);
		return residuals;
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
			} else if (Math.pow(xx[0].length,xx.length) != y.length) {
				LemMING.error(String.format("LevMarBev error :: the length of x (%d) and y (%d) are not equal", Math.pow(xx[0].length,xx.length), y.length));
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
				pBest = new double[nTerms];
				dpBest = new double[nTerms];
				for (int i=0; i<nTerms; i++)
					pBest[i] = p[i];
			}
		}
		
		der = new double[nTerms][npts];
		
		if (maxIter < 1) this.maxIter=1;
		
		pNew = new double[nTerms];

		pInitial = new double[nTerms];
		for (int i=0; i<nTerms; i++)
			pInitial[i] = p[i];
		
		setPfloat(this.pFloat);
		setDy(this.dy);		
		setCalculateResiduals(calcRes);

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
	    if (is1D)
	    	function.fcn(x, p, currentY);
	    else
	    	function.fcn(xx, p, currentY);
   		chiSqrOld = calcRedChiSq(y, currentY, weights);   		
	}

	/** If NaN or infinity is found in the parameter array then return
	 *  {@code true}, otherwise return {@code false} */
	private boolean isNaNInf() {
		for (double d : pBest) {
			if ( Double.isNaN(d) || Double.isInfinite(d) )
				return true;
		}
		return false;
	}
	
	/** Returns the logical AND of the values in the precisionAcheived */
	private boolean isPrecisionAcheived() {
		for (int i=0; i<nfl; i++) {
			if (!precisionAcheived[i])
				return false;
		}
		return true;
	}
	
	/** The Levenberg-Marquardt method */
	private void marquardt() {

		// calculate the derivatives
		if (is1D)
			function.deriv(x, pBest, der);
		else
			function.deriv(xx, pBest, der);
	    
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
					pNew[i] = Double.NaN;
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
				pNew[i] = pBest[i];
	        for (int j=0; j<nfl; j++) {
	            for (int k=0; k<nfl; k++) {
	            	pNew[ifl[j]] += beta[k]*array[j][k]/Math.sqrt(alpha[j][j]*alpha[k][k]);
	            }
	        }
	    
	        // If chiSqr increases then increase lambda and try again
	        if (is1D)
	        	function.fcn(x, pNew, currentY);
	        else
	        	function.fcn(xx, pNew, currentY);
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
	    function.pCheck(pNew, pInitial);
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
	
	/** Start the fitting routine */
	@Override
	public void run() {
		iter = 1;
		while (!isPrecisionAcheived() && (iter < maxIter) ) {
			marquardt();
			
	        // check the relative precision of the fitting parameters
	        for (int i=0, j; i<nfl; i++) {
	        	j = ifl[i];
            	currentParamPrecision[j] = Math.abs(1.0-pBest[j]/pNew[j]);
            	precisionAcheived[j] = ( (iter > 1) && ( Math.abs(pBest[j]) < 1.e-10 || currentParamPrecision[j] < precision ) );
	        }
	        
	        // set the best-fit parameters and evaluate the parameter uncertainties
	        for (int i=0; i<nfl; i++) {
	        	pBest[ifl[i]]=pNew[ifl[i]];
                dpBest[ifl[i]]=Math.sqrt(array[i][i]/alpha[i][i] * chiSqr); // java converts to Infinity if division by zero occurs
	        }

			// check for NaN or infinities in any of the fitting parameters
			if ( isNaNInf()) {
				if (verbose) LemMING.warning("LevMarBev warning :: the parameter array contains NaN or infinity");
		    	for (int i=0; i<nTerms; i++) {
		    		pBest[i] = Double.MAX_VALUE;
		    		dpBest[i] = Double.MAX_VALUE;
		    	}
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
	    		v += String.format("%11.3e", currentParamPrecision[ifl[i]]);
	    	}	    	
	    	LemMING.warning(String.format("LevMarBev warning :: Maximum number of fitting iterations reached\nParameter:%s\nPrecision:%s", s, v));
	    }

	    // determine the residuals
	    if (calcRes) getResuiduals();
	    
	    // determine the resultant fit using the users specified range
	    if (is1D){
	    	if (fitX.length > 0)
	    		function.fcn(fitX, pBest, fitY);
	    } else {
	    	if (fitXX.length > 0)
	    		function.fcn(fitXX, pBest, fitY);
	    }

	    // do some final value adjustments/checks on the best-fit parameters
	    // (e.g., restricting a cosine phase to be in the 0 to 360 degree range)
	    if (is1D)
	    	function.finalCheck(x, y, pBest);
	    else
	    	function.finalCheck(xx, y, pBest);
	    
	    // if the final best-fit parameters are equal to the initial guess then
	    // we must have been caught in a function.pCheck() loop that kept 
	    // re-assigning the initial guess to the best fit. If this is the case
	    // then we should make the values of pBest to be Double.MAX_VALUE
	    // to flag that something went wrong and it is not a good fit.
	    boolean pEqual = true;
	    int numFloated = 0;
	    for (int i=0; i<nTerms; i++) {
	    	if (pInitial[i]!=pBest[i])
	    		pEqual = false;
	    	if (pFloat[i] != 0)
	    		numFloated += 1;
	    }
	    if (pEqual && numFloated > 1) {
	    	for (int i=0; i<nTerms; i++) {
	    		pBest[i] = Double.MAX_VALUE;
	    		dpBest[i] = Double.MAX_VALUE;
	    	}
	    }
	}
	
	/** specify whether to calculate the residuals */
	public void setCalculateResiduals(boolean calcRes) {
		this.calcRes=calcRes;
		if (calcRes) {
			residuals = new double[npts];
		} else {
			residuals = new double[0];
		}
	}

	/** set the values of {@code dy} array */
	public void setDy(double[] dy) {
		if (dy==null) {
			LemMING.error("LevMarBev error :: the dy-array is null");
		} else if (dy.length == 0) {
			this.dy = new double[npts];
			for (int i=0; i<npts ; i++)
				// pick a y-uncertainty that would stand out as being strange so that you know it was manually entered.
				// using a value of 1.0 or 9999.9999 does not change the final uncertainty of each fit parameter 
				// it only makes the reduced chisq value extremely small, implying that the error bars are way too big
				this.dy[i] = 9999.9999;
		} else if (npts != dy.length) {
			LemMING.error(String.format("LevMarBev error :: the length of y (%d) and dy (%d) are not equal", npts, dy.length));
		} else {
			this.dy=dy;
		}
		weights = new double[npts];
		double val;
		for (int i=0; i<npts; i++) {
			val = this.dy[i];
			weights[i] = 1.0/(val*val);
		}
	}

	/** set the values of {@code fitX} to the values from {@code xmin} 
	 * to {@code xmax} (inclusive) using {@code nStep} steps */
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
	 * (inclusive) using {@code nStep} steps */
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

	/** set the maximum number of fitting iterations */
	public void setMaxIter(int maxIter) {
		this.maxIter=maxIter;
	}

	/** set whether each parameter should be fixed (0) or allowed to float (1) during the fit */
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
			ik = new int[nfl];
			jk = new int[nfl];
			beta = new double[nfl];
			alpha = new double[nfl][nfl];
			array = new double[nfl][nfl];
		}
		nFree = (double)(npts - nfl);
	    if (nFree <= 0.0) LemMING.error("LevMarBev error :: the number of free parameters is <= 0 (i.e., there are more floating fitting variables than data points)");
	}

	/** set the precision to achieve for each parameter that is floating */
	public void setPrecision(double precision) {
		this.precision=precision;
	}

	/** specify whether to display warning messages, if any occur */
	public void setVerbose(boolean verbose) {
		this.verbose=verbose;
	}

	@Override
	public void process(GenericLocalization l) {
		float[] w = (float[])l.get("window");
		double[] guess = {LArrays.min(w), l.getX(), l.getY(), LArrays.sum(w)-LArrays.min(w)*w.length, 0.0, 1.0, 1.0};
		System.out.println(guess[0]);
	}
	
}
