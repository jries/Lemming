package org.lemming.math;

import java.util.concurrent.atomic.AtomicInteger;

import org.apache.commons.math3.exception.TooManyEvaluationsException;
import org.apache.commons.math3.exception.TooManyIterationsException;
import org.apache.commons.math3.fitting.leastsquares.LeastSquaresBuilder;
import org.apache.commons.math3.fitting.leastsquares.LevenbergMarquardtOptimizer;
import org.apache.commons.math3.fitting.leastsquares.LeastSquaresOptimizer.Optimum;
import org.apache.commons.math3.util.Precision;

import ij.ImagePlus;
import ij.ImageStack;
import ij.gui.Roi;
import ij.process.ImageProcessor;
import ij.util.ThreadUtil;

/**
* This class handle the calibration by calling the various method of the fitter. Each fit runs on a dedicated thread to allow updating of the GUI.
* 
*/

public class Calibrator {

	/////////////////////////////
	// Fitting parameters
	public static int PARAM_1D_LENGTH = 9;				// Number of parameters to fit in 1D (calibration curve)
	public static int PARAM_2D_LENGTH = 6;				// Number of parameters to fit in 2D (elliptical Gaussian)
	int MAX_ITERATIONS_1D = 100;
	int MAX_ITERATIONS_2D = 100;
	int MAX_EVALUATIONS_2D = 100;
	
	/////////////////////////////
	// Results
	private double[] zgrid;									// z positions of the slices in the stack
	private volatile double[] Wx, Wy; 
	private double[] Calibcurve, paramWx;					// 1D and 2D fit results
	private double[] curveWx, curveWy;						// quadratically fitted curves
	/////////////////////////////
    // Parameters from ImageStack
	private int nSlice;

	/////////////////////////////
	// Input from user
    private int zstep;
    private int rangeStart, rangeEnd;					// Both ends of the restricted z range and length of the restriction
    private volatile Roi roi;
    
	private ImageStack is;
	private Calibration cal;
	
   
	public Calibrator(ImagePlus im, int zstep, Roi r){
		this.is = im.getStack();
		this.zstep = zstep;
    	this.nSlice = im.getNSlices(); 
    	im.getWidth(); 
    	im.getHeight();
    	this.roi = r;
	
    	// Initialize arrays
    	zgrid = new double[nSlice];						// z position of the frames
    	Wx = new double[nSlice];						// width in x of the PSF
    	Wy = new double[nSlice];						// width in y of the PSF
    	Calibcurve = new double[nSlice];
    	curveWx = new double[nSlice];					// value of the calibration on X
    	curveWy = new double[nSlice];					// value of the calibration on Y
    	paramWx = new double[PARAM_1D_LENGTH];			// parameters of the calibration on X
    	cal = new Calibration(zgrid, Wx, Wy, curveWx, curveWy, Calibcurve, paramWx);
	}
	
	
	// ////////////////////////////////////////////////////////////
	// 1D and 2D fits
	public void fitStack() {

		Thread[] threads = ThreadUtil.createThreadArray(Runtime.getRuntime().availableProcessors()-1);
		final AtomicInteger ai = new AtomicInteger(0);

		for (int ithread = 0; ithread < threads.length; ithread++) {

			threads[ithread] = new Thread("fit_" + ithread) {
				@Override
				public void run() {
					for (int i = ai.getAndIncrement(); i < nSlice; i = ai.getAndIncrement()) {
						ImageProcessor ip = is.getProcessor(i + 1);
						Gaussian2DFitter gf = new Gaussian2DFitter(ip, roi, 200, 200);
						double[] results = gf.fit();
						if (results!=null){
							Wx[i]=results[2];
							Wy[i]=results[3];
						}
					}
				}
			};
		}
		ThreadUtil.startAndJoin(threads);

		createZgrid(zgrid, 0);
		fixCurve(Wx);
		fixCurve(Wy);
		// Save results in calibration
		cal.setZgrid(zgrid);
		cal.setWx(Wx);
		cal.setWy(Wy);

		// Display result
		cal.plot(Wx, Wy, "2D gaussian LSQ");
	}	
	
	private static void fixCurve(double[] d) {
		for (int i=1 ; i<d.length-1;i++)
			if (d[i]<0.1) d[i]=(d[i-1]+d[i+1])/2;
	}


	public void fitCalibrationCurve(final int rStart, final int rEnd){	
       Thread t = new Thread(new Runnable() {

			@Override
            public void run() {
				double[] param = new double[PARAM_1D_LENGTH];
				calculateRange(rStart, rEnd);
		    	
				try{
					fitCurves(zgrid, Wx, Wy, param, curveWx, curveWy, rangeStart, rangeEnd, 100, 100);
		    	} catch (TooManyEvaluationsException e) {
		    		System.err.println("Too many evaluations!");				
		    	}  catch (TooManyIterationsException e) {
		    		System.err.println("Too many iterations");
		    	}
				
				// sx2-sy2
				for(int i=0;i<nSlice;i++)
					Calibcurve[i] = curveWx[i]*curveWx[i]-curveWy[i]*curveWy[i]; 

				// Save in calibration
				cal.setcurveWx(curveWx);
				cal.setcurveWy(curveWy);
				cal.setCalibcurve(Calibcurve);
				cal.setparam(param);
				
				// Display result
				cal.plotWxWyFitCurves();
            }
        });
	    t.start();
	    try {
			t.join();
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
	}
	
	//////////////////////////////////////////////////////////////
	// Save
	public void saveCalib(String path){
		cal.saveAsCSV(path);
	}
	
	//////////////////////////////////////////////////////////////
	// Misc functions
	public Calibration getCalibration(){
		return cal;												
	}
	
	private void createZgrid(double[] z, int offset){
		for(int i=0;i<z.length;i++)
			z[i] = (i-offset)*zstep;
	}
	
	private void calculateRange(final int rStart, final int rEnd){
		double minStart = Math.abs(rStart-zgrid[0]);
		double minEnd = Math.abs(rEnd-zgrid[nSlice-1]);
		int iStart = 0;
		int iEnd = nSlice-1;
		for(int i=1;i<nSlice;i++){
			if(Math.abs(rStart-zgrid[i])<minStart){
				minStart = Math.abs(rStart-zgrid[i]);
				iStart = i;
			}
			if(Math.abs(rEnd-zgrid[nSlice-1-i])<minEnd){
				minEnd = Math.abs(rEnd-zgrid[nSlice-1-i]);
				iEnd = nSlice-1-i;
			}
		}
		this.rangeStart = iStart;
		this.rangeEnd = iEnd;
	}
	
	private static LeastSquaresBuilder builder(CalibrationCurve problem){
    	LeastSquaresBuilder builder = new LeastSquaresBuilder();
    	builder.model(problem.getModelFunction(), problem.getModelFunctionJacobian());
		return builder;
    }
	
	private static LevenbergMarquardtOptimizer getOptimizer() {
		final double initialStepBoundFactor = 100;
		final double costRelativeTolerance = 1e-10;
		final double parRelativeTolerance = 1e-10;
		final double orthoTolerance = 1e-10;
		final double threshold = Precision.SAFE_MIN;
        return new LevenbergMarquardtOptimizer(initialStepBoundFactor,
				costRelativeTolerance, parRelativeTolerance, orthoTolerance, threshold);
	}
	
	private static void fitCurves(double[] z, double[] wx, double[] wy, double[] param, double[] curvex, double[] curvey,
			int rStart, int rEnd, int maxIter, int maxEval) {

	  	double[] rangedZ = new double[rEnd-rStart+1];
    	double[] rangedWx = new double[rEnd-rStart+1];
    	double[] rangedWy = new double[rEnd-rStart+1];
    	
    	System.arraycopy(z, rStart, rangedZ, 0, rEnd-rStart+1);
    	System.arraycopy(wx, rStart, rangedWx, 0, rEnd-rStart+1);
    	System.arraycopy(wy, rStart, rangedWy, 0, rEnd-rStart+1); 
    	
        final CalibrationCurve problem = new CalibrationCurve(rangedZ,rangedWx, rangedWy);
        
        LevenbergMarquardtOptimizer optimizer = getOptimizer();

        final Optimum optimum = optimizer.optimize(
                builder(problem)
                        .target(problem.getTarget())
                        .start(problem.getInitialGuess())
                        .maxIterations(maxIter)
                        .maxEvaluations(maxEval)
                        .build()
        );
        
    	// Copy the fitted parameters
        double[] result = optimum.getPoint().toArray();

        for(int i=0; i<PARAM_1D_LENGTH;i++){
        	param[i] = result[i];
        }
        
        // Copy the fitted curve values																				
        double[] values = CalibrationCurve.valuesWith(z, result);

        for(int i=0; i<curvex.length;i++){
        	curvex[i] = values[i];
        	curvey[i] = values[i+curvex.length];
        }
	}
	
}













