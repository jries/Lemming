package org.lemming.math;

import ij.gui.Roi;
import ij.process.ImageProcessor;

import org.apache.commons.math3.exception.ConvergenceException;
import org.apache.commons.math3.exception.TooManyEvaluationsException;
import org.apache.commons.math3.fitting.leastsquares.LeastSquaresBuilder;
import org.apache.commons.math3.fitting.leastsquares.LeastSquaresOptimizer.Optimum;
import org.apache.commons.math3.fitting.leastsquares.LevenbergMarquardtOptimizer;
import org.apache.commons.math3.optim.ConvergenceChecker;
import org.apache.commons.math3.optim.PointVectorValuePair;
import org.apache.commons.math3.util.Precision;

public class GaussianFitterAlternative implements FitterInterface {

	int[] xgrid, ygrid;									// x and y positions of the pixels	
	double[] Ival;										// intensity value of the pixels
	private ImageProcessor ip;
	private Roi roi;
	private int maxIter;
	private int maxEval;
	public static int PARAM_2D_LENGTH = 6;	
	
	public GaussianFitterAlternative(ImageProcessor ip_, Roi roi_, int maxIter_, int maxEval_) {
		ip = ip_;
		roi = roi_;
		maxIter = maxIter_;
		maxEval = maxEval_;
	}
	
	private static LeastSquaresBuilder builder(EllipticalGaussian problem){
    	LeastSquaresBuilder builder = new LeastSquaresBuilder();
    	 builder.model(problem.getModelFunction(), problem.getModelFunctionJacobian());
		return builder;
    }
	
	private static LevenbergMarquardtOptimizer getOptimizer() {
		// Different convergence thresholds seem to have no effect on the resulting fit, only the number of
		// iterations for convergence
		final double initialStepBoundFactor = 100;
		final double costRelativeTolerance = 1e-10;
		final double parRelativeTolerance = 1e-10;
		final double orthoTolerance = 1e-10;
		final double threshold = Precision.SAFE_MIN;
        return new LevenbergMarquardtOptimizer(initialStepBoundFactor,
				costRelativeTolerance, parRelativeTolerance, orthoTolerance, threshold);
	}
	
	private void createGrids(){
		int rwidth = (int) roi.getFloatWidth();
		int rheight = (int) roi.getFloatHeight();
		int xstart = (int) roi.getXBase();
		int ystart = (int) roi.getYBase();

		xgrid = new int[rwidth*rheight];
		ygrid = new int[rwidth*rheight];
		Ival = new double[rwidth*rheight];
		for(int i=0;i<rheight;i++){
			for(int j=0;j<rwidth;j++){
				ygrid[i*rwidth+j] = i+ystart;
				xgrid[i*rwidth+j] = j+xstart;
				Ival[i*rwidth+j] = ip.get(j+xstart,i+ystart);
			}
		}
	}

	@Override
	public double[] fit() {
		createGrids();
		EllipticalGaussian eg = new EllipticalGaussian(xgrid, ygrid);
		double [] result = new double[4];
		LevenbergMarquardtOptimizer optimizer = getOptimizer();
		double[] fittedEG;
		try {
			final Optimum optimum = optimizer.optimize(
	                builder(eg)
	                .target(Ival)
	                .checkerPair(new ConvChecker2DGauss())
	                .start(eg.getInitialGuess(ip,roi))
	                .maxIterations(maxIter)
	                .maxEvaluations(maxEval)
	                .build()
	        );
			fittedEG = optimum.getPoint().toArray();
		} catch(TooManyEvaluationsException | ConvergenceException e){
			//System.out.println("Too many evaluations" + e.getMessage());
        	return null;
		}
        
		//check bounds
		if (!roi.contains((int)Math.round(fittedEG[0]), (int)Math.round(fittedEG[1])))
			return null;
		
        result[0] = fittedEG[0];
        result[1] = fittedEG[1];
        result[2] = Math.abs(fittedEG[2]);
        result[3] = Math.abs(fittedEG[3]);
        	
		return result;
	}
	
	private class ConvChecker2DGauss implements ConvergenceChecker<PointVectorValuePair> {
	    
		int iteration_ = 0;
	    boolean lastResult_ = false;

		public static final int INDEX_X0 = 0;
		public static final int INDEX_Y0 = 1;
		public static final int INDEX_SX = 2;
		public static final int INDEX_SY = 3;
		public static final int INDEX_I0 = 4;
		public static final int INDEX_Bg = 5;
		
		@Override
		public boolean converged(int i, PointVectorValuePair previous, PointVectorValuePair current) {
	         if (i == iteration_)
	             return lastResult_;
	          
	          iteration_ = i;
	          double[] p = previous.getPoint();
	          double[] c = current.getPoint();
	          
	          if ( Math.abs(p[INDEX_I0] - c[INDEX_I0]) < 5  &&
	                  Math.abs(p[INDEX_Bg] - c[INDEX_Bg]) < 1 &&
	                  Math.abs(p[INDEX_X0] - c[INDEX_X0]) < 0.02 &&
	                  Math.abs(p[INDEX_Y0] - c[INDEX_Y0]) < 0.02 &&
	                  Math.abs(p[INDEX_SX] - c[INDEX_SX]) < 2 &&
	                  Math.abs(p[INDEX_SY] - c[INDEX_SY]) < 2 ) {
	             lastResult_ = true;
	             return true;
	          }
	        lastResult_ = false;
			return false;
		}
	}

}
