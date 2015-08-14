package org.lemming.math;

import org.apache.commons.math3.exception.ConvergenceException;
import org.apache.commons.math3.exception.TooManyEvaluationsException;
import org.apache.commons.math3.fitting.leastsquares.LeastSquaresBuilder;
import org.apache.commons.math3.fitting.leastsquares.LevenbergMarquardtOptimizer;
import org.apache.commons.math3.fitting.leastsquares.ParameterValidator;
import org.apache.commons.math3.fitting.leastsquares.LeastSquaresOptimizer.Optimum;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.optim.ConvergenceChecker;
import org.apache.commons.math3.optim.PointVectorValuePair;
import org.apache.commons.math3.util.Precision;

import ij.gui.Roi;
import ij.process.ImageProcessor;

public class GaussianFitterZ implements FitterInterface {
	
	private ImageProcessor ip;
	private Roi roi;
	private int maxIter;
	private int maxEval;
	private int[] xgrid;
	private int[] ygrid;
	private double[] Ival;
	private double[] params;

	public GaussianFitterZ(ImageProcessor ip_, Roi roi_, int maxIter_, int maxEval_, double[] params_) {
		ip = ip_;
		roi = roi_;
		maxIter = maxIter_;
		maxEval = maxEval_;
		params = params_;
	}
	
	private static LeastSquaresBuilder builder(EllipticalGaussianZ problem){
    	LeastSquaresBuilder builder = new LeastSquaresBuilder();
    	 builder.model(problem.getModelFunction(), problem.getModelFunctionJacobian());
		return builder;
    }
	
	private static LevenbergMarquardtOptimizer getOptimizer() {
		// Different convergence thresholds seem to have no effect on the resulting fit, only the number of
		// iterations for convergence
		final double initialStepBoundFactor = 100;
		final double costRelativeTolerance = 1e-9;
		final double parRelativeTolerance = 1e-9;
		final double orthoTolerance = 1e-9;
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
		
		//double max = Double.NEGATIVE_INFINITY;
		for(int i=0;i<rheight;i++){
			for(int j=0;j<rwidth;j++){
				ygrid[i*rwidth+j] = i+ystart;
				xgrid[i*rwidth+j] = j+xstart;
				Ival[i*rwidth+j] = ip.get(j+xstart,i+ystart);
				//max = Math.max(max, Ival[i*rwidth+j]);
			}
		}
		//for (int l=0; l<Ival.length;l++)
		//	Ival[l] /= max;
	}

	@Override
	public double[] fit() {
		createGrids();
		EllipticalGaussianZ eg = new EllipticalGaussianZ(xgrid, ygrid, params);
		LevenbergMarquardtOptimizer optimizer = getOptimizer();
		double[] fittedEG;
		try {
			final Optimum optimum = optimizer.optimize(
	                builder(eg)
	                .target(Ival)
	                .checkerPair(new ConvChecker3DGauss())
	                .parameterValidator(new ParamValidator3DGauss())
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

		return fittedEG;
	}
	
	// Convergence Checker
	private class ConvChecker3DGauss implements
			ConvergenceChecker<PointVectorValuePair> {

		int iteration_ = 0;
		boolean lastResult_ = false;

		public static final int INDEX_X0 = 0;
		public static final int INDEX_Y0 = 1;
		public static final int INDEX_Z0 = 2;
		public static final int INDEX_I0 = 3;
		public static final int INDEX_Bg = 4;

		@Override
		public boolean converged(int i, PointVectorValuePair previous,
				PointVectorValuePair current) {
			if (i == iteration_)
				return lastResult_;

			iteration_ = i;
			double[] p = previous.getPoint();
			double[] c = current.getPoint();

			if (Math.abs(p[INDEX_I0] - c[INDEX_I0]) < 0.1
					&& Math.abs(p[INDEX_Bg] - c[INDEX_Bg]) < 0.01
					&& Math.abs(p[INDEX_X0] - c[INDEX_X0]) < 0.002
					&& Math.abs(p[INDEX_Y0] - c[INDEX_Y0]) < 0.002
					&& Math.abs(p[INDEX_Z0] - c[INDEX_Z0]) < 0.01) {
				lastResult_ = true;
				return true;
			}

			lastResult_ = false;
			return false;
		}
	}

	private class ParamValidator3DGauss implements ParameterValidator {
		public static final int INDEX_I0 = 3;
		public static final int INDEX_Bg = 4;

		@Override
		public RealVector validate(RealVector arg) {
			if (arg.getEntry(INDEX_I0) < 0) {
				arg.setEntry(INDEX_I0, 0);
			}
			if (arg.getEntry(INDEX_Bg) < 0) {
				arg.setEntry(INDEX_Bg, 0);
			}
			return arg;
		}
	}

}
