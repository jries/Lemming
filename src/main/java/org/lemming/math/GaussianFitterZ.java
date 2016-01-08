package org.lemming.math;

import org.apache.commons.math3.exception.ConvergenceException;
import org.apache.commons.math3.exception.TooManyEvaluationsException;
import org.apache.commons.math3.fitting.leastsquares.LeastSquaresBuilder;
import org.apache.commons.math3.fitting.leastsquares.LevenbergMarquardtOptimizer;
import org.apache.commons.math3.fitting.leastsquares.ParameterValidator;
import org.apache.commons.math3.fitting.leastsquares.LeastSquaresOptimizer.Optimum;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.linear.SingularMatrixException;
import org.apache.commons.math3.optim.ConvergenceChecker;
import org.apache.commons.math3.optim.PointVectorValuePair;
import org.apache.commons.math3.util.FastMath;
import org.apache.commons.math3.util.Precision;

import ij.gui.Roi;
import ij.process.ImageProcessor;

/**
 * Fitter module for the 3D astigmatism fit including direct Z calculation
 * 
 * @author Ronny Sczech
 *
 */
public class GaussianFitterZ {
	private static final int INDEX_X0 = 0;
	private static final int INDEX_Y0 = 1;
	private static final int INDEX_Z0 = 2;
	private static final int INDEX_I0 = 3;
	private static final int INDEX_Bg = 4;
	
	private static final int INDEX_WX = 0;
	private static final int INDEX_WY = 1;
	private static final int INDEX_AX = 2;
	private static final int INDEX_AY = 3;
	private static final int INDEX_BX = 4;
	private static final int INDEX_BY = 5;
	private static final int INDEX_C = 6;
	private static final int INDEX_D = 7;
	private static final int INDEX_Mp = 8;
	
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

	public double[] fit() {
		createGrids();
		EllipticalGaussianZ eg = new EllipticalGaussianZ(xgrid, ygrid, params);
		double[] initialGuess = eg.getInitialGuess(ip,roi);
		LevenbergMarquardtOptimizer optimizer = getOptimizer();
		double[] fittedEG;
		int iter = 0;
		try {
			final Optimum optimum = optimizer.optimize(
	                builder(eg)
	                .target(Ival)
	                .checkerPair(new ConvChecker3DGauss())
	                .parameterValidator(new ParamValidator3DGauss())
	                .start(initialGuess)
	                .maxIterations(maxIter)
	                .maxEvaluations(maxEval)
	                .build()
	        );
			fittedEG = optimum.getPoint().toArray();
			iter = optimum.getIterations();
			//System.out.println("Too many evaluations:" + residuals.length);
		} catch(TooManyEvaluationsException | ConvergenceException | SingularMatrixException e){
			//System.out.println("Too many evaluations" + e.getMessage());
        	return null;
		}
        
		//check bounds
		if (!roi.contains((int)FastMath.round(fittedEG[0]), (int)FastMath.round(fittedEG[1])))
			return null;
		
		double[] result = new double[9];
		
		result[0] = fittedEG[0];
		result[1] = fittedEG[1];
		result[2] = fittedEG[2];
		result[3] = fittedEG[3];
		result[4] = fittedEG[4];
		result[5] = get3DErrorX(1, fittedEG, params);
		result[6] = get3DErrorY(1, fittedEG, params);
		result[7] = get3DErrorZ(1, fittedEG, params);
		result[8] = iter;
		return result;
	}
	
	// Errors
	private static double get3DErrorX(double pixelsize, double[] fittedEG, double[] param){				// Mortensen et al., Nat methods (2010), LSQ 2D
	
		double sigmax = getValueWx(fittedEG[INDEX_Z0], param);
		
		double sigma2=2*sigmax*sigmax;
		double N = fittedEG[INDEX_I0];
		double b = fittedEG[INDEX_Bg];
		double a2 = pixelsize*pixelsize;
		
		double t = 2*Math.PI*b*(sigma2+a2/12)/(N*a2);
		
		double errorx2 = Math.abs((sigma2+a2/12)*(16/9+4*t)/N);
		
		return Math.sqrt(errorx2);
	}
	
	private static double get3DErrorY(double pixelsize, double[] fittedEG, double[] param){				// Mortensen et al., Nat methods (2010), LSQ 2D
		
		double sigmay = getValueWy(fittedEG[INDEX_Z0], param);
		
		double sigma2=2*sigmay*sigmay;
		double N = fittedEG[INDEX_I0];
		double b = fittedEG[INDEX_Bg];
		double a2 = pixelsize*pixelsize;
		
		double t = 2*Math.PI*b*(sigma2+a2/12)/(N*a2);
		
		double errory2 = Math.abs((sigma2+a2/12)*(16/9+4*t)/N);
		
		return Math.sqrt(errory2);
	}
	
	private static double get3DErrorZ(double pixelsize, double[] fittedEG, double[] param){				// Rieger & Stallinga, ChemPhyChem (2014), MLE z error
	
		double sigma02 = (param[INDEX_WX]+param[INDEX_WY])*(param[INDEX_WX]+param[INDEX_WY])/4;
		
		double N = fittedEG[INDEX_I0];
		double b = fittedEG[INDEX_Bg];
		double a2 = pixelsize*pixelsize;
		double l2 = param[INDEX_C]*param[INDEX_C];
		double d2 = param[INDEX_D]*param[INDEX_D];
				
		double tau = Math.abs(2*Math.PI*b*(sigma02*(1+l2/d2)+a2/12)/(N*a2));
		
		double sqrttau = Math.sqrt(9*tau/(1+4*tau));
		double errorz = Math.sqrt(1+8*tau+sqrttau);
		//double errorz = (l2+d2)*Math.sqrt(1+8*tau+sqrttau);

		return errorz;
	}
	
	
	//Helper methods
	private static double getValueWx(double z, double[] param) {
		double b = (z-param[INDEX_C]-param[INDEX_Mp])/param[INDEX_D];
		return param[INDEX_WX]*Math.sqrt(1+b*b+param[INDEX_AX]*b*b*b+param[INDEX_BX]*b*b*b*b);
	}

	private static double getValueWy(double z, double[] param) {
		double b = (z+param[INDEX_C]-param[INDEX_Mp])/param[INDEX_D];
		return param[INDEX_WY]*Math.sqrt(1+b*b+param[INDEX_AY]*b*b*b+param[INDEX_BY]*b*b*b*b);
	}
	
	// Convergence Checker
	private class ConvChecker3DGauss implements ConvergenceChecker<PointVectorValuePair> {

		int iteration_ = 0;
		boolean lastResult_ = false;

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

		@Override
		public RealVector validate(RealVector arg) {
			arg.setEntry(INDEX_I0, Math.abs(arg.getEntry(INDEX_I0)));
			arg.setEntry(INDEX_Bg, Math.abs(arg.getEntry(INDEX_Bg)));
			return arg;
		}
	}

}
