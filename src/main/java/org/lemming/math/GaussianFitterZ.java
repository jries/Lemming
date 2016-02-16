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
	private static final int INDEX_C = 6;
	private static final int INDEX_D = 7;

	
	private ImageProcessor ip;
	private Roi roi;
	private int maxIter;
	private int maxEval;
	private int[] xgrid;
	private int[] ygrid;
	private double[] Ival;
	private double[] params;
	private double pixelSize;

	public GaussianFitterZ(ImageProcessor ip_, Roi roi_, int maxIter_, int maxEval_, double pixelSize_, double[] params_) {
		ip = ip_;
		roi = roi_;
		maxIter = maxIter_;
		maxEval = maxEval_;
		params = params_;
		pixelSize = pixelSize_;
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
		double RMS;
		int iter, eval;
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
			RMS = optimum.getRMS();
			iter = optimum.getIterations();
			eval = optimum.getEvaluations();
			//System.out.println("Too many evaluations:" + residuals.length);
		} catch(TooManyEvaluationsException | ConvergenceException | SingularMatrixException e){
			//System.out.println("Too many evaluations" + e.getMessage());
        	return null;
		}
        
		//check bounds
		if (!roi.contains((int)FastMath.round(fittedEG[0]), (int)FastMath.round(fittedEG[1])))
			return null;
		
		double[] result = new double[10];
		double[] error = get3DError(fittedEG, eg);
		result[0] = fittedEG[0]; // X								
		result[1] = fittedEG[1]; // Y
		result[2] = fittedEG[2]; // Z
		result[3] = error[0]; // Sy
		result[4] = error[1]; // Sx
		result[5] = error[2]; // Sz
		result[6] = fittedEG[3]; // I0
		result[7] = RMS;
		result[8] = iter;
		result[9] = eval;
		return result;
	}
	
	// Errors
	private double[] get3DError(double[] fittedEG, EllipticalGaussianZ eg) {
		// see thunderstorm corrections
		double[] error3d = new double[3];
		
		double sx,sy, dx2, dy2,dsx2, dsy2, dz2;
		int r=0, g=2;
		double N = fittedEG[INDEX_I0];
		double b = fittedEG[INDEX_Bg];
		double a2 = pixelSize*pixelSize;
		sx = eg.Sx(fittedEG[INDEX_Z0]);
		sy = eg.Sy(fittedEG[INDEX_Z0]);
		double sigma2 = a2*sx*sy;
		double l2 = params[INDEX_C]*params[INDEX_C];
		double d2 = params[INDEX_D]*params[INDEX_D];
		double tau = 2*FastMath.PI*(b*b+r)*(sigma2*(1+l2/d2)+a2/12)/(N*a2);
		
		dsx2 = (g*sx*sx+a2/12)*(1+8*tau)/N;
		dsy2 = (g*sy*sy+a2/12)*(1+8*tau)/N;
		dx2 = (g*sx*sx+a2/12)*(16/9+4*tau)/N;
		dy2 = (g*sy*sy+a2/12)*(16/9+4*tau)/N;
		error3d[0] = FastMath.sqrt(dx2);
		error3d[1] = FastMath.sqrt(dy2);

		double z2 = fittedEG[INDEX_Z0]*fittedEG[INDEX_Z0];
		double F2 = 4*l2*z2/(l2+d2+z2)/(l2+d2+z2);
		double dF2 = (1-F2)*(dsx2/(sx*sx)+dsy2/(sy*sy));

		dz2 = dF2*(l2+d2+z2)*(l2+d2+z2)*(l2+d2+z2)*(l2+d2+z2)/(4*l2*(l2+d2-z2)*(l2-d2-z2));
		error3d[2] = FastMath.sqrt(dz2);

		return error3d;
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

			if (FastMath.abs(p[INDEX_I0] - c[INDEX_I0]) < 0.1
					&& FastMath.abs(p[INDEX_Bg] - c[INDEX_Bg]) < 0.01
					&& FastMath.abs(p[INDEX_X0] - c[INDEX_X0]) < 0.002
					&& FastMath.abs(p[INDEX_Y0] - c[INDEX_Y0]) < 0.002
					&& FastMath.abs(p[INDEX_Z0] - c[INDEX_Z0]) < 0.01) {
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
			arg.setEntry(INDEX_I0, FastMath.abs(arg.getEntry(INDEX_I0)));
			arg.setEntry(INDEX_Bg, FastMath.abs(arg.getEntry(INDEX_Bg)));
			return arg;
		}
	}

}
