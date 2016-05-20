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

import net.imglib2.Cursor;
import net.imglib2.type.numeric.RealType;
import net.imglib2.view.IntervalView;
import org.lemming.tools.LemmingUtils;

/**
 * a fast symmetric 2D Gaussian fitter
 * 
 * @author Ronny Sczech
 *
 */
public class Symmetric2DFitter<T extends RealType<T>> {
	
	private static final int INDEX_X0 = 0;
	private static final int INDEX_Y0 = 1;
	private static final int INDEX_S = 2;
	private static final int INDEX_SX = 2;
	private static final int INDEX_SY = 3;
	private static final int INDEX_I0 = 3;
	private static final int INDEX_Bg = 4;
	
	private final int maxIter;
	private final int maxEval;
	private int[] xgrid;
	private int[] ygrid;
	private double[] Ival;
	private final IntervalView<T> interval;
	private static final int PARAM_LENGTH = 5;
	private final T bg;
	private final T max;

	public Symmetric2DFitter(final IntervalView<T> interval_, int maxIter_, int maxEval_) {
		interval = interval_;
		maxIter = maxIter_;
		maxEval = maxEval_;
		bg = LemmingUtils.computeMin(interval);
		max = LemmingUtils.computeMax(interval);
	}
	
	private void createGrids(){
		Cursor<T> cursor = interval.cursor();
		int arraySize=(int)(interval.dimension(0)*interval.dimension(1));
		Ival = new double[arraySize];
		xgrid = new int[arraySize];
		ygrid = new int[arraySize];
		int index=0;
		while(cursor.hasNext()){
			cursor.fwd();
			xgrid[index]=cursor.getIntPosition(0);
			ygrid[index]=cursor.getIntPosition(1);
			Ival[index++]=cursor.get().getRealDouble();
		}
	}

	private double[] getInitialGuess(IntervalView<T> interval) {
		double[] initialGuess = new double[PARAM_LENGTH];

		CentroidFitterRA<T> cf = new CentroidFitterRA<>(interval, 0);
		double[] centroid = cf.fit();

		initialGuess[INDEX_X0] = centroid[INDEX_X0];
		initialGuess[INDEX_Y0] = centroid[INDEX_Y0];
		initialGuess[INDEX_S]  = 0.5d*(centroid[INDEX_SY] + centroid[INDEX_SX]);
		initialGuess[INDEX_I0] = max.getRealDouble();
		initialGuess[INDEX_Bg] = bg.getRealDouble();

		return initialGuess;
	}
	
	public double[] fit() {
		createGrids();
		SymmetricGaussian eg = new SymmetricGaussian(xgrid, ygrid);
		final LevenbergMarquardtOptimizer optimizer = new LevenbergMarquardtOptimizer();
		final LeastSquaresBuilder builder = new LeastSquaresBuilder();
		builder.model(eg.getModelFunction(), eg.getModelFunctionJacobian());
		double[] fittedEG;
		int iter;
		try {
			final Optimum optimum = optimizer.optimize(
	                builder
	                .target(Ival)
	                .checkerPair(new ConvChecker2DGauss())
                    .parameterValidator(new ParamValidator2DGauss())
	                .start(getInitialGuess(interval))
	                .maxIterations(maxIter)
	                .maxEvaluations(maxEval)
	                .build()
	        );
			fittedEG = optimum.getPoint().toArray();
			iter = optimum.getIterations();	        
		} catch(TooManyEvaluationsException  e){
        	return null;
		} catch(ConvergenceException e){
        	return null;
		}

		double[] result = new double[7];
		result[0] = fittedEG[0]+0.5;
		result[1] = fittedEG[1]+0.5;
		result[2] = fittedEG[2];
		result[3] = fittedEG[3];
		result[4] = fittedEG[4];
		result[5] = get2DError(1, fittedEG);
		result[6] = iter;
		return result;
	}
	
	private static double get2DError(int pixelsize, double[] fittedEG) {
		double sigma2=2*fittedEG[INDEX_S]*fittedEG[INDEX_S];
		double N = fittedEG[INDEX_I0];
		double b = fittedEG[INDEX_Bg];
		double a2 = pixelsize*pixelsize;
		
		double t = 2*Math.PI*b*(sigma2+a2/12)/(N*a2);
		
		double errorx2 = (sigma2+a2/12)*(16/9+4*t)/N;
		
		return Math.sqrt(errorx2);
	}

	
	private class ConvChecker2DGauss implements ConvergenceChecker<PointVectorValuePair> {
	    
		int iteration_ = 0;
	    boolean lastResult_ = false;

		@Override
		public boolean converged(int i, PointVectorValuePair previous, PointVectorValuePair current) {
			if (i == iteration_)
	           return lastResult_;

			if (i >100){
				 return true;
			}
			
			iteration_ = i;
	          double[] p = previous.getPoint();
	          double[] c = current.getPoint();
	          
	          if ( Math.abs(p[INDEX_I0] - c[INDEX_I0]) < 0.01  &&
	                  Math.abs(p[INDEX_Bg] - c[INDEX_Bg]) < 0.01 &&
	                  Math.abs(p[INDEX_X0] - c[INDEX_X0]) < 0.001 &&
	                  Math.abs(p[INDEX_Y0] - c[INDEX_Y0]) < 0.001 &&
	                  Math.abs(p[INDEX_S] - c[INDEX_S]) < 0.002  ) {
	             lastResult_ = true;
	             return true;
	          }
	        lastResult_ = false;
	        return false;
		}
	}

	private class ParamValidator2DGauss implements ParameterValidator {
		
		@Override
		public RealVector validate(RealVector arg) {
			
			arg.setEntry(INDEX_S, Math.abs(arg.getEntry(INDEX_S)));
			arg.setEntry(INDEX_I0, Math.max(1,Math.min(arg.getEntry(INDEX_I0), max.getRealDouble()*4)));
			arg.setEntry(INDEX_Bg, Math.max(arg.getEntry(INDEX_Bg), bg.getRealDouble()/2));
			arg.setEntry(INDEX_X0, Math.abs(arg.getEntry(INDEX_X0)));
			arg.setEntry(INDEX_Y0, Math.abs(arg.getEntry(INDEX_Y0)));
			return arg;
		}

	}

}
