package org.lemming.math;

import ij.gui.Roi;
import ij.process.ImageProcessor;

import org.apache.commons.math3.exception.ConvergenceException;
import org.apache.commons.math3.exception.TooManyEvaluationsException;
import org.apache.commons.math3.fitting.leastsquares.LeastSquaresBuilder;
import org.apache.commons.math3.fitting.leastsquares.LeastSquaresOptimizer.Optimum;
import org.apache.commons.math3.fitting.leastsquares.LevenbergMarquardtOptimizer;

public class GaussianFitterAlternative implements FitterInterface {

	int[] xgrid, ygrid;									// x and y positions of the pixels	
	double[] Ival;										// intensity value of the pixels
	private ImageProcessor ip;
	private Roi roi;
	private int maxIter;
	public static int PARAM_2D_LENGTH = 6;	
	
	public GaussianFitterAlternative(ImageProcessor ip_, Roi roi_, int maxIter_) {
		ip = ip_;
		roi = roi_;
		maxIter = maxIter_;
	}
	
	private static LeastSquaresBuilder builder(EllipticalGaussian problem){
    	LeastSquaresBuilder builder = new LeastSquaresBuilder();
    	 builder.model(problem.getModelFunction(), problem.getModelFunctionJacobian());
		return builder;
    }
	
	private static LevenbergMarquardtOptimizer getOptimizer() {
        return new LevenbergMarquardtOptimizer();
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
	                .start(eg.getInitialGuess(ip,roi))
	                .maxIterations(maxIter)
	                .maxEvaluations(maxIter)
	                .build()
	        );
			fittedEG =  optimum.getPoint().toArray();
		} catch(TooManyEvaluationsException | ConvergenceException e){
			//System.out.println("Too many evaluations" + e.getMessage());
        	return null;
		}
        
        result[0] = fittedEG[0];
        result[1] = fittedEG[1];
        result[2] = Math.abs(fittedEG[2]);
        result[3] = Math.abs(fittedEG[3]);
        	
		return result;
	}

}
