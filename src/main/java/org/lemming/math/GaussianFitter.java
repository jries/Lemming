package org.lemming.math;

import java.util.Arrays;

import net.imglib2.RandomAccessibleInterval;
import net.imglib2.RealLocalizable;
import net.imglib2.algorithm.localization.LevenbergMarquardtSolver;
import net.imglib2.algorithm.localization.Observation;
import net.imglib2.algorithm.region.localneighborhood.RectangleCursor;
import net.imglib2.algorithm.region.localneighborhood.RectangleNeighborhoodGPL;
import net.imglib2.type.numeric.RealType;

public class GaussianFitter<T extends RealType<T>> implements FitterInterface {

	private RandomAccessibleInterval<T> image;
	private int ndims;
	private RealLocalizable point;
	private double[] typical_sigma;
	//private static double defaultSigma = 1.5;

	public GaussianFitter(RandomAccessibleInterval<T> image_, final double[] sigma) {
		image = image_;
		ndims = image.numDimensions();
		typical_sigma = sigma;
	}
	
	public void setPoint(final RealLocalizable point_){
		point = point_;
	}
	
	
	/**
	 * <p>
	 * Fit an elliptical gaussian to a peak in the image.
	 * First observation arrays are built by collecting pixel positions and
	 * intensities around the given peak location. These arrays are then used to
	 * guess a starting set of parameters, that is then fed to least-square
	 * optimization procedure, using the Levenberg-Marquardt curve fitter.
	 * <p>
	 * Calls to this function does not generate any class field, and can
	 * therefore by readily parallelized with multiple peaks on the same image.
	 * 
	 * @param point
	 *            the approximate coordinates of the peak
	 * @param typical_sigma
	 *            the typical sigma of the peak
	 * @return a <code>2*ndims+1</code> elements double array containing fit
	 *         estimates
	 */
	@Override
	public double[] fit() {
		// Determine the size of the data to gather
		long[] pad_size = new long[ndims];
		for (int i = 0; i < ndims; i++) {
			pad_size[i] = (long) Math.ceil(2 * typical_sigma[i]);
		}
		// Gather data around peak & start parameters
		double[] start_param = new double[2 * ndims + 1];
		final Observation data = gatherObservationData(pad_size, start_param);
		final double[][] X = data.X;
		final double[] I = data.I;
		// Make best guess
		//double[] start_param = makeBestGuess(X, I);
		// Correct for too large sigmas: we drop estimate and replace it by user input
		for (int j = 0; j < ndims; j++) {
			if (start_param[j + ndims + 1] < 1 / (typical_sigma[j] * typical_sigma[j]))
				start_param[j + ndims + 1] = 1 / (typical_sigma[j] * typical_sigma[j]);
		}
		// Prepare optimizer
		int maxiter = 100;
		double lambda = 1e-3;
		double termepsilon = 0.1;

		final double[] a = start_param.clone();

		try {
			LevenbergMarquardtSolver.solve(X, a, I, new GaussianMultiDLM(),
					lambda, termepsilon, maxiter);
		} catch (Exception e) {
			e.printStackTrace();
		}

		// NaN protection: we prefer returning the crude estimate than NaN
		for (int j = 0; j < a.length; j++) {
			if (Double.isNaN(a[j]))
				a[j] = start_param[j];
		}

		return a;
	}

	private Observation gatherObservationData(long[] pad_size, double[] start_param) {
		RectangleNeighborhoodGPL<T> neighborhood = new RectangleNeighborhoodGPL<>(image);
		neighborhood.setSpan(pad_size);
		long[] intPoint = new long[ndims];
		for (int i=0; i<ndims;i++)
			intPoint[i]=Math.round(point.getDoublePosition(i));
		neighborhood.setPosition(intPoint);

		int n_pixels = (int) neighborhood.size();
		double[] tmp_I = new double[n_pixels];
		double[][] tmp_X = new double[n_pixels][ndims];

		RectangleCursor<T> cursor = neighborhood.localizingCursor();
		long[] pos = new long[ndims];
		double[] X_sum = new double[ndims];
		Arrays.fill(X_sum, 0);
		double I_sum = 0;
		double max_I = Double.NEGATIVE_INFINITY;
		double min_I = Double.POSITIVE_INFINITY;
		int index = 0;
		double val;
		while (cursor.hasNext()) {

			cursor.fwd();
			cursor.localize(pos); // This is the absolute ROI position
			if (cursor.isOutOfBounds()) {
				continue;
			}
			
			tmp_I[index] = val = cursor.get().getRealDouble();
			max_I = Math.max(max_I, val);
			min_I = Math.min(min_I, val);
			I_sum += val;
			
			for (int i = 0; i < ndims; i++) {
				tmp_X[index][i] = pos[i];
				X_sum[i] += pos[i] * val;
			}
			
			index++;
		}
		
		// start parameter
		
		start_param[0] = max_I;
		for (int j = 0; j < ndims; j++) {
			start_param[j + 1] = X_sum[j] / I_sum;
		}
		
		/*for (int j = 0; j < ndims; j++) {
			start_param[ndims + j + 1] = defaultSigma;
		}*/
		
		for (int j = 0; j < ndims; j++) {
			double C = 0;
			double dx;
			for (int i = 0; i < tmp_X.length; i++) {
				dx = tmp_X[i][j] - start_param[j];
				C += tmp_I[i] * dx * dx;
			}
			C /= I_sum;
			start_param[ndims + j + 1] = 1 / C / 2;
		}
		
		// Now we possibly resize the arrays, in case we have been too close to
		// the image border.
		double[][] X = null;
		double[] I = null;
		if (index == n_pixels) {
			// OK, we have gone through the whole square
			X = tmp_X;
			I = tmp_I;
		} else {
			// Re-dimension the arrays
			X = new double[index][ndims];
			I = new double[index];
			System.arraycopy(tmp_X, 0, X, 0, index);
			System.arraycopy(tmp_I, 0, I, 0, index);
		}
		
		// Subtract background
		for (int j = 0; j < I.length; j++)
			I[j] -= min_I;

		Observation obs = new Observation();
		obs.I = I;
		obs.X = X;
		return obs;
	}
}
