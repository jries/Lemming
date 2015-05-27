package org.lemming.math;

import net.imglib2.Localizable;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.algorithm.localization.LevenbergMarquardtSolver;
import net.imglib2.algorithm.localization.Observation;
import net.imglib2.algorithm.region.localneighborhood.RectangleCursor;
import net.imglib2.algorithm.region.localneighborhood.RectangleNeighborhoodGPL;
import net.imglib2.type.numeric.RealType;

public class GaussianFitter<T extends RealType<T>> {

	private RandomAccessibleInterval<T> image;
	private int ndims;
	private String errorMessage;

	public GaussianFitter(RandomAccessibleInterval<T> image_) {
		image = image_;
		ndims = image.numDimensions();
	}
	
	//Ensure the image is not null.
	public boolean checkInput() {
		 if (null == image) {
			 errorMessage = "Image is null.";
			 return false;
		 }
		 return true;
	}
	
	/**
	 * <p>
	 * Fit an elliptical gaussian to a peak in the image, using the formula:
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
	public double[] process(final Localizable point, final double[] typical_sigma) {
		// Determine the size of the data to gather
		long[] pad_size = new long[ndims];
		for (int i = 0; i < ndims; i++) {
			pad_size[i] = (long) Math.ceil(2 * typical_sigma[i]);
		}
		// Gather data around peak
		final Observation data = gatherObservationData(point, pad_size);
		final double[][] X = data.X;
		final double[] I = data.I;
		// Make best guess
		double[] start_param = makeBestGuess(X, I);
		// Correct for too large sigmas: we drop estimate and replace it by user
		// input
		for (int j = 0; j < ndims; j++) {
			if (start_param[j + ndims + 1] < 1 / (typical_sigma[j] * typical_sigma[j]))
				start_param[j + ndims + 1] = 1 / (typical_sigma[j] * typical_sigma[j]);
		}
		// Prepare optimizer
		int maxiter = 300;
		double lambda = 1e-3;
		double termepsilon = 1e-1;

		final double[] a = start_param.clone();

		try {
			LevenbergMarquardtSolver.solve(X, a, I, new GaussianMultiDLM(),
					lambda, termepsilon, maxiter);
		} catch (Exception e) {
			e.printStackTrace();
		}

		// NaN protection: we prefer returning the crude estimate that NaN
		for (int j = 0; j < a.length; j++) {
			if (Double.isNaN(a[j]))
				a[j] = start_param[j];
		}

		return a;
	}

	private double[] makeBestGuess(double[][] X, double[] I) {
		double[] start_param = new double[2 * ndims + 1];

		double[] X_sum = new double[ndims];
		for (int j = 0; j < ndims; j++) {
			X_sum[j] = 0;
			for (int i = 0; i < X.length; i++) {
				X_sum[j] += X[i][j] * I[i];
			}
		}

		double I_sum = 0;
		double max_I = Double.NEGATIVE_INFINITY;
		for (int i = 0; i < X.length; i++) {
			I_sum += I[i];
			if (I[i] > max_I) 
				max_I = I[i];
		}

		start_param[0] = max_I;

		for (int j = 0; j < ndims; j++) {
			start_param[j + 1] = X_sum[j] / I_sum;
		}

		for (int j = 0; j < ndims; j++) {
			double C = 0;
			double dx;
			for (int i = 0; i < X.length; i++) {
				dx = X[i][j] - start_param[j + 1];
				C += I[i] * dx * dx;
			}
			C /= I_sum;
			start_param[ndims + j + 1] = 1 / C;
		}
		return start_param;
	}

	private Observation gatherObservationData(Localizable point, long[] pad_size) {
		RectangleNeighborhoodGPL<T> neighborhood = new RectangleNeighborhoodGPL<>(image);
		neighborhood.setSpan(pad_size);
		neighborhood.setPosition(point);

		int n_pixels = (int) neighborhood.size();
		double[] tmp_I = new double[n_pixels];
		double[][] tmp_X = new double[n_pixels][ndims];

		RectangleCursor<T> cursor = neighborhood.localizingCursor();
		long[] pos = new long[image.numDimensions()];

		int index = 0;
		while (cursor.hasNext()) {

			cursor.fwd();
			cursor.localize(pos); // This is the absolute ROI position
			if (cursor.isOutOfBounds()) {
				continue;
			}

			for (int i = 0; i < ndims; i++) {
				tmp_X[index][i] = pos[i];
			}

			tmp_I[index] = cursor.get().getRealDouble();
			index++;
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

		Observation obs = new Observation();
		obs.I = I;
		obs.X = X;
		return obs;
	}
	
	public String getErrorMessage() {
		return errorMessage;
	}
}
