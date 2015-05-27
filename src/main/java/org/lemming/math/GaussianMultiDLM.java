package org.lemming.math;

import net.imglib2.algorithm.localization.FitFunction;

public class GaussianMultiDLM implements FitFunction {

	@Override
	public double grad(double[] x, double[] a, int k) {
		final int ndims = x.length;
		if (k == 0) {
			// With respect to A
			return E(x, a);

		} else if (k <= ndims) {
			// With respect to xi
			int dim = k - 1;
			return 2 * a[dim + ndims] * (x[dim] - a[dim + 1]) * a[0] * E(x, a);

		} else {
			// With respect to ai
			int dim = k - ndims - 1;
			double di = x[dim] - a[dim + 1];
			return -di * di * a[0] * E(x, a);
		}
	}

	@Override
	public double hessian(double[] x, double[] a, int r_, int c_) {
		int c = c_;
		int r = r_;
		if (c < r) {
			int tmp = c;
			c = r;
			r = tmp;
		} // Ensure c >= r, top right half the matrix

		final int ndims = x.length;

		if (r == 0) {
			// 1st line

			if (c == 0) {
				return 0;

			} else if (c <= ndims) {
				// dÂ²G / (dA dxi)
				final int dim = c - 1;
				return 2 * a[dim + ndims] * (x[dim] - a[dim + 1]) * E(x, a);

			} else {
				// dÂ²G / (dA dsi)
				final int dim = c - ndims - 1;
				final double di = x[dim] - a[dim + 1];
				return -di * di * E(x, a);
			}

		} else if (c == r) {
			// diagonal

			if (c <= ndims) {
				// dÂ²G / dxiÂ²
				final int dim = c - 1;
				final double di = x[dim] - a[dim + 1];
				return 2 * a[0] * E(x, a) * a[dim + ndims]
						* (2 * a[dim + ndims] * di * di - 1);
			}
			// dÂ²G / dsiÂ²
			final int dim = c - ndims - 1;
			final double di = x[dim] - a[dim + 1];
			return a[0] * E(x, a) * di * di * di * di;

		} else if (c <= ndims && r <= ndims) {
			// H1
			// dÂ²G / (dxj dxi)
			final int i = c - 1;
			final int j = r - 1;
			final double di = x[i] - a[i + 1];
			final double dj = x[j] - a[j + 1];
			return 4 * a[0] * E(x, a) * a[i + ndims] * a[j + ndims] * di * dj;

		} else if (r <= ndims && c > ndims) {
			// H3
			// dÂ²G / (dxi dsj)
			final int i = r - 1; // xi
			final int j = c - ndims - 1; // sj
			final double di = x[i] - a[i + 1];
			final double dj = x[j] - a[j + 1];
			return -2 * a[0] * E(x, a) * a[i + ndims] * di
					* (1 - a[j + ndims] * dj * dj);

		} else {
			// H2
			// dÂ²G / (dsj dsi)
			final int i = r - ndims - 1; // si
			final int j = c - ndims - 1; // sj
			final double di = x[i] - a[i + 1];
			final double dj = x[j] - a[j + 1];
			return a[0] * E(x, a) * di * di * dj * dj;
		}
	}

	@Override
	public double val(double[] x, double[] a) {
		return a[0] * E(x, a);
	}
	
	private static final double E(final double[] x, final double[] a) {
		double di, sum = 0;
		for (int i = 0; i < x.length; i++) {
			di = x[i] - a[i+1];
			sum += a[i+x.length+1] * di * di;
		}
		return Math.exp(-sum);		
	}

}
