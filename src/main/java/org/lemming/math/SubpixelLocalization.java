package org.lemming.math;

import java.util.ArrayList;
import java.util.List;

import org.apache.commons.math3.util.FastMath;
import org.lemming.interfaces.Element;
import org.lemming.pipeline.LocalizationPrecision3D;
import org.lemming.pipeline.Localization;

import Jama.LUDecomposition;
import Jama.Matrix;
import net.imglib2.Interval;
import net.imglib2.Localizable;
import net.imglib2.Point;
import net.imglib2.RandomAccess;
import net.imglib2.RandomAccessible;
import net.imglib2.RealPoint;
import net.imglib2.RealPositionable;
import net.imglib2.algorithm.localextrema.RefinedPeak;
import net.imglib2.type.numeric.RealType;
import net.imglib2.util.Intervals;

/**
 * Refine a set of peaks to subpixel coordinates. Single-threaded version.
 * <p>
 * A List {@link RefinedPeak} for the given list of {@link Localizable} is
 * computed by, for each peak, fitting a quadratic function to the image and
 * computing the subpixel coordinates of the extremum. This is an iterative
 * procedure. If the extremum is shifted more than 0.5 in one or more the fit is
 * repeated at the corresponding integer coordinates. This is repeated to
 * convergence, for a maximum number of iterations, or until the integer
 * coordinates move out of the valid image.
 * 
 */
public class SubpixelLocalization {
	public static <T extends RealType<T>> List<Element> refinePeaks(final List<Element> sliceLocs, final RandomAccessible<T> img,
			final Interval validInterval, final boolean returnInvalidPeaks, final int maxNumMoves, final boolean allowMaximaTolerance,
			final float maximaTolerance, final boolean[] allowedToMoveInDim, final double pixelDepth) {

		final List<Element> refinedPeaks = new ArrayList<>();

		final int n = img.numDimensions();

		// the current position for the quadratic fit
		final Point currentPosition = new Point(n);

		// gradient vector and Hessian matrix at the current position
		final Matrix g = new Matrix(n, 1);
		final Matrix H = new Matrix(n, n);

		// the current subpixel offset estimate
		final RealPoint subpixelOffset = new RealPoint(n);

		// bounds checking necessary?
		final boolean canMoveOutside = (validInterval == null);
		final Interval interval = canMoveOutside ? null : Intervals.expand(validInterval, -1);

		// the cursor for the computation
		final RandomAccess<T> access = canMoveOutside ? img.randomAccess() : img.randomAccess(validInterval);

		for (final Element locs : sliceLocs) {
			Localization p = (Localization) locs;
			currentPosition.setPosition(new long[] { FastMath.round((p.getX().doubleValue()/pixelDepth)), 
					FastMath.round((p.getY().doubleValue()/pixelDepth)) });

			// fit n-dimensional quadratic function to the extremum and
			// if the extremum is shifted more than 0.5 in one or more
			// directions we test whether it is better there
			// until we
			// - converge (find a stable extremum)
			// - move out of the Img
			// - achieved the maximal number of moves
			boolean foundStableMaxima = false;
			for (int numMoves = 0; numMoves < maxNumMoves; ++numMoves) {
				// check validity of the current location
				if (!(canMoveOutside || Intervals.contains(interval, currentPosition))) {
					break;
				}

				quadraticFitOffset(currentPosition, access, g, H, subpixelOffset);

				// test all dimensions for their change
				// if the absolute value of the subpixel location
				// is bigger than 0.5 we move into that direction
				//
				// Normally, above an offset of 0.5 the base position
				// has to be changed, e.g. a subpixel location of 4.7
				// would mean that the new base location is 5 with an offset of
				// -0.3
				//
				// If we allow an increasing maxima tolerance we will
				// not change the base position that easily. Sometimes
				// it simply jumps from left to right and back, because
				// it is 4.51 (i.e. goto 5), then 4.49 (i.e. goto 4)
				// Then we say, ok, lets keep the base position even if
				// the subpixel location is 0.6...
				foundStableMaxima = true;
				final double threshold = allowMaximaTolerance ? 0.5 + numMoves * maximaTolerance : 0.5;
				for (int d = 0; d < n; ++d) {
					final double diff = subpixelOffset.getDoublePosition(d);
					if (FastMath.abs(diff) > threshold) {
						if (allowedToMoveInDim[d]) {
							// move to another base location
							currentPosition.move(diff > 0 ? 1 : -1, d);
							foundStableMaxima = false;
						}
					}
				}
				if (foundStableMaxima) {
					break;
				}
			}

			if (foundStableMaxima) {
				// set the results if everything went well
				subpixelOffset.move(0.5, 0);
				subpixelOffset.move(0.5, 1);
				final double sx = FastMath.pow(subpixelOffset.getDoublePosition(0),2);
				final double sy = FastMath.pow(subpixelOffset.getDoublePosition(1),2);
				access.setPosition(currentPosition);
				subpixelOffset.move(currentPosition);
				refinedPeaks.add(new LocalizationPrecision3D(subpixelOffset.getDoublePosition(0) * pixelDepth,
						subpixelOffset.getDoublePosition(1) * pixelDepth, 0, sx * pixelDepth, sy * pixelDepth, 0, access.get().getRealDouble(), p.getFrame()));
			} else if (returnInvalidPeaks) {
				refinedPeaks.add(new LocalizationPrecision3D(p.getX(), p.getY(), 0, 0, 0, 0, access.get().getRealDouble(), p.getFrame()));
			}
		}

		return refinedPeaks;
	}

	/**
	 * Estimate subpixel <code>offset</code> of extremum of quadratic function
	 * fitted at <code>p</code>.
	 * 
	 * @param p
	 *            integer position at which to fit quadratic.
	 * @param access
	 *            access to the image values.
	 * @param g
	 *            a <em>n</em> vector where <em>n</em> is the dimensionality of
	 *            the image. (This is a temporary variable to store the
	 *            gradient).
	 * @param H
	 *            a <em>n &times; n</em> matrix where <em>n</em> is the
	 *            dimensionality of the image. (This is a temporary variable to
	 *            store the Hessian).
	 * @param offset
	 *            subpixel offset of extremum value <code>p</code> is stored
	 *            here.
	 * @param <T>	
	 * 			  data type
	 */
	private static <T extends RealType<T>> void quadraticFitOffset(final Localizable p, final RandomAccess<T> access, final Matrix g,
																   final Matrix H, final RealPositionable offset) {
		final int n = p.numDimensions();

		access.setPosition(p);

		final double a1 = access.get().getRealDouble();
		for (int d = 0; d < n; ++d) {
			// @formatter:off
			// gradient
			// we compute the derivative for dimension d like this
			//
			// | a0 | a1 | a2 |
			// ^
			// |
			// Original position of access
			//
			// g(d) = (a2 - a0)/2
			// we divide by 2 because it is a jump over two pixels
			// @formatter:on
			access.bck(d);
			final double a0 = access.get().getRealDouble();
			access.move(2, d);
			final double a2 = access.get().getRealDouble();
			g.set(d, 0, (a2 - a0) * 0.5);

			// @formatter:off
			// Hessian
			// diagonal element for dimension d
			// computed from the row a in the input
			//
			// | a0 | a1 | a2 |
			// ^
			// |
			// Original position of access
			//
			// H(dd) = (a2-a1) - (a1-a0)
			// = a2 - 2*a1 + a0
			// @formatter:on
			H.set(d, d, a2 - 2 * a1 + a0);

			// off-diagonal Hessian elements H(de) = H(ed) are computed as a
			// combination of dimA (dimension a) and dimB (dimension b), i.e. we
			// always operate in a two-dimensional plane
			// ______________________
			// | a0b0 | a1b0 | a2b0 |
			// | a0b1 | a1b1 | a2b1 |
			// | a0b2 | a1b2 | a2b2 |
			// ----------------------
			// where a1b1 is the original position of the access
			//
			// H(ab) = ( (a2b2-a0b2)/2 - (a2b0 - a0b0)/2 )/2
			//
			// we divide by 2 because these are always jumps over two pixels
			for (int e = d + 1; e < n; ++e) {
				access.fwd(e);
				final double a2b2 = access.get().getRealDouble();
				access.move(-2, d);
				final double a0b2 = access.get().getRealDouble();
				access.move(-2, e);
				final double a0b0 = access.get().getRealDouble();
				access.move(2, d);
				final double a2b0 = access.get().getRealDouble();
				// back to the original position
				access.bck(d);
				access.fwd(e);
				final double v = (a2b2 - a0b2 - a2b0 + a0b0) * 0.25;
				H.set(d, e, v);
				H.set(e, d, v);
			}
		}

		// Do not move in a plane if the matrix is singular.
		final LUDecomposition decomp = new LUDecomposition(H);
		if (decomp.isNonsingular()) {
			final Matrix minusOffset = decomp.solve(g);
			for (int d = 0; d < n; ++d) {
				offset.setPosition(-minusOffset.get(d, 0), d);
			}
		} else {
			for (int d = 0; d < n; d++) {
				offset.setPosition(0L, d);
			}
		}
	}
}
