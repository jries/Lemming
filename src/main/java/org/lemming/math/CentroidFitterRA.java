package org.lemming.math;

import java.util.ArrayList;
import java.util.List;

import org.lemming.interfaces.Element;
import org.lemming.modules.Fitter;
import org.lemming.pipeline.Localization;

import net.imglib2.Cursor;
import net.imglib2.Interval;
import net.imglib2.RandomAccess;
import net.imglib2.RealPoint;
import net.imglib2.img.Img;
import net.imglib2.type.numeric.RealType;
import net.imglib2.view.IntervalView;
import net.imglib2.view.Views;

/**
 * Calculating centroids on a {@link #RandomAccessibleInterval}
 * 
 * @author Ronny Sczech
 *
 * @param <T>
 */
public class CentroidFitterRA<T extends RealType<T>>  {
	
	private IntervalView<T> op;
	private double thresh;
	private RealPoint center;

	public CentroidFitterRA(IntervalView<T> op_, double threshold_) {
		op = op_;
		thresh = threshold_;
		center = new RealPoint(op.numDimensions());
		for (int d = 0; d < op.numDimensions(); ++d)
			center.setPosition(op.min(d) + (op.dimension(d) / 2), d);
	}
	
	public double[] fitXY() {
		Cursor<T> c = op.cursor();
		int n = op.numDimensions();

		double[] r = new double[n * 2 + 1];
		double[] sum = new double[n];
		double s;

		while (c.hasNext()) {
			c.fwd();

			s = c.get().getRealDouble() - thresh;
			if (s > 0) {
				for (int i = 0; i < n; i++) {
					int pos = c.getIntPosition(i);
					r[i] += (center.getDoublePosition(i) - pos) * s;
					sum[i] += s;
				}
			}
		}

		for (int i = 0; i < n; i++){
			if (sum[i] == 0)
				return null;
			r[i] = (r[i] / sum[i]) + center.getDoublePosition(i);
		}

		return r;
	}
	
	public double[] fit() {
		Cursor<T> c = op.cursor();
		int n = op.numDimensions();

		double[] r = new double[n * 2 + 1];
		double[] sum = new double[n];
		double s;

		while (c.hasNext()) {
			c.fwd();

			s = c.get().getRealDouble() - thresh;
			if (s > 0) {
				for (int i = 0; i < n; i++) {
					int pos = c.getIntPosition(i);
					r[i] += (center.getDoublePosition(i) - pos) * s;
					sum[i] += s;
				}
			}
		}

		for (int i = 0; i < n; i++){
			if (sum[i] == 0)
				return null;
			r[i] = (r[i] / sum[i]) + center.getDoublePosition(i);
		}

		double[] dev = new double[n];
		c.reset();
		while (c.hasNext()) {
			c.fwd();
			s = c.get().getRealDouble() - thresh;
			if (s > 0)
				for (int i = 0; i < n; i++) {
					dev[i] += Math.abs(c.getIntPosition(i) - r[i]) * s;
				}
		}

		for (int i = 0; i < n; i++)
			r[i + n] = Math.sqrt(dev[i] / sum[i]);

		RandomAccess<T> ra = op.randomAccess();
		for (int i = 0; i < n; i++) {
			ra.setPosition(StrictMath.round(r[i]), i);
		}
		r[n * 2] = ra.get().getRealDouble();
		return r;
	}
	
	public double[] fitXYE() {
		Cursor<T> c = op.cursor();
		int n = op.numDimensions();

		double[] r = new double[n + 1];
		double[] sum = new double[n];
		double s;

		while (c.hasNext()) {
			c.fwd();

			s = c.get().getRealDouble() - thresh;
			if (s > 0) {
				for (int i = 0; i < n; i++) {
					int pos = c.getIntPosition(i);
					r[i] += (center.getDoublePosition(i) - pos) * s;
					sum[i] += s;
				}
			}
		}

		for (int i = 0; i < n; i++){
			if (sum[i] == 0)
				return null;
			r[i] = (r[i] / sum[i]);
		}

		double[] dev = new double[n];
		c.reset();
		while (c.hasNext()) {
			c.fwd();
			s = c.get().getRealDouble() - thresh;
			if (s > 0)
				for (int i = 0; i < n; i++) {
					dev[i] += Math.abs(c.getIntPosition(i) - r[i]) * s;
				}
		}
		
		r[n]= (dev[1] / sum[1])/(dev[0] / sum[0]);

		return r;
	}

	public static <T extends RealType<T>> List<Element> fit(List<Element> sliceLocs, Img<T> curImage, int halfKernel, double pixelSize) {
		final List<Element> found = new ArrayList<>();
        //final Rectangle imageRoi = ip.getRoi();
        long[] imageMin = new long[2];
        long[] imageMax = new long[2];
        for (Element el : sliceLocs) {
            final Localization loc = (Localization) el;
             
            long x = Math.round(loc.getX().doubleValue()/pixelSize);
			long y = Math.round(loc.getY().doubleValue()/pixelSize);
			curImage.min(imageMin);
			curImage.max(imageMax);
			final Interval roi = Fitter.cropInterval(imageMin,imageMax,new long[]{x - halfKernel,y - halfKernel},new long[]{x + halfKernel,y + halfKernel});
			final CentroidFitterRA<T> cf = new CentroidFitterRA<T>(Views.interval(curImage, roi),0);
            final double[] res = cf.fit();
         
            found.add(new Localization(res[0]*pixelSize, res[1]*pixelSize, res[4], 1L));
        }
 
        return found;
	}
}
