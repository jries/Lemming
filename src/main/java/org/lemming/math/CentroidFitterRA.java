package org.lemming.math;

import java.util.ArrayList;
import java.util.Arrays;
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
 * Calculating centroids on a
 * 
 * @author Ronny Sczech
 *
 * @param <T> data type
 */
public class CentroidFitterRA<T extends RealType<T>>  {
	
	private final IntervalView<T> op;
	private final double thresh;
	private final RealPoint center;

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
			r[i] = (r[i] / sum[i]) + center.getDoublePosition(i) + 0.5;
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
		final Cursor<T> c = op.cursor();
		final int n = op.numDimensions();

		double[] r = new double[n + 1];
		double[] t = new double[n];
		double[][] sum = new double[n][];
		for (int i = 0; i < n; i++) sum[i]=new double[(int) op.dimension(i)];
		double[] min = new double[n];
		Arrays.fill(min, Double.MAX_VALUE);
		double[] sumsum = new double[n];
		double[] W = new double[n];
		double[] W2 = new double[n];
		double s; 
		int localPos, i, j;
		long[] pos = new long[n];

		while (c.hasNext()) {
			c.fwd();

			s = c.get().getRealDouble() - thresh;
			c.localize(pos);
			if (s > 0) {
				for (i = 0; i < n; i++){ 
					localPos = (int) (pos[i]-op.min(i));
					sum[i][localPos] += s;	
					}
			}
		}
		
		for (i = 0; i < n; i++)
			for (j = 0; j<op.dimension(i); j++)
				if (min[i]>sum[i][j]) min[i]=sum[i][j];

		for (i = 0; i < n; i++)
			for (j = 0; j<op.dimension(i); j++){
				sum[i][j] -= min[i];
				sumsum[i] += sum[i][j];
				t[i] += sum[i][j]*(j+1);
			}
		
		for (i = 0; i < n; i++)
			t[i] /= sumsum[i];
		
		for (i = 0; i < n; i++)
			for (j = 0; j<op.dimension(i); j++){
				W[i] += sum[i][j]*Math.abs(1-t[i]+j);
			}
		
		for (i = 0; i < n; i++)
			W[i] = W[i] / sumsum[i] * 3;
		
		Arrays.fill(sumsum, 0);
		
		for (i = 0; i < n; i++)
			for (j = 0; j<op.dimension(i); j++)
				if(j >= Math.floor(t[i]-W[i]) && j < Math.floor(t[i]+W[i])){
					sumsum[i] += sum[i][j];
					r[i] += sum[i][j]*(j+1);
				}
		
		for (i = 0; i < n; i++)
			r[i] /= sumsum[i];
		
		for (i = 0; i < n; i++)
			for (j = 0; j<op.dimension(i); j++)
				if(j >= Math.floor(r[i]-W[i]) && j < Math.floor(r[i]+W[i]))
					W2[i] += sum[i][j]*Math.abs(1-r[i]+j);
		
		for (i = 0; i < n; i++)
			W2[i] = W2[i] / sumsum[i];
		
		for (i = 0; i < n; i++)
			r[i] = (r[i]-(op.dimension(i)-1)/2-1);
		r[n]=W2[n-1];
		for (i = n-2; i >= 0; i--)
			r[n] /= W2[i];
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
			final CentroidFitterRA<T> cf = new CentroidFitterRA<>(Views.interval(curImage, roi), 0);
            final double[] res = cf.fit();
         
            found.add(new Localization(res[0]*pixelSize, res[1]*pixelSize, res[4], 1L));
        }
 
        return found;
	}
}
