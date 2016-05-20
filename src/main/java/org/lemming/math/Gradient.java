package org.lemming.math;

import org.apache.commons.math3.util.FastMath;

import net.imglib2.RandomAccess;
import net.imglib2.RealRandomAccess;
import net.imglib2.interpolation.randomaccess.LanczosInterpolatorFactory;
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
public class Gradient<T extends RealType<T>>  {
	
	private final IntervalView<T> op;
	private final double thresh;
	private final int radiusGrad;

	public Gradient(IntervalView<T> op_, final double threshold_, int radiusGrad_){
		op = op_;
		thresh = threshold_;
		radiusGrad=radiusGrad_;	
	}
	
	public final double[] fit(){
		CentroidFitterRA<T> cf = new CentroidFitterRA<>(op, thresh);
		final double[] result = cf.fitXYE();
		final double x0 = result[0];
		final double y0 = result[1];
		final double e0 = result[2];
		int r2 = 2 * radiusGrad;
		int dim = r2 + 1;
		int xi, yi, i, j;	
		final double[] m = new double[r2];
		final double[] n = new double[r2];
		final double[] Gx = new double[r2];
		final double[] Gy = new double[r2];
		final double[] P = new double[r2*r2];
		final double[] gx2 = new double[r2*r2];
		final double[] gy2 = new double[r2*r2];
		final double[] gxy = new double[r2*r2];
		RandomAccess<T> ra = op.randomAccess();
		long[] min = new long[op.numDimensions()];
		for (i=0;i<op.numDimensions();i++)
			min[i]=op.min(i)+Math.max(0, op.dimension(i)/4-radiusGrad+1);
		double a1 = 0;
		double b1 = 0;
		double c1 = 0;
		double d1 = 0;
		double a2 = 0;
		final double b2;
		double c2 = 0;
		double d2 = 0;
		double a3 = 0;
		final double b3, c3, d3, e3, f3;
		double g3 = 0;	
		
		
		//define the coordinates of the gradient grid, set the center pixel as the original point
		for (i=0;i<r2;i++){
			m[i]=0.5+i-radiusGrad;
			n[i]=radiusGrad-0.5-i;
			Gx[i]=(x0-m[i])*e0;
			Gy[i]=y0+n[i];
		}
		
		//define the exact gradient at each position
		for (i=0;i<r2*r2;i++){
			xi = FastMath.floorMod(i, r2);
			yi = FastMath.floorDiv(i, r2);
			P[i]=FastMath.sqrt(Gx[xi]*Gx[xi]+Gy[yi]*Gy[yi]);
		}	
		
		// calculate the measured gradients
		int index=0;
		double valuex;
		double valuey;
		for (long y=dim-radiusGrad-1;y<dim+radiusGrad-1;y++)
			for (long x=dim-radiusGrad-1;x<dim+radiusGrad-1;x++){
				valuex=0;valuey=0;
				ra.setPosition(new long[]{3+x+min[0], (y+min[1])});
				valuex += ra.get().copy().getRealDouble();
				valuey += ra.get().copy().getRealDouble();
				ra.setPosition(new long[]{3+x+min[0], (1+y+min[1])});
				valuex += 2*ra.get().copy().getRealDouble();
				ra.setPosition(new long[]{3+x+min[0], (2+y+min[1])});
				valuex += 2*ra.get().copy().getRealDouble();
				ra.setPosition(new long[]{3+x+min[0], (3+y+min[1])});
				valuex += ra.get().copy().getRealDouble();
				valuey -= ra.get().copy().getRealDouble();
				ra.setPosition(new long[]{x+min[0], (y+min[1])});
				valuex -= ra.get().copy().getRealDouble();
				valuey += ra.get().copy().getRealDouble();
				ra.setPosition(new long[]{x+min[0], (1+y+min[1])});
				valuex -= 2*ra.get().copy().getRealDouble();
				ra.setPosition(new long[]{x+min[0], (2+y+min[1])});
				valuex -= 2*ra.get().copy().getRealDouble();
				ra.setPosition(new long[]{x+min[0], (3+y+min[1])});
				valuex -= ra.get().copy().getRealDouble();
				valuey -= ra.get().copy().getRealDouble();
				ra.setPosition(new long[]{1+x+min[0], (y+min[1])});
				valuey += 2*ra.get().copy().getRealDouble();
				ra.setPosition(new long[]{2+x+min[0], (y+min[1])});
				valuey += 2*ra.get().copy().getRealDouble();
				ra.setPosition(new long[]{1+x+min[0], (3+y+min[1])});
				valuey -= 2*ra.get().copy().getRealDouble();
				ra.setPosition(new long[]{2+x+min[0], (3+y+min[1])});
				valuey -= 2*ra.get().copy().getRealDouble();
				gx2[index]=valuex*valuex;
				gy2[index]=valuey*valuey;
				gxy[index++]=valuex*valuey;
		}
		
		final double[] a1s = new double[r2];
		final double[] b1s = new double[r2];
		final double[] c1s = new double[r2];
		final double[] d1s = new double[r2];
		final double[] a2s = new double[r2];
		final double[] c2s = new double[r2];
		final double[] d2s = new double[r2];
		final double[] a3s = new double[r2];
		final double[] g3s = new double[r2];
		
		//solve the equation to get the best fit [x,y,e] 
		for (int k=0;k<r2;k++)
			for (j=0;j<r2;j++){
				i=k*r2+j;
				a1s[j]+=gy2[i]*m[j]/P[i];
				b1s[j]+=gy2[i]/P[i];
				c1s[j]+=gxy[i]/P[i];
				d1s[j]+=gxy[i]*n[k]/P[i];
				
				a2s[j]+=gxy[i]*m[j]/P[i];
				c2s[j]+=gx2[i]/P[i];
				d2s[j]+=gx2[i]*n[k]/P[i];
				
				a3s[j]+=gy2[i]*m[j]*m[j]/P[i];
				g3s[j]+=gxy[i]*m[j]*n[k]/P[i];
			}
		
		for (j=0;j<r2;j++){
			a1 += a1s[j];
			b1 += b1s[j];
			c1 += c1s[j];
			d1 += d1s[j];
			
			a2 += a2s[j];
			c2 += c2s[j];
			d2 += d2s[j];
			
			a3 += a3s[j];
			g3 += g3s[j];
		}
		
		b1 = -b1;
		d1 = -d1;
		b2 = -c1;
		d2 = -d2;
		b3 = -2*a1;
		c3 = -b1;
		d3 = -d1;
		e3 = -c1;
		f3 = a2;
		g3 = -g3;
		
		final double A1 = (a2*c1-a1*c2)/(b1*c2-b2*c1);
		final double B1 = (c1*d2-c2*d1)/(b1*c2-b2*c1);

		final double A2 = (a1*(b2*c1-b1*c2)+b1*(a1*c2-a2*c1))/(c1*(b1*c2-b2*c1));
		final double B2 = (b1*(c2*d1-c1*d2)+d1*(b2*c1-b1*c2))/(c1*(b1*c2-b2*c1));

		final double A = a3+A1*b3+A1*A1*c3+A1*A2*e3+A2*f3;
		final double B = B1*B1*c3+B1*d3+B1*B2*e3;
		final double C = B1*b3+2*A1*B1*c3+A1*d3+A1*B2*e3+B2*f3+g3;
		
		final double e = (-C+Math.sqrt(C*C-4*A*B))/(2*A);
		if (e==0) return null;
		final double cx= min[0]+radiusGrad+(A1+B1/e);//op.min(i)+(op.dimension(i)-1)/2-radiusGrad;
		final double cy= min[1]+radiusGrad-(A2*e+B2);
		
		if (cx<op.min(0) || cx>op.max(0) || cy<op.min(1) || cy>op.max(1))
			return null;
		final RealRandomAccess<T> interpolant = Views.interpolate(op, new LanczosInterpolatorFactory<T>()).realRandomAccess();
		interpolant.setPosition(new double[]{cx, cy});
		final double intensity = interpolant.get().getRealDouble();
		
		return new double[]{cx,cy,e,intensity};
	}	
}
