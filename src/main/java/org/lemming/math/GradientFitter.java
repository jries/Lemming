package org.lemming.math;

import org.apache.commons.math3.util.FastMath;

import net.imglib2.RandomAccess;
import net.imglib2.type.numeric.RealType;
import net.imglib2.view.IntervalView;

/**
 * Calculating centroids on a {@link #RandomAccessibleInterval}
 * 
 * @author Ronny Sczech
 *
 * @param <T>
 */
public class GradientFitter<T extends RealType<T>>  {
	
	private IntervalView<T> op;
	private double thresh;
	private int radiusGrad;

	public GradientFitter(IntervalView<T> op_, double threshold_, int radiusGrad_){
		op = op_;
		thresh = threshold_;
		radiusGrad=radiusGrad_;
		
	}
	
	public double[] fit(){
		CentroidFitterRA<T> cf = new CentroidFitterRA<>(op, thresh);
		double[] result = cf.fitXYE();
		double x0 = result[0];
		double y0 = result[1];
		double e0 = result[2];
		int r2 = 2 * radiusGrad;
		int dim = (int) ((op.dimension(0)-1)/2);
		
		double[] m = new double[r2];
		double[] n = new double[r2];
		double[] Gx = new double[r2];
		double[] Gy = new double[r2];
		double[] P = new double[r2*r2];
		double[] gx2 = new double[r2*r2];
		double[] gy2 = new double[r2*r2];
		double[] gxy = new double[r2*r2];
		RandomAccess<T> ra = op.randomAccess();
		long[] min = new long[op.numDimensions()];
		op.min(min);
		double a1 = 0, b1 = 0, c1 = 0, d1 = 0, a2 = 0, b2, c2 = 0, d2 = 0, a3 = 0, b3, c3, d3, e3, f3, g3 = 0;		
		
		//define the coordinates of the gradient grid, set the center pixel as the original point
		for (int i=0;i<r2;i++){
			m[i]=0.5*(i+1)-radiusGrad;
			n[i]=radiusGrad-0.5*(i+1);
			Gx[i]=(x0-m[i])*e0;
			Gy[i]=y0-n[i];
		}
		
		//define the exact gradient at each position
		for (int i=0;i<r2*r2;i++){
			int xi = FastMath.floorMod(i, r2);
			int yi = FastMath.floorDiv(i, r2);
			P[i]=FastMath.sqrt(Gx[xi]*Gx[xi]+Gy[yi]*Gy[yi]);
		}
		
		
		// calculate the measured gradients
		int index=0;
		double valuex=0;
		double valuey=0;
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
		
		//solve the equation to get the best fit [x,y,e] 
		for (int i=0;i<r2*r2;i++){
			a1+=gy2[i]*m[i]/P[i];
			b1+=gy2[i]/P[i];
			c1+=gxy[i]/P[i];
			d1+=gxy[i]*n[i]/P[i];
			
			a2+=gxy[i]*m[i]/P[i];
			c2+=gx2[i]/P[i];
			d2+=gx2[i]*n[i]/P[i];
			
			a3+=(gy2[i]*m[i])*(gy2[i]*m[i])/P[i];
			g3+=gxy[i]*m[i]*n[i]/P[i];
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
		
		double A1 = (a2*c1-a1*c2)/(b1*c2-b2*c1);
		double B1 = (c1*d2-c2*d1)/(b1*c2-b2*c1);

		double A2 = (a1*(b2*c1-b1*c2)+b1*(a1*c2-a2*c1))/(c1*(b1*c2-b2*c1));
		double B2 = (b1*(c2*d1-c1*d2)+d1*(b2*c1-b1*c2))/(c1*(b1*c2-b2*c1));

		double A = a3+A1*b3+A1*A1*c3+A1*A2*e3+A2*f3;
		double B = B1*B1*c3+B1*d3+B1*B2*e3;
		double C = B1*b3+2*A1*B1*c3+A1*d3+A1*B2*e3+B2*f3+g3;
		
		double e = (-C+FastMath.sqrt(C*C-4*A*B))/(2*A);
		double cx= A1+B1/e;
		double cy= A2*e+B2;
		
		return new double[]{cx,cy,e};
	}
		
}
