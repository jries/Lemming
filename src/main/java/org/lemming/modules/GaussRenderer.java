package org.lemming.modules;

import java.util.ArrayList;
import java.util.List;

import ij.ImagePlus;
import ij.process.FloatProcessor;
import net.imglib2.Point;

import org.lemming.pipeline.Element;
import org.lemming.pipeline.Localization;
import org.lemming.pipeline.SingleRunModule;

public class GaussRenderer extends SingleRunModule {

	private int width;
	private int height;
	protected String title = "LemMING!";
	private float[] pixels;
	private ImagePlus ip;
	private double area = 1000;
	private double background = 0;
	private double sigmaX = 2;
	private double theta = 0;
	private double sigmaY = 2;
	private double maxVal=-Float.MAX_VALUE;
	private int counter = 0;
	private long start;
	private FloatProcessor fp;

	public GaussRenderer() {
		this(256, 256);
	}
	
	public GaussRenderer(int width, int height, double area, double background, double sigmaX, double sigmaY, double theta){
		this.width = width;
		this.height = height;
		this.area = area;
		this.background = background;
		this.sigmaX = sigmaX;
		this.sigmaY = sigmaY;
		this.theta = theta;
		pixels = new float[width*height];
		ip = new ImagePlus(title, new FloatProcessor(width, height, pixels));
		ip.show();
	}
	
	public GaussRenderer(int width, int height) {
		this.width = width;
		this.height = height;
		pixels = new float[width*height];
		fp = new FloatProcessor(width, height, pixels);
		ip = new ImagePlus(title, fp );
		ip.show();
	}
	
	@Override
	protected void beforeRun() {
		start = System.currentTimeMillis();
	}

	@Override
	public Element process(Element data) {
		
		Localization loc = (Localization) data;
		if (loc==null) return null;
		if (loc.isLast()) {
			cancel();
		}
		
		counter++;
		
		double x = loc.getX();
		double y = loc.getY();
		List<Point> X = getWindowPixels((int)x, (int)y, width, height, sigmaX, sigmaY);
		if(X==null) return null;
		double[] Params = {background, x, y, area, theta, sigmaX, sigmaY};
		List<Float> fcn = gaussian2D(X, Params);
		for (int i=0; i<X.size();i++){
			int idx = X.get(i).getIntPosition(0) + X.get(i).getIntPosition(1) * width;
			pixels[idx] += fcn.get(i);
			if (pixels[idx] > maxVal)
				maxVal = pixels[idx]; 
			fp.setf(idx, pixels[idx]);				
		}
		
		if (counter%100==0){
			ip.setDisplayRange(0, maxVal);
        	ip.updateAndDraw();
		}
		return null;
	}
	
	@Override
	public void afterRun(){
		ip.updateAndDraw();
		System.out.println("Rendering done in "
				+ (System.currentTimeMillis() - start) + "ms.");
		while(ip.isVisible()) pause(10);
	}
	
	/** A 2-dimensional, elliptical, Gaussian function.
	 * 
	 * @param X is a list of (x,y) localizations e.g. [ [x0,y0], [x1,y1], ..., [xn, yn] ] 
	 * @param P has the values
	 * @return
	 * <ul>
	 * <li> P[0] = background signal
	 * <li> P[1] = xLocalization
	 * <li> P[2] = yLocalization
	 * <li> P[3] = area under the curve (the total intensity)
	 * <li> P[4] = rotation angle, in radians, between the fluorescing molecule and the image canvas
	 * <li> P[5] = sigma of the 2D Gaussian in the x-direction, sigmaX
	 * <li> P[6] = sigma of the 2D Gaussian in the y-direction, sigmaY
	 * </ul> */
	public static List<Float> gaussian2D(List<Point> X, double[] P){
		List<Float> fcn = new ArrayList<>(X.size());
		double t12 = Math.cos(P[4]);
		double t16 = Math.sin(P[4]);
		double t20 = Math.pow(1.0 / P[5], 2);
		double t27 = Math.pow(1.0 / P[6], 2);
		double t2 = P[3] / (2.0 * Math.PI * P[5] * P[6]);
		double dx, dy;
		for(Point i : X){
			dx = i.getIntPosition(0) - P[1];
			dy = i.getIntPosition(1) - P[2];
			fcn.add( (float) (P[0] + t2*Math.exp(-0.5*( Math.pow(dx*t12 - dy*t16, 2)*t20 + Math.pow(dx*t16 + dy*t12, 2)*t27 ) )) );	
		}
		return fcn;
	}
	
	/** Generates a list of pixel (x,y) values that surround a fluorophore.
	 *  The size of the window in the x dimension is 2*(5*<code>sigmaX</code>) and 
	 *  in the y dimension is 2*(5*<code>sigmaX</code>*<code>aspectRatio</code>), 
	 *  i.e., a window of 5 sigma. 
	 * 
	 * @param x0 - the x position of the centroid
	 * @param y0 - the y position of the centroid
	 * @param imageWidth - the width of the image
	 * @param imageHeight - the height of the image
	 * @param sigmaX - the sigma value, in the x-direction, for a 2D Gaussian distribution
	 * @param aspectRatio - the aspect ratio for a 2D Gaussian, i.e. sigmaY/sigmaX
	 * @return X - a list of pixels that surrounds (x0,y0) */
	private static List<Point> getWindowPixels(int x0, int y0, int imageWidth, int imageHeight, double sigmaX, double sigmaY){
		
		// Make sure that (x0, y0) is within the image
		if(x0 > imageWidth || y0 > imageHeight) {
			System.err.println(String.format("Warning, localization not within image. Got (%d,%d), image size is (%d,%d)", x0, y0, imageWidth, imageHeight));
			return null;
		}
		
		// Automatically select a window around the fluorophore based on the 
		// sigmax and sigmay (ie. the aspect ratio) values.
		// 5*sigma means that 99.99994% of the fluorescene from a simulated 
		// fluorophore (for a Gaussian PSF) is within the specified window.
		// Also, the window has to be at least 1 x 1 pixel
		int halfWindowX = Math.max(1, (int) Math.round(sigmaX*5.0)); 
		int halfWindowY = Math.max(1, (int) Math.round(sigmaY*5.0));
		
		// make sure that the window remains within the image
		int x1 = Math.max(0, x0 - halfWindowX);
		int y1 = Math.max(0, y0 - halfWindowY);
		int x2 = Math.min(imageWidth - 1, x0 + halfWindowX);
		int y2 = Math.min(imageHeight - 1, y0 + halfWindowY);
		
		// insert the (x,y) window pixel coordinates into the X array
		int size = (x2-x1+1)*(y2-y1+1);
		List<Point> X = new ArrayList<>(size);
		for(int x=x1, y=y1, i=0; i<size; i++){
			X.add(new Point(new int[]{x,y}));
			if (x==x2){
				x = x1;
				y++;
			} else {
				x++;
			}
		}
		return X;
	}

}
