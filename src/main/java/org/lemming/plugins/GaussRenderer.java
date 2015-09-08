package org.lemming.plugins;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import ij.IJ;
import ij.ImagePlus;
import ij.process.FloatProcessor;
import net.imglib2.Point;

import org.lemming.factories.RendererFactory;
import org.lemming.gui.ConfigurationPanel;
import org.lemming.gui.GaussRendererPanel;
import org.lemming.interfaces.Element;
import org.lemming.pipeline.AbstractModule;
import org.lemming.pipeline.FittedLocalization;
import org.lemming.pipeline.SingleRunModule;
import org.scijava.plugin.Plugin;

public class GaussRenderer extends SingleRunModule {
	
	public static final String NAME = "GaussRenderer";
	public static final String KEY = "GAUSSRENDERER";
	public static final String INFO_TEXT = "<html>"
											+ "Gauss Renderer Plugin"
											+ "</html>";
	private int width;
	private int height;
	protected String title = "LemMING!";
	private float[] pixels;
	private ImagePlus ip;
	private double area = 1000;
	private double background = 0;
	private double theta = 0;
	private double maxVal = -Float.MAX_VALUE;
	private int counter = 0;
	private long start;
	private FloatProcessor fp;

	public GaussRenderer() {
		this(256, 256);
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
		
		FittedLocalization loc = (FittedLocalization) data;
		if (loc==null) return null;
		if (loc.isLast()) {
			cancel();
		}
		
		counter++;
		
		double x = loc.getX();
		double y = loc.getY();
		double sigmaX = loc.getsX();
		double sigmaY = loc.getsY();		
		
		List<Point> X = getWindowPixels((int)x, (int)y, width, height, sigmaX, sigmaY);
		if(X==null) return null;
		double[] Params = {background, x, y, area, theta, sigmaX, sigmaY};
		List<Float> fcn = gaussian2D(X, Params);
		for (int i=0; i<X.size();i++){
			int idx = X.get(i).getIntPosition(0) + X.get(i).getIntPosition(1) * width;
			pixels[idx] += fcn.get(i);
			maxVal = Math.max(pixels[idx], maxVal); 
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
		System.out.println("Rendering done in "	+ (System.currentTimeMillis() - start) + "ms.");
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
			IJ.error(String.format("Warning, localization not within image. Got (%d,%d), image size is (%d,%d)", x0, y0, imageWidth, imageHeight));
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

	@Override
	public boolean check() {
		return inputs.size()==1;
	}
	
	@Plugin( type = RendererFactory.class, visible = true )
	public static class Factory implements RendererFactory{

		private Map<String, Object> settings;
		private GaussRendererPanel configPanel = new GaussRendererPanel();

		@Override
		public String getInfoText() {
			return INFO_TEXT;
		}

		@Override
		public String getKey() {
			return KEY;
		}

		@Override
		public String getName() {
			return NAME;
		}
		@Override
		public boolean setAndCheckSettings(Map<String, Object> settings) {
			this.settings = settings;
			return true;
		}

		@Override
		public AbstractModule getRenderer() {
			int w = (int) settings.get("WIDTH");
			int h = (int) settings.get("HEIGHT");
			return new GaussRenderer(w,h);
		}

		@Override
		public ConfigurationPanel getConfigurationPanel() {
			return configPanel;
		}
		
	}
}
