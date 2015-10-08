package org.lemming.plugins;

import java.util.Map;
import java.util.concurrent.atomic.AtomicIntegerArray;

import ij.IJ;
import ij.ImagePlus;
import ij.process.ImageProcessor;
import ij.process.ShortProcessor;
import net.imglib2.Point;

import org.apache.commons.math3.util.FastMath;
import org.lemming.factories.RendererFactory;
import org.lemming.gui.ConfigurationPanel;
import org.lemming.gui.RendererPanel;
import org.lemming.interfaces.Element;
import org.lemming.modules.Renderer;
import org.lemming.pipeline.ElementMap;
import org.lemming.pipeline.FittedLocalization;
import org.lemming.pipeline.Localization;
import org.scijava.plugin.Plugin;

public class GaussRenderer extends Renderer {
	
	public static final String NAME = "GaussRenderer";
	public static final String KEY = "GAUSSRENDERER";
	public static final String INFO_TEXT = "<html>"
											+ "Gauss Renderer Plugin"
											+ "</html>";
	private int xBins;
	private int xmin;
	private int xmax;
	private int ymin;
	private int ymax;
	private AtomicIntegerArray sharedPixels;
	private short area = 10000;
	private double background = 0;
	private double theta = 0;
//	private double maxVal = -Float.MAX_VALUE;
//	private int counter = 0;
	private long start;
	private double x;
	private double y;
	private double sigmaX;
	private double sigmaY;
	private double xwidth;
	private double ywidth;
	private double xindex;
	private double yindex;
	private int yBins;
	private double[] Params;
	private short[] pixels;

	
	public GaussRenderer(final int xBins, final int yBins, final int xmin, final int xmax, final int ymin, final int ymax) {
		this.xBins = xBins;
		this.yBins = yBins;
		this.xmin = xmin;
		this.xmax = xmax;
		this.ymin = ymin;
		this.ymax = ymax;
		this.xwidth = (double)(xmax - xmin) / xBins;
    	this.ywidth = (double)(ymax - ymin) / yBins;
		ImageProcessor fp = new ShortProcessor(xBins, yBins);
		pixels = (short[]) fp.getPixels();
		sharedPixels = new AtomicIntegerArray(xBins*yBins);
		ip = new ImagePlus(title,fp);
	}
	
	@Override
	protected void beforeRun() {
		start = System.currentTimeMillis();
	}

	@Override
	public Element process(Element data) {
		if (data==null) return null;
		if (data.isLast()) {
			cancel();
		}
		if (data instanceof Localization){
			FittedLocalization loc = (FittedLocalization) data;
			x = loc.getX();
			y = loc.getY();
			sigmaX = loc.getsX();
			sigmaY = loc.getsY();
		}
		if (data instanceof ElementMap){
			ElementMap map = (ElementMap) data;
			try{
				x = map.get("x").doubleValue();
				y = map.get("y").doubleValue();
				sigmaX = map.get("sx").doubleValue();
				sigmaY = map.get("sy").doubleValue();
			} catch (NullPointerException ne) {}
		}
		
		 if ( (x >= xmin) && (x <= xmax) && (y >= ymin) && (y <= ymax)) {
	        	xindex = (x - xmin) / xwidth;
	        	yindex = (y - ymin) / ywidth;
		 }
		
//		counter++;
		
		final Point[] X = getWindowPixels(xindex, yindex, xBins, yBins, sigmaX, sigmaY);
		Params = new double[]{background, xindex, yindex, area, theta, sigmaX, sigmaY};
		final double[] fcn = gaussian2D(X, Params);
		for (int i=0; i<X.length;i++){
			int idx = X[i].getIntPosition(0) + X[i].getIntPosition(1) * xBins;
			sharedPixels.addAndGet(idx, (int) fcn[i]);
//			maxVal = FastMath.max(pixels[idx], maxVal); 
		}
		
//		if (counter%100==0){
//			ip.setDisplayRange(0, maxVal);
//			window.repaint();
//		}
		return null;
	}
	
	@Override
	public void afterRun(){
		for (int i = 0 ;i< pixels.length;i++)
			pixels[i]=(short) sharedPixels.get(i);
			
		ip.updateImage();
		System.out.println("Rendering done in "	+ (System.currentTimeMillis() - start) + "ms.");
	}
	
	/** A 2-dimensional, elliptical, Gaussian function.
	 * 
	 * @param x2 is a list of (x,y) localizations e.g. [ [x0,y0], [x1,y1], ..., [xn, yn] ] 
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
	public static double[] gaussian2D(Point[] x2, double[] P){
		final double fcn[] = new double[x2.length];
		final double t12 = FastMath.cos(P[4]);
		final double t16 = FastMath.sin(P[4]);
		final double t20 = FastMath.pow(1.0 / P[5], 2);
		final double t27 = FastMath.pow(1.0 / P[6], 2);
		final double t2 = P[3] / (2.0 * FastMath.PI * P[5] * P[6]);
		double dx, dy;
		int index=0;
		for(Point i : x2){
			dx = i.getIntPosition(0) - P[1];
			dy = i.getIntPosition(1) - P[2];
			fcn[index++]= P[0] + t2*FastMath.exp(-0.5*( FastMath.pow(dx*t12 - dy*t16, 2)*t20 + FastMath.pow(dx*t16 + dy*t12, 2)*t27 ) );	
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
	private static Point[] getWindowPixels(final double x0, final double y0, final int imageWidth, final int imageHeight, final double sigmaX, final double sigmaY){
		
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
		final double halfWindowX = FastMath.max(1, sigmaX*5.0); 
		final double halfWindowY = FastMath.max(1, sigmaY*5.0);
		
		// make sure that the window remains within the image
		final long x1 = FastMath.max(0, FastMath.round(x0 - halfWindowX));
		final long y1 = FastMath.max(0, FastMath.round(y0 - halfWindowY));
		final long x2 = FastMath.min(imageWidth - 1, FastMath.round(x0 + halfWindowX));
		final long y2 = FastMath.min(imageHeight - 1, FastMath.round(y0 + halfWindowY));
		
		// insert the (x,y) window pixel coordinates into the X array
		final int size = (int)((x2-x1+1)*(y2-y1+1));
		Point[] X = new Point[size]; 
		for(long x=x1, y=y1, i=0; i<size; i++){
			X[(int) i]=new Point(new long[]{x,y});
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
		private RendererPanel configPanel = new RendererPanel();

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
			return settings!= null;
		}

		@Override
		public Renderer getRenderer() {
			final int xBins = (int) settings.get(RendererPanel.KEY_xBins);
			final int yBins = (int) settings.get(RendererPanel.KEY_yBins);
			final int xmin = (int) settings.get(RendererPanel.KEY_xmin);
			final int xmax = (int) settings.get(RendererPanel.KEY_xmax);
			final int ymin = (int) settings.get(RendererPanel.KEY_ymin);
			final int ymax = (int) settings.get(RendererPanel.KEY_ymax);
			final Integer width = (Integer) settings.get(RendererFactory.KEY_RENDERER_WIDTH);
			final Integer height = (Integer) settings.get(RendererFactory.KEY_RENDERER_HEIGHT);
			if (width != null && height != null)
				return new GaussRenderer(width.intValue(), height.intValue(), xmin, width.intValue(), ymin, height.intValue());
			return new GaussRenderer(xBins, yBins, xmin, xmax, ymin, ymax);
		}

		@Override
		public ConfigurationPanel getConfigurationPanel() {
			return configPanel;
		}
		
		
	}
}
