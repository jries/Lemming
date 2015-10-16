package org.lemming.plugins;

import java.awt.image.IndexColorModel;
import java.util.List;
import java.util.Map;

import ij.process.ImageProcessor;
import ij.process.FloatProcessor;
import net.imglib2.Point;

import org.apache.commons.math3.special.Erf;
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
	private double xmin;
	private double xmax;
	private double ymin;
	private double ymax;
	private int area = 10000;
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
	private volatile float[] pixels;
	private static double sqrt2 = FastMath.sqrt(2);

	
	public GaussRenderer(final int xBins, final int yBins, final double xmin, final double xmax, final double ymin, final double ymax, final int numLocs) {
		this.xBins = xBins;
		this.yBins = yBins;
		this.xmin = xmin;
		this.xmax = xmax;
		this.ymin = ymin;
		this.ymax = ymax;
		this.xwidth = (xmax - xmin) / xBins;
    	this.ywidth = (ymax - ymin) / yBins;
    	this.area = numLocs;
    	if (Runtime.getRuntime().freeMemory()<(xBins*yBins*4)){ 
    		cancel(); return;
    	}
    	pixels = new float[xBins*yBins];
		ImageProcessor fp = new FloatProcessor(xBins, yBins, pixels, getDefaultColorModel());
		ip.setProcessor(fp);
		ip.updateAndRepaintWindow();
	}
	
	@Override
	protected void beforeRun() {
		start = System.currentTimeMillis();
	}

	@Override
	public Element processData(Element data) {
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
			} catch (NullPointerException ne) {return null;}
		}
		
		 if ( (x >= xmin) && (x <= xmax) && (y >= ymin) && (y <= ymax)) {
        	xindex = (x - xmin) / xwidth;
        	yindex = (y - ymin) / ywidth;
		 
			final Point[] X = getWindowPixels(x, y, xmin,xmax, ymin,ymax, xwidth, ywidth, sigmaX, sigmaY);
			Params = new double[]{background, xindex, yindex, area, theta, sigmaX, sigmaY};
			final double[] fcn = gaussian2D(X, Params);
			for (int i=0; i<X.length;i++){
				int idx = X[i].getIntPosition(0) + X[i].getIntPosition(1) * xBins;
				if(idx<pixels.length)
					pixels[idx] += (float)fcn[i];
	//			maxVal = FastMath.max(pixels[idx], maxVal); 
			}
		}
		
//		if (counter%100==0){
//			ip.setDisplayRange(0, maxVal);
//			window.repaint();
//		}
		return null;
	}
	
	@Override
	public void afterRun(){
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
		final double sigma = (P[5] + P[6]) / 2;
		final double dn = FastMath.ceil( sigma * 2.5);
		final double intcorrection = FastMath.pow(Erf.erf((dn+0.5)/sigma/sqrt2), 2);
		final double gaussnorm = P[3] / (2.0 * FastMath.PI * P[5] * P[6] * intcorrection);
		double dx, dy;
		int index=0;
		for(Point i : x2){
			dx = i.getIntPosition(0) - P[1];
			dy = i.getIntPosition(1) - P[2];
			fcn[index++]= P[0] + gaussnorm*FastMath.exp(-0.5*( FastMath.pow(dx*t12 - dy*t16, 2)*t20 + FastMath.pow(dx*t16 + dy*t12, 2)*t27 ) );	
		}
		return fcn;
	}
	
	/** Generates a list of pixel (x,y) values that surround a fluorophore.
	 *  The size of the window in the x dimension is 2*(5*<code>sigmaX</code>) and 
	 *  in the y dimension is 2*(5*<code>sigmaY</code>), 
	 *  i.e., a window of 5 sigma. 
	 * 
	 * @param x0 - the x position of the centroid
	 * @param y0 - the y position of the centroid
	 * @param imageWidth - the width of the image
	 * @param imageHeight - the height of the image
	 * @param sigmaX - the sigma value, in the x-direction, for a 2D Gaussian distribution
	 * @param sigmaY - the sigma value, in the y-direction, for a 2D Gaussian distribution
	 * @return X - a list of pixels that surrounds (x0,y0) */
	private static Point[] getWindowPixels(final double x0, final double y0, final double xmin, final double xmax, 
			final double ymin, final double ymax, final double xwidth, final double ywidth, double sigmaX, double sigmaY){
		
		// Automatically select a window around the fluorophore based on the 
		// sigmax and sigmay (ie. the aspect ratio) values.
		// 5*sigma means that 99.99994% of the fluorescence from a simulated 
		// fluorophore (for a Gaussian PSF) is within the specified window.
		final double halfWindowX = FastMath.min(xwidth, sigmaX*5.0); 
		final double halfWindowY = FastMath.min(ywidth, sigmaY*5.0);
		
		// make sure that the window remains within the ROI
		final long x1 = FastMath.round((FastMath.max(xmin, x0 - halfWindowX)-xmin)/xwidth);
		final long y1 = FastMath.round((FastMath.max(ymin, y0 - halfWindowY)-ymin)/ywidth);
		final long x2 = FastMath.round((FastMath.min(xmax, x0 + halfWindowX)-xmin)/xwidth);
		final long y2 = FastMath.round((FastMath.min(ymax, y0 + halfWindowY)-ymin)/ywidth);
		
		// insert the (x,y) window pixel coordinates into the X array
		final int size = (int) FastMath.abs((x2-x1+1)*(y2-y1+1));
		//final int size = (int)Math.abs(((x2-x1+1)*(y2-y1+1)));
		Point[] X = new Point[size]; 
		long x = x1;
		long y = y1;
		for(int i=0; i<size; i++){
			X[i]=new Point(new long[]{x,y});
			if (x>=x2){
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
	
	@Override
	public void preview(List<Element> previewList) {
		for(Element el : previewList)
			processData(el);
		ip.updateAndDraw();
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
			configPanel.setSettings(settings);
			return settings!= null;
		}

		@Override
		public Renderer getRenderer() {
			final int xBins = (int) settings.get(RendererFactory.KEY_xBins);
			final int yBins = (int) settings.get(RendererFactory.KEY_yBins);
			final double xmin = (double) settings.get(RendererFactory.KEY_xmin);
			final double xmax = (double) settings.get(RendererFactory.KEY_xmax);
			final double ymin = (double) settings.get(RendererFactory.KEY_ymin);
			final double ymax = (double) settings.get(RendererFactory.KEY_ymax);
			final int numLocs = (int) settings.get(RendererFactory.KEY_numLocs);
			return new GaussRenderer(xBins, yBins, xmin, xmax, ymin, ymax, numLocs);
		}

		@Override
		public ConfigurationPanel getConfigurationPanel() {
			configPanel.setName(KEY);
			return configPanel;
		}
		
	}
	
	private static IndexColorModel getDefaultColorModel() {
		byte[] r = new byte[256];
		byte[] g = new byte[256];
		byte[] b = new byte[256];
		for(int i=0; i<256; i++) {
			r[i]=(byte)i;
			g[i]=(byte)i;
			b[i]=(byte)i;
		}
		return new IndexColorModel(8, 256, r, g, b);
	}

}
