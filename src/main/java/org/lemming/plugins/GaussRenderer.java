package org.lemming.plugins;

import java.awt.image.IndexColorModel;
import java.util.List;
import java.util.Map;

import ij.process.ImageProcessor;
import ij.process.FloatProcessor;
import ij.process.ImageStatistics;

import org.apache.commons.math3.special.Erf;
import org.apache.commons.math3.util.FastMath;
import org.lemming.factories.RendererFactory;
import org.lemming.gui.ConfigurationPanel;
import org.lemming.gui.GaussRendererPanel;
import org.lemming.interfaces.Element;
import org.lemming.modules.Renderer;
import org.lemming.pipeline.ElementMap;
import org.lemming.pipeline.LocalizationPrecision3D;
import org.lemming.pipeline.Localization;
import org.scijava.plugin.Plugin;

public class GaussRenderer extends Renderer {
	
	public static final String NAME = "GaussRenderer";
	public static final String KEY = "GAUSSRENDERER";
	public static final String INFO_TEXT = "<html>"
											+ "Gauss Renderer Plugin"
											+ "</html>";
	private double xmin;
	private double xmax;
	private double ymin;
	private double ymax;
	private long start;
	private double xwidth;
	private double ywidth;
	private double xindex;
	private double yindex;
	private volatile float[] pixels;
	private double[] template;
	private static double sqrt2 = FastMath.sqrt(2);
	private static int sizeGauss = 600;
	private static double roiks = 2.5;
	private static int maxKernel = 30;
	private double sigmaTemplate = sizeGauss/(8*roiks);
	private int xbins;
	private int ybins;
	private double x = 0, y = 0, sX = 0, sY = 0;

	
	public GaussRenderer(final int xBins, final int yBins, final double xmin, final double xmax, final double ymin, final double ymax) {
		this.xmin = xmin;
		this.xmax = xmax;
		this.ymin = ymin;
		this.ymax = ymax;
		this.xwidth = (xmax - xmin) / xBins;
		this.ywidth = (ymax - ymin) / yBins;
		this.xbins = xBins;
    	this.ybins = yBins;
    	if (Runtime.getRuntime().freeMemory()<(xBins*yBins*4)){ 
    		cancel(); return;
    	}
    	pixels = new float[xBins*yBins];
		ImageProcessor fp = new FloatProcessor(xbins, ybins, pixels, getDefaultColorModel());
		ip.setProcessor(fp);
		ip.updateAndRepaintWindow();
	}
	
	@Override
	protected void beforeRun() {
		start = System.currentTimeMillis();
		template = createGaussTemplate();
	}

	@Override
	public Element processData(Element data) {
		if (data==null) return null;
		if (data.isLast()) {
			cancel();
		}
		
		if (data instanceof Localization){
			LocalizationPrecision3D loc = (LocalizationPrecision3D) data;
			try{
				x = loc.getX();
				y = loc.getY();
				sX = loc.getsX();
				sY = loc.getsY();
			} catch (NullPointerException ne) {return null;}
		}
		if (data instanceof ElementMap){
			ElementMap map = (ElementMap) data;
			try{
				x = map.get("x").doubleValue();
				y = map.get("y").doubleValue();
				sX = map.get("sx").doubleValue();
				sY = map.get("sy").doubleValue();
			} catch (NullPointerException ne) {return null;}
		}
		
		 if ( (x >= xmin) && (x <= xmax) && (y >= ymin) && (y <= ymax)) {
        	xindex = (x - xmin) / xwidth;
        	yindex = (y - ymin) / ywidth;
        	doWork(xindex, yindex, sX / xwidth, sY / xwidth);
		}
		
		return null;
	}
	
	@Override
	public void afterRun(){
		ImageStatistics stats = ip.getStatistics();
		ip.setDisplayRange(stats.histMin, stats.histMax);
		ip.updateAndDraw();
		System.out.println("Rendering done in "	+ (System.currentTimeMillis() - start) + "ms.");
	}
	
	private void doWork(final double xpix, final double ypix, double sigmaX_, double sigmaY_){
		int w = 2 * sizeGauss + 1;
		double sigmaX = Math.max(sigmaX_, sigmaTemplate/sizeGauss);
		double sigmaY = Math.max(sigmaY_, sigmaTemplate/sizeGauss);
		long dnx = (long) Math.ceil(roiks*sigmaX);
	    long dny = (long) Math.ceil(roiks*sigmaY);
	    dnx = Math.min(maxKernel, dnx);
	    dny = Math.min(maxKernel, dny);
	    long xr = StrictMath.round(xpix);
	    long yr = StrictMath.round(ypix);
	    double dx = xpix-xr;
	    double dy = ypix-yr;
	    double intcorrectionx = Erf.erf((dnx+0.5)/sigmaX/sqrt2);
	    double intcorrectiony = Erf.erf((dny+0.5)/sigmaY/sqrt2);
	    double gaussnorm = 10/(2*Math.PI*sigmaX*sigmaY*intcorrectionx*intcorrectiony);
	    int idx, t_idx;
		long xt,yt,xax,yax,yp,xp;
		
	    for(xax = -dnx; xax <= dnx; xax++){
	    	xt = StrictMath.round((xax+dx)*sigmaTemplate/sigmaX)+sizeGauss;
	    	for(yax = -dny; yax <= dny; yax++){
	    		yt = StrictMath.round((yax+dy)*sigmaTemplate/sigmaY)+sizeGauss;
	    		xp = xr+xax; 
	            yp = yr+yax;
	            if (xp>=0 && yp>=0 && xt>=0 && yt>=0 && xt<w && yt<w && xp<xbins && yp<ybins){
	            	idx = (int) (xp + yp * xbins);
	            	t_idx = (int) (xt + yt * w);
	            	double value = template[t_idx] * gaussnorm;
	            	pixels[idx] += value;
	            }
	    	}
	    }
		
	}
	
	private double[] createGaussTemplate(){
		int w = 2 * sizeGauss + 1;
		double[] T = new double[w*w];
		int index = 0;
		double value = 0;
		double factor = 0.5/Math.pow(sigmaTemplate, 2);
		for (int yg = -sizeGauss ; yg<=sizeGauss; yg++)
			for(int xg = -sizeGauss; xg <= sizeGauss; xg++){
				index = (xg+sizeGauss) + (yg+sizeGauss) * w;
				value = FastMath.exp(-(xg*xg+yg*yg)*factor);
				T[index] = value;
			}
		return T;
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
			configPanel.setSettings(settings);
			return settings!= null;
		}

		@Override
		public Renderer getRenderer() {
			final int xBins = (Integer) settings.get(RendererFactory.KEY_xBins);
			final int yBins = (Integer) settings.get(RendererFactory.KEY_yBins);
			final double xmin = (Double) settings.get(RendererFactory.KEY_xmin);
			final double xmax = (Double) settings.get(RendererFactory.KEY_xmax);
			final double ymin = (Double) settings.get(RendererFactory.KEY_ymin);
			final double ymax = (Double) settings.get(RendererFactory.KEY_ymax);
			return new GaussRenderer(xBins, yBins, xmin, xmax, ymin, ymax);
		}

		@Override
		public ConfigurationPanel getConfigurationPanel() {
			configPanel.setName(KEY);
			return configPanel;
		}

		@Override
		public Map<String, Object> getInitialSettings() {
			return configPanel.getInitialSettings();
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
