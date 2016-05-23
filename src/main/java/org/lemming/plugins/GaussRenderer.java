package org.lemming.plugins;

import java.util.Map;

import org.apache.commons.math3.special.Erf;
import org.apache.commons.math3.util.FastMath;
import org.lemming.factories.RendererFactory;
import org.lemming.gui.ConfigurationPanel;
import org.lemming.gui.GaussRendererPanel;
import org.lemming.interfaces.Element;
import org.lemming.modules.Renderer;
import org.lemming.pipeline.ElementMap;
import org.lemming.pipeline.LocalizationPrecision3D;
import org.scijava.plugin.Plugin;

import net.imglib2.RandomAccess;
import net.imglib2.type.numeric.real.FloatType;

public class GaussRenderer extends Renderer {
	
	private static final String NAME = "GaussRenderer";
	private static final String KEY = "GAUSSRENDERER";
	private static final String INFO_TEXT = "<html>"
											+ "Gauss Renderer Plugin"
											+ "</html>";
	private final double xmin;
	private final double xmax;
	private final double ymin;
	private final double ymax;
	private final double xwidth;
	private final double ywidth;
	private volatile RandomAccess<FloatType> pixels;
	private double[] template;
	private static final double sqrt2 = FastMath.sqrt(2);
	private static final int sizeGauss = 600;
	private static final double roiks = 2.5;
	private static final int maxKernel = 30;
	private final double sigmaTemplate = sizeGauss/(8*roiks);
	
	public GaussRenderer(final int xBins, final int yBins, final double xmin, final double xmax, final double ymin, final double ymax) {
		super(xBins, yBins);
		this.xmin = xmin;
		this.xmax = xmax;
		this.ymin = ymin;
		this.ymax = ymax;
		this.xwidth = (xmax - xmin) / xBins;
		this.ywidth = (ymax - ymin) / yBins;
    	if (Runtime.getRuntime().freeMemory()<(xBins*yBins*4)){ 
    		cancel(); return;
    	}
    	pixels = img.randomAccess();
	}
	
	@Override
	protected void beforeRun() {
		start = System.currentTimeMillis();
		template = createGaussTemplate();
	}

	@Override
	public Element processData(Element data) {
		final double x;
		final double y;
		final double sX;
		final double sY;
		if (data instanceof LocalizationPrecision3D){
			LocalizationPrecision3D loc = (LocalizationPrecision3D) data;
			x = loc.getX().doubleValue();
			y = loc.getY().doubleValue();
			sX = loc.getsX().doubleValue();
			sY = loc.getsY().doubleValue();
			
		} else if (data instanceof ElementMap){
			ElementMap map = (ElementMap) data;
			try{
				x = map.get("x").doubleValue();
				y = map.get("y").doubleValue();
				sX = map.get("sx").doubleValue();
				sY = map.get("sy").doubleValue();
			} catch (NullPointerException ne) {
				return null;
			}
		} else {
			return null;
		}
		if (data.isLast())
			cancel();
		
		 if ( (x >= xmin) && (x <= xmax) && (y >= ymin) && (y <= ymax)) {
        	final double xindex = (x - xmin) / xwidth;
        	final double yindex = (y - ymin) / ywidth;
        	doWork(xindex, yindex, sX / xwidth, sY / xwidth);
		}
		
		return null;
	}
	
	@Override
	public void afterRun(){
		System.out.println("Rendering done in "	+ (System.currentTimeMillis() - start) + "ms.");
	}
	
	private void doWork(double xpix, double ypix, double sigmaX_, double sigmaY_){
		final int w = 2 * sizeGauss + 1;
		final double sigmaX = Math.max(sigmaX_, sigmaTemplate/sizeGauss);
		final double sigmaY = Math.max(sigmaY_, sigmaTemplate/sizeGauss);
		final long dnx = (long) Math.min(Math.ceil(roiks*sigmaX), maxKernel) ;
		final long dny = (long) Math.min(Math.ceil(roiks*sigmaY), maxKernel);
	    final long xr = Math.round(xpix);
	    final long yr = Math.round(ypix);
	    final double dx = xpix-xr;
	    final double dy = ypix-yr;
	    final double intcorrectionx = Erf.erf((dnx+0.5)/sigmaX/sqrt2);
	    final double intcorrectiony = Erf.erf((dny+0.5)/sigmaY/sqrt2);
	    final double gaussnorm = 10/(2*Math.PI*sigmaX*sigmaY*intcorrectionx*intcorrectiony);
	    int t_idx;
		long xt,yt,xax,yax,yp,xp;
		
	    for(xax = -dnx; xax <= dnx; xax++){
	    	xt = Math.round((xax+dx)*sigmaTemplate/sigmaX)+sizeGauss;
	    	for(yax = -dny; yax <= dny; yax++){
	    		yt = Math.round((yax+dy)*sigmaTemplate/sigmaY)+sizeGauss;
	    		xp = xr+xax; 
	            yp = yr+yax;
	            if (xp>=0 && yp>=0 && xt>=0 && yt>=0 && xt<w && yt<w && xp<xBins && yp<yBins){
	            	pixels.setPosition(new long[]{xp, yp});
	            	t_idx = (int) (xt + yt * w);
	            	final FloatType value = new FloatType();
	            	value.setReal(template[t_idx] * gaussnorm);
	            	pixels.get().add(value);
	            }
	    	}
	    }
		
	}
	
	private double[] createGaussTemplate(){
		final int w = 2 * sizeGauss + 1;
		final double[] T = new double[w*w];
		int index;
		double value;
		final double factor = 0.5/Math.pow(sigmaTemplate, 2);
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

	
	@Plugin( type = RendererFactory.class )
	public static class Factory implements RendererFactory{

		private Map<String, Object> settings;
		private final GaussRendererPanel configPanel = new GaussRendererPanel();

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
}
