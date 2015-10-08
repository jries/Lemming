package org.lemming.plugins;

import ij.ImagePlus;
import ij.process.ShortProcessor;

import java.util.Map;

import org.apache.commons.math3.util.FastMath;
import org.lemming.factories.RendererFactory;
import org.lemming.gui.ConfigurationPanel;
import org.lemming.gui.RendererPanel;
import org.lemming.interfaces.Element;
import org.lemming.modules.Renderer;
import org.lemming.pipeline.ElementMap;
import org.lemming.pipeline.Localization;
import org.scijava.plugin.Plugin;

public class HistogramRenderer extends Renderer {
	
	public static final String NAME = "Histogram Renderer";
	public static final String KEY = "HISTOGRAMRENDERER";
	public static final String INFO_TEXT = "<html>"
											+ "Histogram Renderer Plugin"
											+ "</html>";
	
	private int xBins;
	private int xmin;
	private int xmax;
	private int ymin;
	private int ymax;
	private volatile short[] values;
	private long start;
	private float xwidth;
	private float ywidth;
	private float x;
	private float y;
	private int index;
	private int xindex;
	private int yindex;

	public HistogramRenderer(){
		this(256,256,0,256,0,256);
	}

	public HistogramRenderer(final int xBins, final int yBins, final int xmin, final int xmax, final int ymin, final int ymax) {
		this.xBins = xBins;
		this.xmin = xmin;
		this.xmax = xmax;
		this.ymin = ymin;
		this.ymax = ymax;
		this.xwidth = (float)(xmax - xmin) / xBins;
    	this.ywidth = (float)(ymax - ymin) / yBins;
		ShortProcessor sp = new ShortProcessor(xBins, yBins);
		ip = new ImagePlus(title, sp);
		values = (short[]) sp.getPixels();
	}
	
	@Override
	protected void beforeRun() {
		start = System.currentTimeMillis();
	}

	@Override
	public Element process(Element data) {
		if(data == null) return null;
		if(data.isLast())
				cancel();
		if (data instanceof Localization){
			Localization loc = (Localization) data;
			x = (float) loc.getX();
			y = (float) loc.getY();
		}
		if (data instanceof ElementMap){
			ElementMap map = (ElementMap) data;
			try{
				x = map.get("x").floatValue();
				y = map.get("y").floatValue();
			} catch (NullPointerException ne) {}
		}
		
        if ( (x >= xmin) && (x <= xmax) && (y >= ymin) && (y <= ymax)) {
        	xindex = FastMath.round((x - xmin) / xwidth);
        	yindex = FastMath.round((y - ymin) / ywidth);
        	index = xindex+yindex*xBins;
        	if (index < values.length)
        		values[index]++;
        }		
		        
		return null;
	}
	
	@Override
	public void afterRun(){
		ip.updateImage();
		System.out.println("Rendering done in "
				+ (System.currentTimeMillis() - start) + "ms.");
		//while(ip.isVisible()) pause(10);
	}
	
	
	@Plugin( type = RendererFactory.class, visible = true )
	public static class Factory implements RendererFactory{

		private RendererPanel configPanel = new RendererPanel();
		private Map<String, Object> settings;

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
				return new HistogramRenderer(width.intValue(), height.intValue(), xmin, width.intValue(), ymin, height.intValue());
			return new HistogramRenderer(xBins, yBins, xmin, xmax, ymin, ymax);
		}

		@Override
		public ConfigurationPanel getConfigurationPanel() {
			return configPanel;
		}

		@Override
		public boolean setAndCheckSettings(Map<String, Object> settings) {
			this.settings = settings;
			return settings != null;
		}
		
	}
	
//	private static IndexColorModel getDefaultColorModel() {
//		byte[] r = new byte[256];
//		byte[] g = new byte[256];
//		byte[] b = new byte[256];
//		for(int i=0; i<256; i++) {
//			r[i]=(byte)i;
//			g[i]=(byte)i;
//			b[i]=(byte)i;
//		}
//		return new IndexColorModel(8, 256, r, g, b);
//	}

}
