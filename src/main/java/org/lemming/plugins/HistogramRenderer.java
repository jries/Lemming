package org.lemming.plugins;

import ij.process.ShortProcessor;

import java.awt.image.IndexColorModel;
import java.util.List;
import java.util.Map;

import org.lemming.factories.RendererFactory;
import org.lemming.gui.ConfigurationPanel;
import org.lemming.gui.HistogramRendererPanel;
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
	private double xmin;
	private double ymin;
	private long start;
	private double xwidth;
	private double ywidth;
	private double x;
	private double y;
	private int index;
	private long xindex;
	private long yindex;
	private volatile short[] values; // volatile keyword keeps the array on the heap available
	private double xmax;
	private double ymax;

	public HistogramRenderer(){
		this(256,256,0,256,0,256);
	}

	public HistogramRenderer(final int xBins, final int yBins, final double xmin, final double xmax, final double ymin, final double ymax) {
		this.xBins = xBins;
		this.xmin = xmin;
		this.ymin = ymin;
		this.xmax = xmax;
		this.ymax = ymax;
		this.xwidth = (xmax - xmin) / xBins;
    	this.ywidth = (ymax - ymin) / yBins;
    	if (Runtime.getRuntime().freeMemory()<(xBins*yBins*4)){ 
    		cancel(); return;
    	}
    	values = new short[xBins*yBins];
		ShortProcessor sp = new ShortProcessor(xBins, yBins,values,getDefaultColorModel());
		ip.setProcessor(sp);
		ip.updateAndRepaintWindow();
	}
	
	@Override
	protected void beforeRun() {
		start = System.currentTimeMillis();
	}

	@Override
	public Element processData(Element data) {
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
			} catch (NullPointerException ne) { return null;}
		}
		
		synchronized(this){
        if ( (x >= xmin) && (x <= xmax) && (y >= ymin) && (y <= ymax)) {
	    	xindex = Math.round((x - xmin) / xwidth);
	    	yindex = Math.round((y - ymin) / ywidth);
	    	index = (int) (xindex+yindex*xBins);
	    	if (index>=0 && index<values.length)
	    		values[index]++;
		
			} 
        }   
		return null;
	}
	
	@Override
	public void afterRun(){
		ip.getProcessor().setMinAndMax(0, 3);
		ip.updateAndDraw();
		System.out.println("Rendering done in "
				+ (System.currentTimeMillis() - start) + "ms.");
		//while(ip.isVisible()) pause(10);
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

	@Override
	public void preview(List<Element> previewList) {
		for(Element el : previewList)
			processData(el);
		ip.updateAndDraw();
	}
	
	
	@Plugin( type = RendererFactory.class, visible = true )
	public static class Factory implements RendererFactory{

		private HistogramRendererPanel configPanel = new HistogramRendererPanel();
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
			final int xBins = (int) settings.get(RendererFactory.KEY_xBins);
			final int yBins = (int) settings.get(RendererFactory.KEY_yBins);
			final double xmin = (double) settings.get(RendererFactory.KEY_xmin);
			final double xmax = (double) settings.get(RendererFactory.KEY_xmax);
			final double ymin = (double) settings.get(RendererFactory.KEY_ymin);
			final double ymax = (double) settings.get(RendererFactory.KEY_ymax);
			return new HistogramRenderer(xBins, yBins, xmin, xmax, ymin, ymax);
		}

		@Override
		public ConfigurationPanel getConfigurationPanel() {
			configPanel.setName(KEY);
			return configPanel;
		}

		@Override
		public boolean setAndCheckSettings(Map<String, Object> settings) {
			this.settings = settings;
			configPanel.setSettings(settings);
			return settings != null;
		}

		@Override
		public Map<String, Object> getInitialSettings() {
			return configPanel.getInitialSettings();
		}
		
	}
}
