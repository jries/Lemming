package org.lemming.plugins;

import ij.ImagePlus;
import ij.process.FloatProcessor;

import java.util.Map;

import org.lemming.factories.RendererFactory;
import org.lemming.gui.ConfigurationPanel;
import org.lemming.gui.HistogramRendererPanel;
import org.lemming.interfaces.Element;
import org.lemming.pipeline.AbstractModule;
import org.lemming.pipeline.Localization;
import org.lemming.pipeline.SingleRunModule;
import org.scijava.plugin.Plugin;

public class HistogramRenderer extends SingleRunModule {
	
	public static final String NAME = "Histogram Renderer";
	public static final String KEY = "HISTOGRAMRENDERER";
	public static final String INFO_TEXT = "<html>"
											+ "Histogram Renderer Plugin"
											+ "</html>";
	
	private int xBins;
	private int yBins;
	private double xmin;
	private double xmax;
	private double ymin;
	private double ymax;
	private float[] values;
	private ImagePlus ip;
	protected String title = "LemMING!"; // title of the image
	private long counter = 0;
	private long start;

	public HistogramRenderer(){
		this(256,256,0,256,0,256);
	}

	public HistogramRenderer(int xBins, int yBins, double xmin, double xmax, double ymin, double ymax) {
		this.xBins = xBins;
		this.yBins = yBins;
		this.xmin = xmin;
		this.xmax = xmax;
		this.ymin = ymin;
		this.ymax = ymax;
		values = new float[xBins*yBins];
		ip = new ImagePlus(title, new FloatProcessor(xBins, yBins,values));
		ip.setDisplayRange(0, 5);
		ip.show();		
	}
	
	@Override
	protected void beforeRun() {
		start = System.currentTimeMillis();
	}

	@Override
	public Element process(Element data) {
		Localization loc = (Localization) data;
		if(loc==null) return null;
		
		counter ++;
		
		if(loc.isLast())
			cancel();
		
		double x = loc.getX();
		double y = loc.getY();
        if ( (x >= xmin) && (x <= xmax) && (y >= ymin) && (y <= ymax)) {
        	double xwidth = (xmax - xmin) / xBins;
        	double ywidth = (ymax - ymin) / yBins;
        	long xindex = Math.round((x - xmin) / xwidth);
        	long yindex = Math.round((y - ymin) / ywidth);
        	values[(int) (xindex+yindex*xBins)]++;
        }		
		
        if (counter%100==0)
        	ip.updateAndDraw();
        
		return null;
	}
	
	@Override
	public void afterRun(){
		ip.updateAndDraw();
		System.out.println("Rendering done in "
				+ (System.currentTimeMillis() - start) + "ms.");
		while(ip.isVisible()) pause(10);
	}

	@Override
	public boolean check() {
		return inputs.size()==1;
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
		public AbstractModule getRenderer() {
			//Map<String, Object> settings = configPanel.getSettings();
			final int xBins = (int) settings.get(HistogramRendererPanel.KEY_xBins);
			final int yBins = (int) settings.get(HistogramRendererPanel.KEY_yBins);
			final double xmin = (double) settings.get(HistogramRendererPanel.KEY_xmin);
			final double xmax = (double) settings.get(HistogramRendererPanel.KEY_xmax);
			final double ymin = (double) settings.get(HistogramRendererPanel.KEY_ymin);
			final double ymax = (double) settings.get(HistogramRendererPanel.KEY_ymax);
			return new HistogramRenderer(xBins, yBins, xmin, xmax, ymin, ymax);
		}

		@Override
		public ConfigurationPanel getConfigurationPanel() {
			return configPanel;
		}

		@Override
		public boolean setAndCheckSettings(Map<String, Object> settings) {
			this.settings = settings;
			return true;
		}
		
	}

}
