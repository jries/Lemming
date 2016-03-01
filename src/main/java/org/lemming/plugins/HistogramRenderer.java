package org.lemming.plugins;

import ij.process.ByteProcessor;

import java.util.Map;

import org.lemming.factories.RendererFactory;
import org.lemming.gui.ConfigurationPanel;
import org.lemming.gui.HistogramRendererPanel;
import org.lemming.interfaces.Element;
import org.lemming.modules.Renderer;
import org.lemming.pipeline.ElementMap;
import org.lemming.pipeline.LocalizationPrecision3D;
import org.lemming.tools.LemmingUtils;
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
	private double xwidth;
	private double ywidth;
	private volatile byte[] values; // volatile keyword keeps the array on the heap available
	private double xmax;
	private double ymax;
	private double zmin;
	private double zmax;


	public HistogramRenderer(){
		this(256,256,0,256,0,256,0,255);
	}

	public HistogramRenderer(final int xBins, final int yBins, final double xmin, final double xmax, 
			final double ymin, final double ymax, final double zmin, final double zmax) {
		this.xBins = xBins;
		this.xmin = xmin;
		this.ymin = ymin;
		this.xmax = xmax;
		this.ymax = ymax;
		this.xwidth = (xmax - xmin) / xBins;
    	this.ywidth = (ymax - ymin) / yBins;
    	this.zmin = zmin;
    	this.zmax = zmax;
    	if (Runtime.getRuntime().freeMemory()<(xBins*yBins*4)){ 
    		cancel(); return;
    	}
    	values = new byte[xBins * yBins];
		ByteProcessor sp = new ByteProcessor(xBins, yBins, values, LemmingUtils.Ice());
		ip.setProcessor(sp);
		ip.updateAndRepaintWindow();
	}

	@Override
	public Element processData(Element data) {
		final double x, y, z;
		if (data instanceof LocalizationPrecision3D) {
			LocalizationPrecision3D loc = (LocalizationPrecision3D) data;
			x = loc.getX().doubleValue();
			y = loc.getY().doubleValue();
			z = loc.getZ().doubleValue();
		} else if (data instanceof ElementMap) {
			ElementMap map = (ElementMap) data;
			try {
				x = map.get("x").doubleValue();
				y = map.get("y").doubleValue();
				z = map.get("z").doubleValue();
			} catch (NullPointerException ne) {
				return null;
			}
		} else {
			return null;
		}
		if (data.isLast())
			cancel();
		long rz = StrictMath.round((z - zmin) / (zmax - zmin) * 256) + 1;
		
        if ( (x >= xmin) && (x <= xmax) && (y >= ymin) && (y <= ymax)) {
        	final long xindex = StrictMath.round((x - xmin) / xwidth);
			final long yindex = StrictMath.round((y - ymin) / ywidth);
			final int index = (int) (xindex + yindex * xBins);
			if (index >= 0 && index < values.length) {
				if (values[index] > 0)
					values[index] = (byte) ((values[index] + rz + 1) / 2);
				else
					values[index] = (byte) rz;
			}
        }   
		return null;
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
			final double zmin = (double) settings.get(RendererFactory.KEY_zmin);
			final double zmax = (double) settings.get(RendererFactory.KEY_zmax);
			return new HistogramRenderer(xBins, yBins, xmin, xmax, ymin, ymax, zmin, zmax);
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
			return HistogramRendererPanel.getInitialSettings();
		}
		
	}
}
