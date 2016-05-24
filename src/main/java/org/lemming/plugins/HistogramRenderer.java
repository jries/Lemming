package org.lemming.plugins;

import java.util.Map;

import org.lemming.factories.RendererFactory;
import org.lemming.gui.ConfigurationPanel;
import org.lemming.gui.HistogramRendererPanel;
import org.lemming.interfaces.Element;
import org.lemming.modules.Renderer;
import org.lemming.pipeline.ElementMap;
import org.lemming.pipeline.LocalizationPrecision3D;
import org.scijava.plugin.Plugin;

import net.imglib2.RandomAccess;
import net.imglib2.type.numeric.real.FloatType;

public class HistogramRenderer extends Renderer {
	
	private static final String NAME = "Histogram Renderer";
	private static final String KEY = "HISTOGRAMRENDERER";
	private static final String INFO_TEXT = "<html>"
											+ "Histogram Renderer Plugin"
											+ "</html>";

	private final double xmin;
	private final double ymin;
	private volatile RandomAccess<FloatType> values; // volatile keyword keeps the array on the heap available
	private final double zmin;
	private final double zmax;
	
	private final double xwidth;
	private final double ywidth;

	public HistogramRenderer(){
		this(256,256,0,256,0,256,0,255);
	}

	public HistogramRenderer(final int xBins, final int yBins, final double xmin, final double xmax, 
			final double ymin, final double ymax, final double zmin, final double zmax) {
		super(xBins, yBins);
		this.xmin = xmin;
		this.ymin = ymin;
		this.zmin = zmin;
    	this.zmax = zmax;
    	xwidth = (xmax - xmin) / xBins;
    	ywidth = (ymax - ymin) / yBins;
    	
    	if (Runtime.getRuntime().freeMemory()<(xBins*yBins*4)){ 
    		cancel(); return;
    	}
    	values = img.randomAccess();
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
		final FloatType rz = new FloatType();
		rz.setReal((z - zmin) / (zmax - zmin));
		final long xindex = Math.round((x - xmin) / xwidth);
		final long yindex = Math.round((y - ymin) / ywidth);
		
        if ( (xindex>=0) && (yindex>=0) && (xindex < xBins) && (yindex < yBins)) {	
			values.setPosition(new long[]{xindex, yindex});
			values.get().add(rz);
        }   
		return null;
	}
	
	@Plugin( type = RendererFactory.class )
	public static class Factory implements RendererFactory{

		private final HistogramRendererPanel configPanel = new HistogramRendererPanel();
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
