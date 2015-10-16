package org.lemming.factories;

import java.util.Map;

import org.lemming.gui.ConfigurationPanel;
import org.lemming.interfaces.PluginInterface;
import org.lemming.modules.Renderer;

public interface RendererFactory extends PluginInterface {
	
	/**
	 * Check that the given settings map is suitable for target detector.
	 *
	 * @param settings 
	 * the map to test.
	 * @return <code>true</code> if the settings map is valid.
	 */
	public boolean setAndCheckSettings( final Map< String, Object > settings );
	
	/**
	 *  @return  Renderer to process
	 */
	public Renderer getRenderer();
	
	/**
	 * Returns a new GUI panel able to configure the settings suitable for this
	 * specific factory.
	 */
	public ConfigurationPanel getConfigurationPanel();
	
	public static final String KEY_xBins = "xBins";
	public static final String KEY_yBins = "yBins";
	public static final String KEY_xmin = "xmin";
	public static final String KEY_xmax = "xmax";
	public static final String KEY_ymin = "ymin";
	public static final String KEY_ymax = "ymax";
	public static final Object KEY_numLocs = "numLocs";
	
}
