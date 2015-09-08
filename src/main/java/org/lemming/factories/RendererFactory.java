package org.lemming.factories;

import java.util.Map;

import org.lemming.gui.ConfigurationPanel;
import org.lemming.interfaces.PluginInterface;
import org.lemming.pipeline.AbstractModule;

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
	public AbstractModule getRenderer();
	
	/**
	 * Returns a new GUI panel able to configure the settings suitable for this
	 * specific factory.
	 */
	public ConfigurationPanel getConfigurationPanel();
}
