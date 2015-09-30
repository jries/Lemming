package org.lemming.factories;

import java.util.Map;
import org.lemming.gui.ConfigurationPanel;
import org.lemming.interfaces.PluginInterface;
import org.lemming.modules.ImageMath.operators;
import org.lemming.pipeline.AbstractModule;

public interface PreProcessingFactory extends PluginInterface {
	
	/**
	 * Check that the given settings map is suitable for target detector.
	 *
	 * @param settings 
	 * the map to test.
	 * @return <code>true</code> if the settings map is valid.
	 */
	public boolean setAndCheckSettings( final Map< String, Object > settings );
	
	/**
	 *  @return  Module to process
	 */
	public AbstractModule getModule();
	
	/**
	 * Returns a new GUI panel able to configure the settings suitable for this
	 * specific detector factory.
	 */
	public ConfigurationPanel getConfigurationPanel();
	
	/**
	 * Returns the operator (none, plus, minus, divide or multiply) for the math operation after calculation of the module
	 */
	public operators getOperator();
	
	/**
	 * Returns the number of frames used in the preview
	 */
	public int processingFrames();
	
}
