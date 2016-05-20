package org.lemming.factories;

import java.util.Map;

import net.imglib2.type.numeric.RealType;

import org.lemming.gui.ConfigurationPanel;
import org.lemming.interfaces.PluginInterface;
import org.lemming.modules.Fitter;

/**
 * Factory for Fitter
 * 
 * @author Ronny Sczech
 *
 */

public interface FitterFactory extends PluginInterface{

	/**
	 * Check that the given settings map is suitable for target detector.
	 *
	 * @param settings 
	 * the map to test.
	 * @return <code>true</code> if the settings map is valid.
	 */
	boolean setAndCheckSettings(final Map<String, Object> settings);
	
	/**
	 * @param <T>T</T> data type
	 * @return  Module to process
	 */
	<T extends RealType<T>> Fitter<T> getFitter();
	
	/**
	 * @return getConfigurationPanel Returns a new GUI panel able to configure the settings suitable for this
	 * specific detector factory.
	 */
	ConfigurationPanel getConfigurationPanel();
	
	/**
	 *  @return halfkernel size
	 */
	int getHalfKernel();
	
	/**
	 *  @return mark for GPU use
	 */
	boolean hasGPU();
}
