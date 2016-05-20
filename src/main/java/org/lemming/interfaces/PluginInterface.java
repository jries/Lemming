package org.lemming.interfaces;

import org.scijava.plugin.SciJavaPlugin;

/**
 * Interface for Plug-ins which are registered by the SciJava Framework
 * 
 * @author Ronny Sczech
 *
 */

public interface PluginInterface extends SciJavaPlugin {
	
	/**
	 * Returns a html string containing a descriptive information about this
	 * module.
	 *
	 * @return a html string.
	 */
	String getInfoText();


	/**
	 * Returns a unique identifier of this module.
	 *
	 * @return the action key, as a string.
	 */
	String getKey();

	/**
	 * Returns the human-compliant name of this module.
	 *
	 * @return the name, as a String.
	 */
	String getName();
}
