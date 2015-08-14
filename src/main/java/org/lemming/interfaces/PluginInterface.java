package org.lemming.interfaces;

import org.scijava.plugin.SciJavaPlugin;

public interface PluginInterface extends SciJavaPlugin {
	
	/**
	 * Returns a html string containing a descriptive information about this
	 * module.
	 *
	 * @return a html string.
	 */
	public String getInfoText();


	/**
	 * Returns a unique identifier of this module.
	 *
	 * @return the action key, as a string.
	 */
	public String getKey();

	/**
	 * Returns the human-compliant name of this module.
	 *
	 * @return the name, as a String.
	 */
	public String getName();
}
