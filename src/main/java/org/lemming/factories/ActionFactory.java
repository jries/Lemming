package org.lemming.factories;

import org.lemming.interfaces.PluginInterface;

/**
 * Factory for manipulating localization data
 * 
 * @author Ronny Sczech
 *
 */
public interface ActionFactory extends PluginInterface {
	
	Runnable create();
}
