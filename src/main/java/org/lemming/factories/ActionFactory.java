package org.lemming.factories;

import org.lemming.interfaces.PluginInterface;


public interface ActionFactory extends PluginInterface {
	
	public Runnable create();
}
