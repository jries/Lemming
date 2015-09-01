package org.lemming.factories;

import org.lemming.interfaces.PluginInterface;
import org.lemming.pipeline.AbstractModule;

public interface RendererFactory extends PluginInterface {
	
	/*
	 *  @return  Renderer to process
	 */
	public AbstractModule getRenderer();
	

}
