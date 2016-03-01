package org.lemming.plugins;

import org.lemming.factories.ActionFactory;
import org.scijava.plugin.Plugin;

public class ActionPlugin implements Runnable {
	
	public static final String NAME = "Empty Action Plugin";

	public static final String KEY = "ACTIONPLUGIN";

	public static final String INFO_TEXT = "<html>"
			+ "Empty Action Plugin"
			+ "</html>";

	@Override
	public void run() {
		
	}

	@Plugin( type = ActionFactory.class, visible = true )
	public static class Factory implements ActionFactory{

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
		public Runnable create() {
			return new ActionPlugin();
		}
		
	}

}
