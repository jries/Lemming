package org.lemming.plugins;

import org.lemming.factories.ActionFactory;
import org.scijava.plugin.Plugin;

public class ActionPlugin implements Runnable {
	
	private static final String NAME = "Empty Action Plugin";

	private static final String KEY = "ACTIONPLUGIN";

	private static final String INFO_TEXT = "<html>"
			+ "Empty Action Plugin"
			+ "</html>";

	@Override
	public void run() {
		
	}

	@Plugin( type = ActionFactory.class)
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
