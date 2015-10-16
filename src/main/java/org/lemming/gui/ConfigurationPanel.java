package org.lemming.gui;

import java.util.Map;

import javax.swing.JPanel;

public abstract class ConfigurationPanel extends JPanel {
	

	/**
	 * 
	 */
	private static final long serialVersionUID = 3160662804934210143L;
	
	public static final String propertyName = "CONFIG_PANEL";
	
	/**
	 * Echo the parameters of the given settings on this panel.  
	 */
	public abstract void setSettings(final Map<String, Object> settings);
	
	/**
	 * @return  a new settings map object with its values set
	 * by this panel.
	 */
	public abstract Map<String, Object> getSettings();
		
	protected void fireChanged() {
		firePropertyChange(propertyName, null, getSettings());
	}


}
