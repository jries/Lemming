package org.lemming.gui;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.Map;

import javax.swing.JMenuItem;
import javax.swing.JPopupMenu;
import javax.swing.GroupLayout;
import javax.swing.GroupLayout.Alignment;
import javax.swing.JLabel;

public class GaussRendererPanel extends ConfigurationPanel {

	/**
	 * 
	 */
	private static final long serialVersionUID = 3986312897446569301L;
	protected Map<String, Object> settings;
	private RendererSettingsPanel dlg;

	public GaussRendererPanel() {
		setBorder(null);
		
		JLabel lblRightClickFor = new JLabel("Right click for settings");
		GroupLayout groupLayout = new GroupLayout(this);
		groupLayout.setHorizontalGroup(
			groupLayout.createParallelGroup(Alignment.LEADING)
				.addGroup(groupLayout.createSequentialGroup()
					.addContainerGap()
					.addComponent(lblRightClickFor)
					.addContainerGap(368, Short.MAX_VALUE))
		);
		groupLayout.setVerticalGroup(
			groupLayout.createParallelGroup(Alignment.LEADING)
				.addGroup(groupLayout.createSequentialGroup()
					.addContainerGap()
					.addComponent(lblRightClickFor)
					.addContainerGap(273, Short.MAX_VALUE))
		);
		setLayout(groupLayout);
		dlg = new RendererSettingsPanel();
		JPopupMenu popup = new JPopupMenu();
		JMenuItem menuItem = new JMenuItem("Settings");
		menuItem.addActionListener(new ActionListener(){

			@Override
			public void actionPerformed(ActionEvent e) {
				dlg.setSettings();
				fireChanged();
			}
			
		});
		popup.add(menuItem);
		setComponentPopupMenu(popup);
	}

	@Override
	public void setSettings(Map<String, Object> settings) {

	}

	@Override
	public Map<String, Object> getSettings() {
		return dlg.getSettings();
	}
}
