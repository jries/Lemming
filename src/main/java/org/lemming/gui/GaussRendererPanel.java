package org.lemming.gui;

import java.util.HashMap;
import java.util.Map;

import javax.swing.GroupLayout;
import javax.swing.GroupLayout.Alignment;
import javax.swing.JLabel;
import javax.swing.LayoutStyle.ComponentPlacement;
import javax.swing.JTextField;
import org.lemming.factories.RendererFactory;
import org.lemming.tools.WaitForKeyListener;

public class GaussRendererPanel extends ConfigurationPanel {

	/**
	 * 
	 */
	private static final long serialVersionUID = 3986312897446569301L;
	private JTextField textFieldWidth;
	private JTextField textFieldHeight;

	public GaussRendererPanel() {
		setBorder(null);
		
		JLabel lblWidth = new JLabel("width");
		
		textFieldWidth = new JTextField();
		textFieldWidth.addKeyListener(new WaitForKeyListener(1000, new Runnable(){
			@Override
			public void run() {
				fireChanged();
			}
		}));
		textFieldWidth.setText("100");
		textFieldWidth.setColumns(10);
		
		textFieldHeight = new JTextField();
		textFieldHeight.addKeyListener(new WaitForKeyListener(1000, new Runnable(){
			@Override
			public void run() {
				fireChanged();
			}
		}));
		textFieldHeight.setText("100");
		textFieldHeight.setColumns(10);
		
		JLabel lblHeight = new JLabel("height");
		GroupLayout groupLayout = new GroupLayout(this);
		groupLayout.setHorizontalGroup(
			groupLayout.createParallelGroup(Alignment.LEADING)
				.addGroup(groupLayout.createSequentialGroup()
					.addContainerGap()
					.addGroup(groupLayout.createParallelGroup(Alignment.LEADING)
						.addComponent(lblHeight, GroupLayout.DEFAULT_SIZE, 42, Short.MAX_VALUE)
						.addComponent(lblWidth, GroupLayout.DEFAULT_SIZE, 42, Short.MAX_VALUE))
					.addPreferredGap(ComponentPlacement.RELATED)
					.addGroup(groupLayout.createParallelGroup(Alignment.LEADING)
						.addComponent(textFieldWidth, GroupLayout.DEFAULT_SIZE, 64, Short.MAX_VALUE)
						.addComponent(textFieldHeight, GroupLayout.DEFAULT_SIZE, 64, Short.MAX_VALUE))
					.addGap(332))
		);
		groupLayout.setVerticalGroup(
			groupLayout.createParallelGroup(Alignment.LEADING)
				.addGroup(groupLayout.createSequentialGroup()
					.addGap(18)
					.addGroup(groupLayout.createParallelGroup(Alignment.BASELINE)
						.addComponent(lblWidth)
						.addComponent(textFieldWidth, GroupLayout.PREFERRED_SIZE, 28, GroupLayout.PREFERRED_SIZE))
					.addGap(18)
					.addGroup(groupLayout.createParallelGroup(Alignment.BASELINE)
						.addComponent(lblHeight)
						.addComponent(textFieldHeight, GroupLayout.PREFERRED_SIZE, GroupLayout.DEFAULT_SIZE, GroupLayout.PREFERRED_SIZE))
					.addGap(208))
		);
		setLayout(groupLayout);
		
	}

	@Override
	public void setSettings(Map<String, Object> settings) {

	}

	@Override
	public Map<String, Object> getSettings() {
		Map <String,Object> settings1 = new HashMap<>(2);
		settings1.put(RendererFactory.KEY_RENDERER_WIDTH, Integer.parseInt(textFieldWidth.getText()));
		settings1.put(RendererFactory.KEY_RENDERER_HEIGHT, Integer.parseInt(textFieldHeight.getText()));
		return settings1;
	}
}
