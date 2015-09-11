package org.lemming.gui;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.Map;
import javax.swing.GroupLayout;
import javax.swing.GroupLayout.Alignment;
import javax.swing.JLabel;
import javax.swing.JMenuItem;
import javax.swing.JPopupMenu;
import javax.swing.LayoutStyle.ComponentPlacement;
import javax.swing.JSpinner;
import javax.swing.SpinnerNumberModel;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

public class HistogramRendererPanel extends ConfigurationPanel {

	/**
	 * 
	 */
	private static final long serialVersionUID = -3031663211936690561L;
	public static final String KEY_xBins = "xBins";
	public static final String KEY_yBins = "yBins";
	public static final String KEY_xmin = "xmin";
	public static final String KEY_xmax = "xmax";
	public static final String KEY_ymin = "ymin";
	public static final String KEY_ymax = "ymax";
	protected Map<String, Object> settings = null;
	private RangeSlider rangeSliderX;
	private RangeSlider rangeSliderY;
	private JSpinner spinnerXBins;
	private JSpinner spinnerYBins;

	public HistogramRendererPanel() {
		setBorder(null);
		
		rangeSliderX = new RangeSlider();
		rangeSliderX.setValue(0);
		rangeSliderX.setUpperValue(100);
		rangeSliderX.setMinorTickSpacing(25);
		rangeSliderX.setMajorTickSpacing(100);
		rangeSliderX.setPaintTicks(true);
		rangeSliderX.setPaintLabels(true);
		rangeSliderX.addChangeListener(new ChangeListener(){

			@Override
			public void stateChanged(ChangeEvent arg0) {
				
			}});
		
		JLabel lblX = new JLabel("X");
		
		JLabel lblY = new JLabel("Y");
		
		rangeSliderY = new RangeSlider();
		rangeSliderY.setMinorTickSpacing(25);
		rangeSliderY.setMajorTickSpacing(100);
		rangeSliderY.setValue(0);
		rangeSliderY.setUpperValue(100);
		rangeSliderY.setPaintTicks(true);
		rangeSliderY.setPaintLabels(true);
		rangeSliderX.addChangeListener(new ChangeListener(){

			@Override
			public void stateChanged(ChangeEvent arg0) {
				
			}});
		
		JLabel lblXBins = new JLabel("X Bins");
		
		JLabel lblYBins = new JLabel("Y Bins");
		
		spinnerXBins = new JSpinner();
		spinnerXBins.setModel(new SpinnerNumberModel(new Integer(100), null, null, new Integer(1)));
		
		spinnerYBins = new JSpinner();
		spinnerYBins.setModel(new SpinnerNumberModel(new Integer(100), null, null, new Integer(1)));
		GroupLayout groupLayout = new GroupLayout(this);
		groupLayout.setHorizontalGroup(
			groupLayout.createParallelGroup(Alignment.LEADING)
				.addGroup(groupLayout.createSequentialGroup()
					.addContainerGap()
					.addGroup(groupLayout.createParallelGroup(Alignment.LEADING, false)
						.addGroup(groupLayout.createSequentialGroup()
							.addComponent(lblX)
							.addPreferredGap(ComponentPlacement.RELATED)
							.addComponent(rangeSliderX, GroupLayout.PREFERRED_SIZE, 243, GroupLayout.PREFERRED_SIZE))
						.addGroup(groupLayout.createSequentialGroup()
							.addComponent(lblY)
							.addPreferredGap(ComponentPlacement.RELATED)
							.addComponent(rangeSliderY, GroupLayout.DEFAULT_SIZE, GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
						.addGroup(groupLayout.createParallelGroup(Alignment.TRAILING, false)
							.addGroup(groupLayout.createSequentialGroup()
								.addComponent(lblYBins)
								.addPreferredGap(ComponentPlacement.RELATED)
								.addComponent(spinnerYBins))
							.addGroup(Alignment.LEADING, groupLayout.createSequentialGroup()
								.addComponent(lblXBins)
								.addPreferredGap(ComponentPlacement.RELATED)
								.addComponent(spinnerXBins, GroupLayout.PREFERRED_SIZE, 70, GroupLayout.PREFERRED_SIZE))))
					.addContainerGap(174, Short.MAX_VALUE))
		);
		groupLayout.setVerticalGroup(
			groupLayout.createParallelGroup(Alignment.LEADING)
				.addGroup(groupLayout.createSequentialGroup()
					.addContainerGap()
					.addGroup(groupLayout.createParallelGroup(Alignment.LEADING)
						.addComponent(lblX)
						.addComponent(rangeSliderX, GroupLayout.PREFERRED_SIZE, GroupLayout.DEFAULT_SIZE, GroupLayout.PREFERRED_SIZE))
					.addPreferredGap(ComponentPlacement.RELATED)
					.addGroup(groupLayout.createParallelGroup(Alignment.LEADING)
						.addComponent(lblY)
						.addComponent(rangeSliderY, GroupLayout.PREFERRED_SIZE, GroupLayout.DEFAULT_SIZE, GroupLayout.PREFERRED_SIZE))
					.addPreferredGap(ComponentPlacement.RELATED)
					.addGroup(groupLayout.createParallelGroup(Alignment.BASELINE)
						.addComponent(lblXBins)
						.addComponent(spinnerXBins, GroupLayout.PREFERRED_SIZE, GroupLayout.DEFAULT_SIZE, GroupLayout.PREFERRED_SIZE))
					.addPreferredGap(ComponentPlacement.RELATED)
					.addGroup(groupLayout.createParallelGroup(Alignment.BASELINE)
						.addComponent(spinnerYBins, GroupLayout.PREFERRED_SIZE, GroupLayout.DEFAULT_SIZE, GroupLayout.PREFERRED_SIZE)
						.addComponent(lblYBins))
					.addContainerGap(160, Short.MAX_VALUE))
		);
		setLayout(groupLayout);
		JPopupMenu popup = new JPopupMenu();
		JMenuItem menuItem = new JMenuItem("Settings");
		menuItem.addActionListener(new ActionListener(){
			@Override
			public void actionPerformed(ActionEvent e) {
				RendererSettingsPanel dlg = new RendererSettingsPanel();
				settings = dlg.getSettings();
				if (settings!=null){
					//rangeSliderX.setLabelTable(null);
					rangeSliderX.setMaximum((int) settings.get("WIDTH"));
					rangeSliderX.setMajorTickSpacing((int) settings.get("WIDTH"));
					rangeSliderX.setMinorTickSpacing((int) settings.get("WIDTH") /4);
					//rangeSliderY.setLabelTable(null);
					rangeSliderY.setMaximum((int) settings.get("HEIGHT"));
					rangeSliderY.setMajorTickSpacing((int) settings.get("HEIGHT"));
					rangeSliderY.setMinorTickSpacing((int) settings.get("HEIGHT")/4);
				}
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
		settings.put(KEY_xmin, rangeSliderX.getValue());
		settings.put(KEY_xmax, rangeSliderX.getUpperValue());
		settings.put(KEY_ymin, rangeSliderY.getValue());
		settings.put(KEY_ymax, rangeSliderY.getUpperValue());
		settings.put(KEY_xBins, spinnerXBins.getValue());
		settings.put(KEY_yBins, spinnerYBins.getValue());
		return settings;
	}
}
