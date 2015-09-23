package org.lemming.gui;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.HashMap;
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
	protected Map<String, Object> dlgSettings = null;
	private RangeSlider rangeSliderX;
	private RangeSlider rangeSliderY;
	private JSpinner spinnerXBins;
	private JSpinner spinnerYBins;

	public HistogramRendererPanel() {
		setBorder(null);
		
		rangeSliderX = new RangeSlider(0,100);
		rangeSliderX.addChangeListener(new ChangeListener() {
			public void stateChanged(ChangeEvent e) {
				fireChanged();
			}
		});
		rangeSliderX.setMinorTickSpacing(25);
		rangeSliderX.setMajorTickSpacing(100);
		rangeSliderX.setPaintTicks(true);
		rangeSliderX.setPaintLabels(true);
		
		JLabel lblX = new JLabel("X");
		
		JLabel lblY = new JLabel("Y");
		
		rangeSliderY = new RangeSlider(0,100);
		rangeSliderY.addChangeListener(new ChangeListener() {
			public void stateChanged(ChangeEvent e) {
				fireChanged();
			}
		});
		rangeSliderY.setMinorTickSpacing(25);
		rangeSliderY.setMajorTickSpacing(100);
		rangeSliderY.setPaintTicks(true);
		rangeSliderY.setPaintLabels(true);
		
		JLabel lblXBins = new JLabel("X Bins");
		
		JLabel lblYBins = new JLabel("Y Bins");
		
		spinnerXBins = new JSpinner();
		spinnerXBins.addChangeListener(new ChangeListener() {
			public void stateChanged(ChangeEvent e) {
				fireChanged();
			}
		});
		spinnerXBins.setModel(new SpinnerNumberModel(new Integer(100), null, null, new Integer(1)));
		
		spinnerYBins = new JSpinner();
		spinnerYBins.addChangeListener(new ChangeListener() {
			public void stateChanged(ChangeEvent e) {
				fireChanged();
			}
		});
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
				dlgSettings = dlg.getSettings();
				if (dlgSettings!=null){
					int width = (int) dlgSettings.get(RendererSettingsPanel.KEY_RENDERER_WIDTH);
					rangeSliderX.setMaximum(width);
					rangeSliderX.setMajorTickSpacing(width);
					rangeSliderX.setMinorTickSpacing(width /4);
					int height = (int) dlgSettings.get(RendererSettingsPanel.KEY_RENDERER_HEIGHT);
					rangeSliderY.setMaximum(height);
					rangeSliderY.setMajorTickSpacing(height);
					rangeSliderY.setMinorTickSpacing(height/4);
				}
			}				
			
		});
		popup.add(menuItem);
		setComponentPopupMenu(popup);
	}

	@Override
	public void setSettings(Map<String, Object> settings) {
		rangeSliderX.setValue((int) settings.get(KEY_xmin));
		if ((int) settings.get(KEY_xmax) > rangeSliderX.getMaximum())
			rangeSliderX.setMaximum((int) settings.get(KEY_xmax));
		rangeSliderX.setUpperValue((int) settings.get(KEY_xmax));
		rangeSliderY.setValue((int) settings.get(KEY_ymin));
		if ((int) settings.get(KEY_ymax) > rangeSliderY.getMaximum())
			rangeSliderY.setMaximum((int) settings.get(KEY_ymax));
		rangeSliderY.setUpperValue((int) settings.get(KEY_ymax));
		spinnerXBins.setValue(settings.get(KEY_xBins));
		spinnerYBins.setValue(settings.get(KEY_yBins));
		validate();
		repaint();
	}

	@Override
	public Map<String, Object> getSettings() {
		HashMap<String, Object> settings = new HashMap<>();
		settings.put(KEY_xmin, rangeSliderX.getValue());
		settings.put(KEY_xmax, rangeSliderX.getUpperValue());
		settings.put(KEY_ymin, rangeSliderY.getValue());
		settings.put(KEY_ymax, rangeSliderY.getUpperValue());
		settings.put(KEY_xBins, spinnerXBins.getValue());
		settings.put(KEY_yBins, spinnerYBins.getValue());
		if (dlgSettings!=null){
			for (String key : dlgSettings.keySet())
				settings.put(key,dlgSettings.get(key));
		}
		return settings;
	}
}
