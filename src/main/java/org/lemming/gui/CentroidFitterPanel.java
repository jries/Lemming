package org.lemming.gui;

import java.util.HashMap;
import java.util.Map;

import javax.swing.GroupLayout;
import javax.swing.GroupLayout.Alignment;
import javax.swing.JLabel;
import javax.swing.JSpinner;
import javax.swing.LayoutStyle.ComponentPlacement;
import javax.swing.SpinnerNumberModel;

import java.io.File;

import javax.swing.JTextField;
import javax.swing.SwingConstants;

import org.lemming.tools.WaitForChangeListener;
import org.lemming.tools.WaitForKeyListener;

public class CentroidFitterPanel extends ConfigurationPanel {
	/**
	 * 
	 */
	private static final long serialVersionUID = 3081886846323191618L;
	public static final String KEY_WINDOW_SIZE = "WINDOW_SIZE";
	public static final String KEY_CALIBRATION_FILENAME = "CALIBRATION_FILENAME";
	public static final String KEY_CAMERA_FILENAME = "CAMERA_FILENAME";
	public static final String KEY_CENTROID_THRESHOLD = "CENTROID_THRESHOLD";
	private final JSpinner spinnerWindowSize;
	private File calibFile;
	private File camFile;
	private final JTextField textFieldThreshold;
	
	public CentroidFitterPanel() {
		setBorder(null);
		
		JLabel lblWindowSize = new JLabel("Window Size");
		
		spinnerWindowSize = new JSpinner();
		spinnerWindowSize.addChangeListener(new WaitForChangeListener(500, new Runnable() {
			@Override
			public void run() {
				fireChanged();
			}
		}));
		spinnerWindowSize.setModel(new SpinnerNumberModel(10, null, null, 1));

		JLabel lblCentroidThreshold = new JLabel("Centroid Threshold");
		textFieldThreshold = new JTextField();
		textFieldThreshold.addKeyListener(new WaitForKeyListener(500, new Runnable() {
			@Override
			public void run() {
				fireChanged();
			}
		}));
		textFieldThreshold.setHorizontalAlignment(SwingConstants.TRAILING);
		textFieldThreshold.setText("100");
		textFieldThreshold.setColumns(10);
		
		GroupLayout groupLayout = new GroupLayout(this);
		groupLayout.setHorizontalGroup(
			groupLayout.createParallelGroup(Alignment.LEADING)
				.addGroup(groupLayout.createSequentialGroup()
					.addContainerGap()
					.addGroup(groupLayout.createParallelGroup(Alignment.LEADING)
						.addComponent(lblWindowSize)
						.addComponent(lblCentroidThreshold))
					.addPreferredGap(ComponentPlacement.RELATED)
					.addGroup(groupLayout.createParallelGroup(Alignment.LEADING, false)
						.addComponent(textFieldThreshold, 0, 0, Short.MAX_VALUE)
						.addComponent(spinnerWindowSize, GroupLayout.DEFAULT_SIZE, 62, Short.MAX_VALUE))
					.addContainerGap(148, Short.MAX_VALUE))
		);
		groupLayout.setVerticalGroup(
			groupLayout.createParallelGroup(Alignment.LEADING)
				.addGroup(groupLayout.createSequentialGroup()
					.addContainerGap()
					.addGroup(groupLayout.createParallelGroup(Alignment.BASELINE)
						.addComponent(lblWindowSize)
						.addComponent(spinnerWindowSize, GroupLayout.PREFERRED_SIZE, GroupLayout.DEFAULT_SIZE, GroupLayout.PREFERRED_SIZE))
					.addGap(7)
					.addGroup(groupLayout.createParallelGroup(Alignment.BASELINE)
						.addComponent(lblCentroidThreshold, GroupLayout.PREFERRED_SIZE, 28, GroupLayout.PREFERRED_SIZE)
						.addComponent(textFieldThreshold, GroupLayout.PREFERRED_SIZE, GroupLayout.DEFAULT_SIZE, GroupLayout.PREFERRED_SIZE))
					.addContainerGap(231, Short.MAX_VALUE))
		);
		setLayout(groupLayout);
	}

	@Override
	public void setSettings(Map<String, Object> settings) {
		try{
			spinnerWindowSize.setValue(settings.get(KEY_WINDOW_SIZE));
			textFieldThreshold.setText(String.valueOf(settings.get(KEY_CENTROID_THRESHOLD)));
			calibFile = (File) settings.get(KEY_CALIBRATION_FILENAME);
			camFile = (File) settings.get(KEY_CAMERA_FILENAME);
		} catch (Exception e){e.printStackTrace();}
	}

	@Override
	public Map<String, Object> getSettings() {
		final Map< String, Object > settings = new HashMap<>( 4 );
		settings.put(KEY_WINDOW_SIZE, spinnerWindowSize.getValue());
		settings.put(KEY_CENTROID_THRESHOLD, Double.parseDouble(textFieldThreshold.getText()));
		if (calibFile == null){
			return settings;
		}
		settings.put(KEY_CALIBRATION_FILENAME, calibFile.getAbsolutePath());
		if (camFile == null){
			//IJ.error("Please provide a Camera File!");
			return settings;
		}
		settings.put(KEY_CAMERA_FILENAME, camFile.getAbsolutePath());
		return settings;
	}
}
