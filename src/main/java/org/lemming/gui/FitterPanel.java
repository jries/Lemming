package org.lemming.gui;

import ij.IJ;

import java.util.HashMap;
import java.util.Map;

import javax.swing.GroupLayout;
import javax.swing.GroupLayout.Alignment;
import javax.swing.JLabel;
import javax.swing.JSpinner;
import javax.swing.LayoutStyle.ComponentPlacement;
import javax.swing.SpinnerNumberModel;
import javax.swing.JButton;
import javax.swing.JFileChooser;

import java.awt.event.ActionListener;
import java.io.File;
import java.awt.event.ActionEvent;

public class FitterPanel extends ConfigurationPanel {

	/**
	 * 
	 */
	private static final long serialVersionUID = 3081886846323191618L;
	public static final String KEY_WINDOW_SIZE = "WINDOW_SIZE";
	public static final String KEY_QUEUE_SIZE = "QUEUE_SIZE";
	public static final String KEY_CALIBRATION_FILE = "CALIBRATION_FILE";
	public static final String KEY_CAMERA_FILE = "CAMERA_FILE";
	private JSpinner spinnerWindowSize;
	private JSpinner spinnerQueueSize;
	private JButton btnCamera;
	private JButton btnCalibration;
	private JLabel lblCamera;
	private JLabel lblCalibration;
	protected File calibFile;
	protected File camFile;

	public FitterPanel() {
		setBorder(null);
		
		JLabel lblWindowSize = new JLabel("Window Size");
		
		spinnerWindowSize = new JSpinner();
		spinnerWindowSize.setModel(new SpinnerNumberModel(new Integer(10), null, null, new Integer(1)));
		
		JLabel lblQueueSize = new JLabel("Queue Size");
		
		spinnerQueueSize = new JSpinner();
		spinnerQueueSize.setModel(new SpinnerNumberModel(new Integer(60), null, null, new Integer(1)));
		
		lblCalibration = new JLabel("File");
		lblCalibration.setAlignmentX(0.5f);
		
		btnCalibration = new JButton("Calib. File");
		btnCalibration.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				JFileChooser fc = new JFileChooser(System.getProperty("user.home")+"/ownCloud/storm");
		    	fc.setFileSelectionMode(JFileChooser.FILES_ONLY);
		    	fc.setDialogTitle("Import Calibration File");
		    	int returnVal = fc.showOpenDialog(null);
		    	 
		        if (returnVal != JFileChooser.APPROVE_OPTION)
		        	return;
		        calibFile = fc.getSelectedFile();
		        lblCalibration.setText(calibFile.getName());
			}
		});
		
		lblCamera = new JLabel("File");
		
		btnCamera = new JButton("Cam File");
		btnCamera.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				
					JFileChooser fc = new JFileChooser(System.getProperty("user.home")+"/ownCloud/storm");
			    	fc.setFileSelectionMode(JFileChooser.FILES_ONLY);
			    	fc.setDialogTitle("Import Camera File");
			    	int returnVal = fc.showOpenDialog(null);
			    	 
			        if (returnVal != JFileChooser.APPROVE_OPTION)
			        	return;
			        camFile = fc.getSelectedFile();
			        lblCamera.setText(camFile.getName());
			}
		});
		
		GroupLayout groupLayout = new GroupLayout(this);
		groupLayout.setHorizontalGroup(
			groupLayout.createParallelGroup(Alignment.LEADING)
				.addGroup(groupLayout.createSequentialGroup()
					.addContainerGap()
					.addGroup(groupLayout.createParallelGroup(Alignment.LEADING)
						.addGroup(groupLayout.createSequentialGroup()
							.addGroup(groupLayout.createParallelGroup(Alignment.LEADING)
								.addComponent(lblWindowSize)
								.addComponent(lblQueueSize))
							.addPreferredGap(ComponentPlacement.RELATED)
							.addGroup(groupLayout.createParallelGroup(Alignment.TRAILING, false)
								.addComponent(spinnerQueueSize)
								.addComponent(spinnerWindowSize, GroupLayout.DEFAULT_SIZE, 62, Short.MAX_VALUE)))
						.addGroup(groupLayout.createSequentialGroup()
							.addComponent(btnCamera, GroupLayout.PREFERRED_SIZE, 90, GroupLayout.PREFERRED_SIZE)
							.addGap(12)
							.addComponent(lblCamera, GroupLayout.PREFERRED_SIZE, 200, GroupLayout.PREFERRED_SIZE))
						.addGroup(groupLayout.createSequentialGroup()
							.addComponent(btnCalibration, GroupLayout.PREFERRED_SIZE, 90, GroupLayout.PREFERRED_SIZE)
							.addGap(12)
							.addComponent(lblCalibration, GroupLayout.PREFERRED_SIZE, 200, GroupLayout.PREFERRED_SIZE)))
					.addContainerGap(136, Short.MAX_VALUE))
		);
		groupLayout.setVerticalGroup(
			groupLayout.createParallelGroup(Alignment.LEADING)
				.addGroup(groupLayout.createSequentialGroup()
					.addContainerGap()
					.addGroup(groupLayout.createParallelGroup(Alignment.BASELINE)
						.addComponent(lblWindowSize)
						.addComponent(spinnerWindowSize, GroupLayout.PREFERRED_SIZE, GroupLayout.DEFAULT_SIZE, GroupLayout.PREFERRED_SIZE))
					.addGap(17)
					.addGroup(groupLayout.createParallelGroup(Alignment.BASELINE)
						.addComponent(lblQueueSize)
						.addComponent(spinnerQueueSize, GroupLayout.PREFERRED_SIZE, GroupLayout.DEFAULT_SIZE, GroupLayout.PREFERRED_SIZE))
					.addPreferredGap(ComponentPlacement.UNRELATED)
					.addGap(18)
					.addGroup(groupLayout.createParallelGroup(Alignment.LEADING)
						.addComponent(btnCamera)
						.addComponent(lblCamera, GroupLayout.PREFERRED_SIZE, 27, GroupLayout.PREFERRED_SIZE))
					.addGap(6)
					.addGroup(groupLayout.createParallelGroup(Alignment.LEADING)
						.addComponent(btnCalibration)
						.addComponent(lblCalibration, GroupLayout.PREFERRED_SIZE, 27, GroupLayout.PREFERRED_SIZE))
					.addContainerGap(153, Short.MAX_VALUE))
		);
		setLayout(groupLayout);
	}

	@Override
	public void setSettings(Map<String, Object> settings) {
		spinnerWindowSize.setValue(settings.get(KEY_WINDOW_SIZE));
		spinnerQueueSize.setValue(settings.get(KEY_QUEUE_SIZE));
		camFile = (File) settings.get(KEY_CAMERA_FILE);
		lblCamera.setText(camFile.getName());
		calibFile = (File) settings.get(KEY_CALIBRATION_FILE);
		lblCalibration.setText(calibFile.getName());
	}

	@Override
	public Map<String, Object> getSettings() {
		final Map< String, Object > settings = new HashMap<>( 4 );
		settings.put(KEY_WINDOW_SIZE, spinnerWindowSize.getValue());
		settings.put(KEY_QUEUE_SIZE, spinnerQueueSize.getValue());
		if (calibFile == null){
			IJ.error("Please provide a Calibration File!");
			return settings;
		}
		if (camFile == null){
			IJ.error("Please provide a Camera File!");
			return settings;
		}
		settings.put(KEY_CALIBRATION_FILE, calibFile.getAbsolutePath());
		settings.put(KEY_CAMERA_FILE, camFile.getAbsolutePath());
		return settings;
	}
}
