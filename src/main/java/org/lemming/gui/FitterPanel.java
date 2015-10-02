package org.lemming.gui;

import java.util.HashMap;
import java.util.Map;

import javax.swing.GroupLayout;
import javax.swing.GroupLayout.Alignment;
import javax.swing.JLabel;
import javax.swing.JSpinner;
import javax.swing.LayoutStyle.ComponentPlacement;

import org.lemming.math.Calibrator;

import javax.swing.SpinnerNumberModel;
import javax.swing.JButton;
import javax.swing.JFileChooser;

import java.awt.event.ActionListener;
import java.io.File;
import java.awt.event.ActionEvent;
import javax.swing.event.ChangeListener;
import javax.swing.event.ChangeEvent;

public class FitterPanel extends ConfigurationPanel {

	/**
	 * 
	 */
	private static final long serialVersionUID = 3081886846323191618L;
	public static final String KEY_WINDOW_SIZE = "WINDOW_SIZE";
	public static final String KEY_QUEUE_SIZE = "QUEUE_SIZE";
	public static final String KEY_CALIBRATION_FILENAME = "CALIBRATION_FILENAME";
	public static final String KEY_CAMERA_FILENAME = "CAMERA_FILENAME";
	private JSpinner spinnerWindowSize;
	private JButton btnCamera;
	private JButton btnCalibration;
	private JLabel lblCamera;
	private JLabel lblCalibration;
	protected File calibFile;
	protected File camFile;
	protected boolean doubleClicked = false;
	protected int default_step = 10;
	protected Calibrator calibrator;
	private JButton btnNewCalibration;

	public FitterPanel() {
		setBorder(null);
		
		JLabel lblWindowSize = new JLabel("Window Size");
		
		spinnerWindowSize = new JSpinner();
		spinnerWindowSize.addChangeListener(new ChangeListener() {
			public void stateChanged(ChangeEvent arg0) {
				fireChanged();
			}
		});
		spinnerWindowSize.setModel(new SpinnerNumberModel(new Integer(10), null, null, new Integer(1)));
		
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
		        fireChanged();
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
		
		btnNewCalibration = new JButton("New Calibration");
		btnNewCalibration.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent arg0) {
				calibrate();
			}
		});
		
		GroupLayout groupLayout = new GroupLayout(this);
		groupLayout.setHorizontalGroup(
			groupLayout.createParallelGroup(Alignment.LEADING)
				.addGroup(groupLayout.createSequentialGroup()
					.addGroup(groupLayout.createParallelGroup(Alignment.LEADING)
						.addGroup(groupLayout.createSequentialGroup()
							.addContainerGap()
							.addComponent(lblWindowSize)
							.addGap(7)
							.addComponent(spinnerWindowSize, GroupLayout.PREFERRED_SIZE, 62, GroupLayout.PREFERRED_SIZE))
						.addGroup(groupLayout.createSequentialGroup()
							.addContainerGap()
							.addComponent(btnCamera, GroupLayout.PREFERRED_SIZE, 90, GroupLayout.PREFERRED_SIZE)
							.addPreferredGap(ComponentPlacement.RELATED)
							.addComponent(lblCamera, GroupLayout.PREFERRED_SIZE, 200, GroupLayout.PREFERRED_SIZE))
						.addGroup(groupLayout.createSequentialGroup()
							.addContainerGap()
							.addComponent(btnCalibration, GroupLayout.PREFERRED_SIZE, 90, GroupLayout.PREFERRED_SIZE)
							.addPreferredGap(ComponentPlacement.RELATED)
							.addComponent(lblCalibration, GroupLayout.PREFERRED_SIZE, 200, GroupLayout.PREFERRED_SIZE))
						.addGroup(groupLayout.createSequentialGroup()
							.addContainerGap()
							.addComponent(btnNewCalibration)))
					.addContainerGap(148, Short.MAX_VALUE))
		);
		groupLayout.setVerticalGroup(
			groupLayout.createParallelGroup(Alignment.LEADING)
				.addGroup(groupLayout.createSequentialGroup()
					.addContainerGap()
					.addGroup(groupLayout.createParallelGroup(Alignment.BASELINE)
						.addComponent(lblWindowSize)
						.addComponent(spinnerWindowSize, GroupLayout.PREFERRED_SIZE, GroupLayout.DEFAULT_SIZE, GroupLayout.PREFERRED_SIZE))
					.addPreferredGap(ComponentPlacement.RELATED)
					.addGroup(groupLayout.createParallelGroup(Alignment.BASELINE)
						.addComponent(btnCamera)
						.addComponent(lblCamera, GroupLayout.PREFERRED_SIZE, 27, GroupLayout.PREFERRED_SIZE))
					.addPreferredGap(ComponentPlacement.RELATED)
					.addGroup(groupLayout.createParallelGroup(Alignment.BASELINE)
						.addComponent(btnCalibration)
						.addComponent(lblCalibration, GroupLayout.PREFERRED_SIZE, 27, GroupLayout.PREFERRED_SIZE))
					.addPreferredGap(ComponentPlacement.RELATED)
					.addComponent(btnNewCalibration)
					.addContainerGap(161, Short.MAX_VALUE))
		);
		setLayout(groupLayout);
	}

	protected void calibrate() {
        CalibrationDialog dlg = new CalibrationDialog(getFocusCycleRootAncestor());
        dlg.setLocationRelativeTo(getFocusCycleRootAncestor());
        dlg.setVisible(true);
		Map<String, Object> settings = dlg.getSettings();
		dlg.dispose();
		calibFile = (File) settings.get(CalibrationDialog.KEY_CALIBRATION_FILE);
		if (calibFile!=null)
			lblCalibration.setText(calibFile.getName());
		validate();
		repaint();
		fireChanged();
	}

	@Override
	public void setSettings(Map<String, Object> settings) {
		spinnerWindowSize.setValue(settings.get(KEY_WINDOW_SIZE));
		camFile = (File) settings.get(KEY_CAMERA_FILENAME);
		lblCamera.setText(camFile.getName());
		calibFile = (File) settings.get(KEY_CALIBRATION_FILENAME);
		lblCalibration.setText(calibFile.getName());
		validate();
		repaint();
	}

	@Override
	public Map<String, Object> getSettings() {
		final Map< String, Object > settings = new HashMap<>( 4 );
		settings.put(KEY_WINDOW_SIZE, spinnerWindowSize.getValue());
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
