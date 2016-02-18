package org.lemming.gui;

import java.util.HashMap;
import java.util.Map;

import javax.swing.GroupLayout;
import javax.swing.GroupLayout.Alignment;
import javax.swing.JLabel;
import javax.swing.JSpinner;
import javax.swing.LayoutStyle.ComponentPlacement;

import org.lemming.math.Calibrator;
import org.lemming.tools.WaitForChangeListener;
import org.lemming.tools.WaitForKeyListener;

import javax.swing.SpinnerNumberModel;
import javax.swing.JButton;
import javax.swing.JFileChooser;

import java.awt.event.ActionListener;
import java.io.File;
import java.awt.event.ActionEvent;

import javax.swing.JTextField;
import javax.swing.SwingConstants;

public class FitterPanel extends ConfigurationPanel {

	/**
	 * 
	 */
	private static final long serialVersionUID = 3081886846323191618L;
	public static final String KEY_WINDOW_SIZE = "WINDOW_SIZE";
	public static final String KEY_CENTROID_THRESHOLD = "CENTROID_THRESHOLD";
	public static final String KEY_CALIBRATION_FILENAME = "CALIBRATION_FILENAME";
	public static final String KEY_CAMERA_FILENAME = "CAMERA_FILENAME";
	public static final String KEY_STEPSIZE = "STEPSIZE";
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
	private JLabel lblThreshold;
	private JTextField textFieldThreshold;

	public FitterPanel() {
		setBorder(null);
		
		JLabel lblWindowSize = new JLabel("Window Size");
		
		spinnerWindowSize = new JSpinner();
		spinnerWindowSize.addChangeListener(new WaitForChangeListener(500, new Runnable(){
			@Override
			public void run() {
				fireChanged();
			}
		}));
		spinnerWindowSize.setModel(new SpinnerNumberModel(new Integer(5), null, null, new Integer(1)));
		
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
		
		lblThreshold = new JLabel("Threshold");
		
		textFieldThreshold = new JTextField();
		textFieldThreshold.setHorizontalAlignment(SwingConstants.TRAILING);
		textFieldThreshold.setText("100");
		textFieldThreshold.addKeyListener(new WaitForKeyListener(1000, new Runnable(){
			@Override
			public void run() {
				fireChanged();
			}}));
		textFieldThreshold.setColumns(10);
		
		GroupLayout groupLayout = new GroupLayout(this);
		groupLayout.setHorizontalGroup(
			groupLayout.createParallelGroup(Alignment.LEADING)
				.addGroup(groupLayout.createSequentialGroup()
					.addContainerGap()
					.addGroup(groupLayout.createParallelGroup(Alignment.LEADING)
						.addGroup(groupLayout.createSequentialGroup()
							.addGroup(groupLayout.createParallelGroup(Alignment.LEADING)
								.addComponent(lblWindowSize)
								.addComponent(lblThreshold))
							.addGap(7)
							.addGroup(groupLayout.createParallelGroup(Alignment.LEADING, false)
								.addComponent(textFieldThreshold, 0, 0, Short.MAX_VALUE)
								.addComponent(spinnerWindowSize, GroupLayout.DEFAULT_SIZE, 62, Short.MAX_VALUE)))
						.addGroup(groupLayout.createSequentialGroup()
							.addComponent(btnCamera, GroupLayout.PREFERRED_SIZE, 90, GroupLayout.PREFERRED_SIZE)
							.addPreferredGap(ComponentPlacement.RELATED)
							.addComponent(lblCamera, GroupLayout.PREFERRED_SIZE, 200, GroupLayout.PREFERRED_SIZE))
						.addGroup(groupLayout.createSequentialGroup()
							.addComponent(btnCalibration, GroupLayout.PREFERRED_SIZE, 90, GroupLayout.PREFERRED_SIZE)
							.addPreferredGap(ComponentPlacement.RELATED)
							.addComponent(lblCalibration, GroupLayout.PREFERRED_SIZE, 200, GroupLayout.PREFERRED_SIZE))
						.addComponent(btnNewCalibration))
					.addContainerGap(148, Short.MAX_VALUE))
		);
		groupLayout.setVerticalGroup(
			groupLayout.createParallelGroup(Alignment.LEADING)
				.addGroup(groupLayout.createSequentialGroup()
					.addContainerGap()
					.addGroup(groupLayout.createParallelGroup(Alignment.BASELINE)
						.addComponent(lblWindowSize)
						.addComponent(spinnerWindowSize, GroupLayout.PREFERRED_SIZE, GroupLayout.DEFAULT_SIZE, GroupLayout.PREFERRED_SIZE))
					.addGap(9)
					.addGroup(groupLayout.createParallelGroup(Alignment.BASELINE)
						.addComponent(textFieldThreshold, GroupLayout.PREFERRED_SIZE, GroupLayout.DEFAULT_SIZE, GroupLayout.PREFERRED_SIZE)
						.addComponent(lblThreshold, GroupLayout.PREFERRED_SIZE, 28, GroupLayout.PREFERRED_SIZE))
					.addPreferredGap(ComponentPlacement.RELATED)
					.addGroup(groupLayout.createParallelGroup(Alignment.BASELINE)
						.addComponent(lblCamera, GroupLayout.PREFERRED_SIZE, 27, GroupLayout.PREFERRED_SIZE)
						.addComponent(btnCamera))
					.addPreferredGap(ComponentPlacement.RELATED)
					.addGroup(groupLayout.createParallelGroup(Alignment.BASELINE)
						.addComponent(btnCalibration)
						.addComponent(lblCalibration, GroupLayout.PREFERRED_SIZE, 27, GroupLayout.PREFERRED_SIZE))
					.addPreferredGap(ComponentPlacement.RELATED)
					.addComponent(btnNewCalibration)
					.addGap(124))
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
		revalidate();
		fireChanged();
	}

	@Override
	public void setSettings(Map<String, Object> settings) {
		spinnerWindowSize.setValue(settings.get(KEY_WINDOW_SIZE));
		camFile = (File) settings.get(KEY_CAMERA_FILENAME);
		lblCamera.setText(camFile.getName());
		textFieldThreshold.setText(String.valueOf(settings.get(KEY_CENTROID_THRESHOLD)));
		calibFile = (File) settings.get(KEY_CALIBRATION_FILENAME);
		lblCalibration.setText(calibFile.getName());
		revalidate();
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
