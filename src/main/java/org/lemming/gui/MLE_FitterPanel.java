package org.lemming.gui;

import java.util.HashMap;
import java.util.Map;

import javax.swing.GroupLayout;
import javax.swing.GroupLayout.Alignment;
import javax.swing.JLabel;
import javax.swing.JSpinner;
import javax.swing.LayoutStyle.ComponentPlacement;

import org.lemming.tools.WaitForChangeListener;
import javax.swing.SpinnerNumberModel;
import javax.swing.JButton;
import javax.swing.JFileChooser;

import java.awt.event.ActionListener;
import java.io.File;
import java.awt.event.ActionEvent;

import java.awt.Font;

public class MLE_FitterPanel extends ConfigurationPanel {

	/**
	 * 
	 */
	private static final long serialVersionUID = 3081886846323191618L;
	public static final String KEY_WINDOW_SIZE = "WINDOW_SIZE";
	public static final String KEY_USABLE_PIXEL = "USABLE_PIXEL";
	private static final String KEY_CALIBRATION_FILENAME = "CALIBRATION_FILENAME";
	private final JSpinner spinnerWindowSize;
	private final JLabel lblCalibration;
	private File calibFile;

	public MLE_FitterPanel() {
		setBorder(null);
		
		JLabel lblWindowSize = new JLabel("Window Size");
		
		spinnerWindowSize = new JSpinner();
		spinnerWindowSize.setFont(new Font("Dialog", Font.PLAIN, 12));
		spinnerWindowSize.addChangeListener(new WaitForChangeListener(500, new Runnable() {
			@Override
			public void run() {
				fireChanged();
			}
		}));
		spinnerWindowSize.setModel(new SpinnerNumberModel(5, null, null, 1));
		
		lblCalibration = new JLabel("File");
		lblCalibration.setAlignmentX(0.5f);

		JButton btnCalibration = new JButton("Calib. File");
		btnCalibration.addActionListener(new ActionListener(){
			@Override
			public void actionPerformed(ActionEvent e){
            JFileChooser fc = new JFileChooser(System.getProperty("user.home")+"/ownCloud/storm");
            fc.setFileSelectionMode(JFileChooser.FILES_ONLY);
            fc.setDialogTitle("Import Calibration File");
            int returnVal = fc.showOpenDialog(null);

            if (returnVal != JFileChooser.APPROVE_OPTION)
                return;
            calibFile = fc.getSelectedFile();
            lblCalibration.setText(calibFile.getName());
            fireChanged();
			}});

		JButton btnNewCalibration = new JButton("New Calibration");
		btnNewCalibration.addActionListener(new ActionListener(){
			@Override
			public void actionPerformed(ActionEvent e){calibrate();}});
		
		GroupLayout groupLayout = new GroupLayout(this);
		groupLayout.setHorizontalGroup(
			groupLayout.createParallelGroup(Alignment.LEADING)
				.addGroup(groupLayout.createSequentialGroup()
					.addContainerGap()
					.addGroup(groupLayout.createParallelGroup(Alignment.LEADING)
						.addGroup(groupLayout.createSequentialGroup()
							.addComponent(lblWindowSize)
							.addGap(7)
							.addComponent(spinnerWindowSize, GroupLayout.PREFERRED_SIZE, 62, GroupLayout.PREFERRED_SIZE))
						.addGroup(groupLayout.createSequentialGroup()
							.addComponent(btnCalibration, GroupLayout.PREFERRED_SIZE, 90, GroupLayout.PREFERRED_SIZE)
							.addPreferredGap(ComponentPlacement.RELATED)
							.addComponent(lblCalibration, GroupLayout.PREFERRED_SIZE, 200, GroupLayout.PREFERRED_SIZE))
						.addComponent(btnNewCalibration))
					.addContainerGap(136, Short.MAX_VALUE))
		);
		groupLayout.setVerticalGroup(
			groupLayout.createParallelGroup(Alignment.LEADING)
				.addGroup(groupLayout.createSequentialGroup()
					.addContainerGap()
					.addGroup(groupLayout.createParallelGroup(Alignment.BASELINE)
						.addComponent(lblWindowSize)
						.addComponent(spinnerWindowSize, GroupLayout.PREFERRED_SIZE, GroupLayout.DEFAULT_SIZE, GroupLayout.PREFERRED_SIZE))
					.addGap(43)
					.addGroup(groupLayout.createParallelGroup(Alignment.BASELINE)
						.addComponent(btnCalibration)
						.addComponent(lblCalibration, GroupLayout.PREFERRED_SIZE, 27, GroupLayout.PREFERRED_SIZE))
					.addPreferredGap(ComponentPlacement.RELATED)
					.addComponent(btnNewCalibration)
					.addGap(170))
		);
		setLayout(groupLayout);
	}

	private void calibrate() {
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
		calibFile = (File) settings.get(KEY_CALIBRATION_FILENAME);
		lblCalibration.setText(calibFile.getName());
		revalidate();
	}

	@Override
	public Map<String, Object> getSettings() {
		final Map< String, Object > settings = new HashMap<>( 4 );
		settings.put(KEY_WINDOW_SIZE, spinnerWindowSize.getValue());
		if (calibFile == null){
			return settings;
		}
		settings.put(KEY_CALIBRATION_FILENAME, calibFile.getAbsolutePath());
		return settings;
	}
}
