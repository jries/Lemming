package org.lemming.gui;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import javax.swing.JDialog;
import javax.swing.JFileChooser;
import javax.swing.GroupLayout;
import javax.swing.GroupLayout.Alignment;
import javax.swing.JLabel;
import javax.swing.JSpinner;
import javax.swing.LayoutStyle.ComponentPlacement;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import javax.swing.filechooser.FileNameExtensionFilter;

import org.lemming.math.Calibrator;

import ij.IJ;
import ij.ImagePlus;
import ij.gui.Roi;
import ij.gui.StackWindow;
import ij.io.OpenDialog;
import ij.plugin.FolderOpener;

import javax.swing.SpinnerNumberModel;
import javax.swing.JButton;
import java.awt.event.ActionListener;
import java.io.File;
import java.awt.Container;
import java.awt.Dialog;
import java.awt.Frame;
import java.awt.event.ActionEvent;
import javax.swing.SwingConstants;

/**
 * a dialog for calibration parameters for use in 3D astigmatism fitter
 * 
 * @author Ronny Sczech
 *
 */
class CalibrationDialog extends JDialog {

	/**
	 * 
	 */
	private static final long serialVersionUID = 77072999967630231L;
	public static final String KEY_CALIBRATION_FILE = "CALIBRATION_FILE";
	private JLabel lblRange;
	private final JSpinner spinnerStepSize;
	private RangeSlider rangeSlider;
	private JButton btnFitCurve;
	private JButton btnSaveCalibration;
	private StackWindow calibWindow;
	private Calibrator calibrator;
	private File calibFile;

	public CalibrationDialog(Container container) {
		super((Frame) container);
		setModalityType(Dialog.ModalityType.DOCUMENT_MODAL);
		setDefaultCloseOperation(JDialog.DISPOSE_ON_CLOSE);
		setTitle("Fitter Calibration");

		JLabel lblStepSize = new JLabel("Step Size");
		
		spinnerStepSize = new JSpinner();
		spinnerStepSize.setModel(new SpinnerNumberModel(10, null, null, 1));

		JButton btnFitBeads = new JButton("Fit beads");
		btnFitBeads.addActionListener(new ActionListener(){
			@Override
			public void actionPerformed(ActionEvent e){
				fitbeads();
				lblRange.setEnabled(true);
				rangeSlider.setEnabled(true);
				btnFitCurve.setEnabled(true);
			}});
		
		lblRange = new JLabel("Range");
		lblRange.setEnabled(false);
		
		rangeSlider = new RangeSlider();
		rangeSlider.setEnabled(false);
		rangeSlider.setUpperValue(100);
		rangeSlider.setValue(0);
		rangeSlider.setMajorTickSpacing(200);
		rangeSlider.setMinorTickSpacing(50);
		rangeSlider.setMaximum(400);
		rangeSlider.setPaintLabels(true);
		rangeSlider.setPaintTicks(true);
		
		btnFitCurve = new JButton("Fit curve");
		btnFitCurve.addActionListener(new ActionListener(){
			@Override
			public void actionPerformed(ActionEvent e){
				fitCurve();
				btnSaveCalibration.setEnabled(true);}});
		btnFitCurve.setEnabled(false);
		
		btnSaveCalibration = new JButton("Save Calibration");
		btnSaveCalibration.addActionListener(new ActionListener(){
			@Override
			public void actionPerformed(ActionEvent e){saveCalibration();}});
		btnSaveCalibration.setEnabled(false);
		
		JLabel lblMinrange = new JLabel(" ");
		
		JLabel lblMaxrange = new JLabel(" ");
		lblMaxrange.setHorizontalAlignment(SwingConstants.TRAILING);
		GroupLayout groupLayout = new GroupLayout(getContentPane());
		groupLayout.setHorizontalGroup(
			groupLayout.createParallelGroup(Alignment.LEADING)
				.addGroup(groupLayout.createSequentialGroup()
					.addContainerGap()
					.addGroup(groupLayout.createParallelGroup(Alignment.LEADING, false)
						.addGroup(groupLayout.createSequentialGroup()
							.addComponent(lblStepSize)
							.addPreferredGap(ComponentPlacement.RELATED)
							.addComponent(spinnerStepSize, GroupLayout.PREFERRED_SIZE, 53, GroupLayout.PREFERRED_SIZE))
						.addGroup(groupLayout.createSequentialGroup()
							.addComponent(btnFitBeads)
							.addPreferredGap(ComponentPlacement.RELATED)
							.addComponent(lblRange))
						.addGroup(groupLayout.createSequentialGroup()
							.addComponent(btnFitCurve)
							.addGap(18)
							.addComponent(btnSaveCalibration))
						.addComponent(rangeSlider, GroupLayout.PREFERRED_SIZE, 255, GroupLayout.PREFERRED_SIZE)
						.addGroup(groupLayout.createSequentialGroup()
							.addComponent(lblMinrange)
							.addPreferredGap(ComponentPlacement.RELATED, GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
							.addComponent(lblMaxrange)
							.addGap(3)))
					.addContainerGap())
		);
		groupLayout.setVerticalGroup(
			groupLayout.createParallelGroup(Alignment.LEADING)
				.addGroup(groupLayout.createSequentialGroup()
					.addGroup(groupLayout.createParallelGroup(Alignment.BASELINE)
						.addComponent(lblStepSize)
						.addComponent(spinnerStepSize, GroupLayout.PREFERRED_SIZE, GroupLayout.DEFAULT_SIZE, GroupLayout.PREFERRED_SIZE))
					.addPreferredGap(ComponentPlacement.RELATED)
					.addGroup(groupLayout.createParallelGroup(Alignment.BASELINE)
						.addComponent(btnFitBeads)
						.addComponent(lblRange))
					.addPreferredGap(ComponentPlacement.RELATED)
					.addGroup(groupLayout.createParallelGroup(Alignment.BASELINE)
						.addComponent(lblMinrange)
						.addComponent(lblMaxrange))
					.addPreferredGap(ComponentPlacement.RELATED, GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
					.addComponent(rangeSlider, GroupLayout.PREFERRED_SIZE, GroupLayout.DEFAULT_SIZE, GroupLayout.PREFERRED_SIZE)
					.addPreferredGap(ComponentPlacement.RELATED)
					.addGroup(groupLayout.createParallelGroup(Alignment.BASELINE)
						.addComponent(btnFitCurve)
						.addComponent(btnSaveCalibration))
					.addGap(34))
		);
		rangeSlider.addChangeListener(new ChangeListener() {
			public void stateChanged(ChangeEvent arg0) {
				if(lblMinrange == null || lblMaxrange == null) return;
				lblMinrange.setText(String.valueOf(rangeSlider.getValue()));
				lblMaxrange.setText(String.valueOf(rangeSlider.getUpperValue()));
			}
		});
		groupLayout.setAutoCreateGaps(true);
		getContentPane().setLayout(groupLayout);
		pack();
		importImages();
	}

	private void importImages() {
		final OpenDialog od = new OpenDialog("Import Calibration Images");
		if(od.getFileName()==null) return;
       
		final File file = new File(od.getDirectory()+od.getFileName());
        
        ImagePlus calibImage = new ImagePlus();
		if (file.isDirectory()){
        	calibImage = FolderOpener.open(file.getAbsolutePath());
        }
        
        if (file.isFile()){
        	calibImage = new ImagePlus(file.getAbsolutePath());
        }
        
        calibWindow = new StackWindow(calibImage);
        calibImage.setRoi(calibImage.getWidth()/2 - 10, calibImage.getHeight()/2 - 10, 20, 20);	
	}
	
	protected void fitbeads() {
		Roi roitemp = calibWindow.getImagePlus().getRoi();
		Roi calibRoi;
		try{																				
			double w = roitemp.getFloatWidth();
			double h = roitemp.getFloatHeight();
			if (w!=h) {
				IJ.showMessage("Needs a quadratic ROI /n(hint: press Shift).");
				return;
			}
			calibRoi = roitemp;
		} catch (NullPointerException e) {
			calibRoi = new Roi(0, 0, calibWindow.getImagePlus().getWidth(), calibWindow.getImagePlus().getHeight());
		} 
		
		int zstep = (int) this.spinnerStepSize.getValue();
		calibrator = new Calibrator(calibWindow.getImagePlus(), zstep, calibRoi);	
		calibrator.fitStack();
		double[] zgrid = calibrator.getZgrid();
		Arrays.sort(zgrid);
		this.rangeSlider.setMinimum((int) zgrid[0]);
		this.rangeSlider.setMaximum((int) zgrid[zgrid.length-1]);
		this.rangeSlider.setValue((int) zgrid[0]);
		this.rangeSlider.setUpperValue((int) zgrid[zgrid.length-1]);

		calibWindow.close();
	}
	
	protected void fitCurve() {
		int rangeMin = this.rangeSlider.getValue();
		int rangeMax = this.rangeSlider.getUpperValue();
		calibrator.fitBSplines(rangeMin, rangeMax);
	}


	public Map<String, Object> getSettings() {
		HashMap<String, Object> setting = new HashMap<>(1);
		if (calibFile!=null) {
			setting.put(KEY_CALIBRATION_FILE, calibFile);
		}
		return setting;
	}

	private void saveCalibration() {
		JFileChooser fc = new JFileChooser(System.getProperty("user.home"));
		fc.setLocation(getLocation());
    	fc.setDialogTitle("Save calibration");   
        FileNameExtensionFilter filter = new FileNameExtensionFilter("Csv files", "csv");
        fc.setFileFilter(filter);
    	 
    	if (fc.showSaveDialog(this) == JFileChooser.APPROVE_OPTION) {
    	    calibFile = fc.getSelectedFile();
    	    calibrator.saveCalib(calibFile.getAbsolutePath());
    	}		
    	calibrator.closePlotWindows();
    	setVisible(false);
	}
}
