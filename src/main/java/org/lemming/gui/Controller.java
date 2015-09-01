package org.lemming.gui;

import ij.ImagePlus;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.border.EmptyBorder;
import javax.swing.JTabbedPane;

import java.awt.GridBagLayout;
import java.awt.GridBagConstraints;
import java.awt.Insets;
import javax.swing.JButton;

import java.awt.FlowLayout;

import javax.swing.GroupLayout;
import javax.swing.GroupLayout.Alignment;
import javax.swing.ImageIcon;
import javax.swing.JComponent;
import javax.swing.JLabel;
import javax.swing.JList;
import javax.swing.LayoutStyle.ComponentPlacement;
import javax.swing.DefaultListCellRenderer;
import javax.swing.JComboBox;
import javax.swing.DefaultComboBoxModel;
import javax.swing.JCheckBox;
import javax.swing.JFileChooser;
import javax.swing.JSpinner;
import javax.swing.UIManager;
import javax.swing.UnsupportedLookAndFeelException;

import org.japura.gui.CheckComboBox;
import org.japura.gui.model.ListCheckModel;
import org.lemming.factories.DetectorFactory;
import org.lemming.providers.DetectorProvider;

import java.beans.PropertyChangeListener;
import java.beans.PropertyChangeEvent;
import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.awt.Component;

import javax.swing.SpinnerNumberModel;

public class Controller extends JFrame implements ActionListener,PropertyChangeListener{

	private static final long serialVersionUID = -2596199192028890712L;

	private JPanel contentPane;
	
	private JTabbedPane tabbedPane;
	
	private JPanel panelLoc;
	
	private JPanel panelRecon;
	
	private JPanel panelPreview;

	private JButton btnCalibFile;

	private JLabel lblCalibFile;

	private JButton btnCameraFile;

	private JLabel lblCamFile;

	private CheckComboBox jComboBoxPreprocessing;

	private JLabel lblPreprocessing;

	private JLabel lblPeakDet;

	private JComboBox<String> comboBoxPeakDet;

	private JCheckBox chckbxROI;

	private JLabel lblSkipFrames;

	private JSpinner spinnerSkipFrames;

	private JButton btnLoad;

	private JButton btnSave;
	private JPanel panelMiddle;

	private DetectorProvider detectorProvider;
	private JButton btnProcess;

	private ConfigurationPanel panelDown;

	private DetectorFactory detectorFactory;

	/**
	 * Create the frame.
	 * @param imp 
	 */
	public Controller(ImagePlus imp) {
		try {
			UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
		} catch (ClassNotFoundException | InstantiationException | IllegalAccessException | UnsupportedLookAndFeelException e1) {
			e1.printStackTrace();
		}
		setTitle("Lemming");
		setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
		setBounds(100, 100, 640, 440);
		contentPane = new JPanel();
		contentPane.setBorder(new EmptyBorder(5, 5, 5, 5));
		setContentPane(contentPane);
		GridBagLayout gbl_contentPane = new GridBagLayout();
		gbl_contentPane.columnWidths = new int[] {200, 300, 0};
		gbl_contentPane.rowHeights = new int[] {200, 15, 0};
		gbl_contentPane.columnWeights = new double[]{1.0, 1.0, Double.MIN_VALUE};
		gbl_contentPane.rowWeights = new double[]{1.0, 0.0, Double.MIN_VALUE};
		contentPane.setLayout(gbl_contentPane);
		
		tabbedPane = new JTabbedPane(JTabbedPane.TOP);
		GridBagConstraints gbc_tabbedPane = new GridBagConstraints();
		gbc_tabbedPane.insets = new Insets(0, 0, 5, 5);
		gbc_tabbedPane.fill = GridBagConstraints.BOTH;
		gbc_tabbedPane.gridx = 0;
		gbc_tabbedPane.gridy = 0;
		contentPane.add(tabbedPane, gbc_tabbedPane);
		
		panelLoc = new JPanel();
		panelLoc.setBorder(UIManager.getBorder("List.focusCellHighlightBorder"));
		tabbedPane.addTab("Localize", null, panelLoc, null);
		GridBagLayout gbl_panelLoc = new GridBagLayout();
		gbl_panelLoc.columnWidths = new int[] {280, 0};
		gbl_panelLoc.rowHeights = new int[] {100, 15, 150};
		gbl_panelLoc.columnWeights = new double[]{1.0, Double.MIN_VALUE};
		gbl_panelLoc.rowWeights = new double[]{1.0, 0.0, 1.0};
		panelLoc.setLayout(gbl_panelLoc);
		
		JPanel panelUpper = new JPanel();
		
		btnCameraFile = new JButton("Cam File");
		
		lblCamFile = new JLabel("File");
		
		btnCalibFile = new JButton("Calib. File");
		
		lblCalibFile = new JLabel("File");
		lblCalibFile.setAlignmentX(Component.CENTER_ALIGNMENT);
		
		lblPreprocessing = new JLabel("Preprocessing");
		
		jComboBoxPreprocessing = new CheckComboBox();
		
		lblPeakDet = new JLabel("Peak Detector");
		btnCalibFile.addActionListener(this);
		btnCameraFile.addActionListener(this);
		GridBagConstraints gbc_panelUpper = new GridBagConstraints();
		gbc_panelUpper.anchor = GridBagConstraints.NORTHWEST;
		gbc_panelUpper.insets = new Insets(0, 0, 5, 0);
		gbc_panelUpper.gridx = 0;
		gbc_panelUpper.gridy = 0;
		panelLoc.add(panelUpper, gbc_panelUpper);
		
		comboBoxPeakDet = new JComboBox<>();
		GroupLayout gl_panelUpper = new GroupLayout(panelUpper);
		gl_panelUpper.setHorizontalGroup(
			gl_panelUpper.createParallelGroup(Alignment.LEADING)
				.addGroup(gl_panelUpper.createSequentialGroup()
					.addGroup(gl_panelUpper.createParallelGroup(Alignment.LEADING)
						.addGroup(gl_panelUpper.createParallelGroup(Alignment.LEADING, false)
							.addGroup(gl_panelUpper.createSequentialGroup()
								.addComponent(btnCameraFile, GroupLayout.PREFERRED_SIZE, 90, GroupLayout.PREFERRED_SIZE)
								.addPreferredGap(ComponentPlacement.RELATED)
								.addComponent(lblCamFile, GroupLayout.DEFAULT_SIZE, GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
							.addGroup(gl_panelUpper.createSequentialGroup()
								.addComponent(btnCalibFile, GroupLayout.PREFERRED_SIZE, 90, GroupLayout.PREFERRED_SIZE)
								.addPreferredGap(ComponentPlacement.RELATED)
								.addComponent(lblCalibFile, GroupLayout.DEFAULT_SIZE, GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
							.addGroup(gl_panelUpper.createSequentialGroup()
								.addContainerGap()
								.addComponent(lblPreprocessing)
								.addPreferredGap(ComponentPlacement.UNRELATED)
								.addComponent(jComboBoxPreprocessing, GroupLayout.PREFERRED_SIZE, 170, GroupLayout.PREFERRED_SIZE)))
						.addGroup(gl_panelUpper.createSequentialGroup()
							.addContainerGap()
							.addComponent(lblPeakDet)
							.addGap(18)
							.addComponent(comboBoxPeakDet, 0, 164, Short.MAX_VALUE)))
					.addContainerGap())
		);
		gl_panelUpper.setVerticalGroup(
			gl_panelUpper.createParallelGroup(Alignment.LEADING)
				.addGroup(gl_panelUpper.createSequentialGroup()
					.addGroup(gl_panelUpper.createParallelGroup(Alignment.LEADING)
						.addComponent(btnCameraFile)
						.addComponent(lblCamFile, GroupLayout.PREFERRED_SIZE, 27, GroupLayout.PREFERRED_SIZE))
					.addPreferredGap(ComponentPlacement.RELATED)
					.addGroup(gl_panelUpper.createParallelGroup(Alignment.LEADING)
						.addComponent(btnCalibFile)
						.addComponent(lblCalibFile, GroupLayout.PREFERRED_SIZE, 27, GroupLayout.PREFERRED_SIZE))
					.addPreferredGap(ComponentPlacement.RELATED)
					.addGroup(gl_panelUpper.createParallelGroup(Alignment.BASELINE)
						.addComponent(jComboBoxPreprocessing)
						.addComponent(lblPreprocessing, GroupLayout.PREFERRED_SIZE, 27, GroupLayout.PREFERRED_SIZE))
					.addPreferredGap(ComponentPlacement.RELATED)
					.addGroup(gl_panelUpper.createParallelGroup(Alignment.LEADING)
						.addComponent(lblPeakDet, GroupLayout.PREFERRED_SIZE, 27, GroupLayout.PREFERRED_SIZE)
						.addComponent(comboBoxPeakDet, GroupLayout.PREFERRED_SIZE, 27, GroupLayout.PREFERRED_SIZE))
					.addContainerGap())
		);
		panelUpper.setLayout(gl_panelUpper);
		comboBoxPeakDet.addActionListener(this);
		
		panelMiddle = new JPanel();
		GridBagConstraints gbc_panelMiddle = new GridBagConstraints();
		gbc_panelMiddle.insets = new Insets(0, 0, 5, 0);
		gbc_panelMiddle.gridx = 0;
		gbc_panelMiddle.gridy = 1;
		panelLoc.add(panelMiddle, gbc_panelMiddle);
		
		chckbxROI = new JCheckBox("use ROI");
		panelMiddle.add(chckbxROI);
		chckbxROI.addActionListener(this);
		
		lblSkipFrames = new JLabel("     Skip frames");
		panelMiddle.add(lblSkipFrames);
		
		spinnerSkipFrames = new JSpinner();
		spinnerSkipFrames.setModel(new SpinnerNumberModel(new Integer(0), null, null, new Integer(1)));
		panelMiddle.add(spinnerSkipFrames);
		
		panelRecon = new JPanel();
		tabbedPane.addTab("Reconstruct", null, panelRecon, null);
		
		spinnerSkipFrames.addPropertyChangeListener(this);
		spinnerSkipFrames.setVisible(false);
		lblSkipFrames.setVisible(false);
		
		
		JLabel myIcon = new JLabel(new ImageIcon("test.jpg"));
		panelPreview = new JPanel();
		panelPreview.add(myIcon);
		GridBagConstraints gbc_panelPreview = new GridBagConstraints();
		gbc_panelPreview.insets = new Insets(30, 0, 10, 0);
		gbc_panelPreview.fill = GridBagConstraints.BOTH;
		gbc_panelPreview.gridx = 1;
		gbc_panelPreview.gridy = 0;
		contentPane.add(panelPreview, gbc_panelPreview);
		this.repaint();
		
		JPanel panelButtons = new JPanel();
		panelButtons.setBorder(null);
		GridBagConstraints gbc_panelButtons = new GridBagConstraints();
		gbc_panelButtons.fill = GridBagConstraints.BOTH;
		gbc_panelButtons.gridx = 0;
		gbc_panelButtons.gridy = 1;
		contentPane.add(panelButtons, gbc_panelButtons);
		panelButtons.setLayout(new FlowLayout(FlowLayout.CENTER, 20, 0));
		
		btnLoad = new JButton("Load");
		btnLoad.addActionListener(this);
		panelButtons.add(btnLoad);
		
		btnSave = new JButton("Save");
		btnSave.addActionListener(this);
		panelButtons.add(btnSave);
		
		btnProcess = new JButton("Process");
		btnProcess.addActionListener(this);
		panelButtons.add(btnProcess);
		init();
	}
	
	private void init(){
		createDetectorProvider();
		createPreProcessingProvider();
	}

	@Override
	public void actionPerformed(ActionEvent e) {
		Object s = e.getSource();
		// Localization Panel
		if (s == this.btnCalibFile){
			JFileChooser fc = new JFileChooser(System.getProperty("user.home")+"/ownCloud/storm");
	    	fc.setFileSelectionMode(JFileChooser.FILES_ONLY);
	    	fc.setDialogTitle("Import Calibration File");
	    	int returnVal = fc.showOpenDialog(this);
	    	 
	        if (returnVal != JFileChooser.APPROVE_OPTION)
	        	return;
	        File file = fc.getSelectedFile();
	        this.lblCalibFile.setText(file.getName());
		}
		
		if (s == this.btnCameraFile){
			JFileChooser fc = new JFileChooser(System.getProperty("user.home")+"/ownCloud/storm");
	    	fc.setFileSelectionMode(JFileChooser.FILES_ONLY);
	    	fc.setDialogTitle("Import Camera File");
	    	int returnVal = fc.showOpenDialog(this);
	    	 
	        if (returnVal != JFileChooser.APPROVE_OPTION)
	        	return;
	        File file = fc.getSelectedFile();
	        this.lblCalibFile.setText(file.getName());
		}
		
		if (s == this.chckbxROI){
			if (this.chckbxROI.isSelected()){
				this.lblSkipFrames.setVisible(true);
				this.spinnerSkipFrames.setVisible(true);
			} else {
				this.lblSkipFrames.setVisible(false);
				this.spinnerSkipFrames.setVisible(false);
			}
		}
		
		if (s == this.comboBoxPeakDet){
			if (panelDown != null)				// remove panel if one exists
				panelLoc.remove(panelDown);
			chooseDetector();
			
		}
		
		if (s == this.btnProcess){
			if (panelDown != null){
				
				detectorFactory.setAndCheckSettings(panelDown.getSettings());
				detectorFactory.getDetector();
				System.out.println("Detector " + detectorFactory.getDetector().getClass().getSimpleName());
			}
			
		}
	}

	@Override
	public void propertyChange(PropertyChangeEvent evt) {
		
	}
	
	private void chooseDetector(){
		final int index = comboBoxPeakDet.getSelectedIndex();
		final String key = detectorProvider.getVisibleKeys().get( index );
		
		detectorFactory = detectorProvider.getFactory( key );
		panelDown = detectorFactory.getDetectorConfigurationPanel();
		System.out.println("Detector_"+index+" : "+key);
		
		GridBagConstraints gbc_panelDown = new GridBagConstraints();
		gbc_panelDown.anchor = GridBagConstraints.NORTHWEST;
		gbc_panelDown.gridx = 0;
		gbc_panelDown.gridy = 2;
		panelLoc.add(panelDown, gbc_panelDown);
		this.validate();
		this.repaint();
	}
	
	private void createDetectorProvider(){
		detectorProvider = new DetectorProvider();
		final List< String > visibleKeys = detectorProvider.getVisibleKeys();
		final List< String > detectorNames = new ArrayList<>( visibleKeys.size() );
		final List< String > infoTexts = new ArrayList<>( visibleKeys.size() );
		for ( final String key : visibleKeys )
		{
			detectorNames.add( detectorProvider.getFactory( key ).getName() );
			infoTexts.add( detectorProvider.getFactory( key ).getInfoText() );
		}
		String[] names = detectorNames.toArray(new String[] {});
		comboBoxPeakDet.setModel(new DefaultComboBoxModel<>(names));
		comboBoxPeakDet.setRenderer(new ToolTipRenderer(infoTexts));
		comboBoxPeakDet.setSelectedIndex(0);
	}
	
	private void createPreProcessingProvider() {
		jComboBoxPreprocessing.setTextFor(CheckComboBox.NONE, "none"); 
		jComboBoxPreprocessing.setTextFor(CheckComboBox.MULTIPLE, "multiple"); 
		jComboBoxPreprocessing.setTextFor(CheckComboBox.ALL, "all");
		ListCheckModel model = jComboBoxPreprocessing.getModel();
		model.addElement("none");
	}
	
	
	private class ToolTipRenderer extends DefaultListCellRenderer {
		private static final long serialVersionUID = 1L;
		
		List<String> tooltips;
		
		public ToolTipRenderer(List<String> tooltips){
			 this.tooltips = tooltips;
		}

	    @SuppressWarnings("rawtypes")
		@Override
	    public Component getListCellRendererComponent(JList list, Object value, int index, boolean isSelected, boolean cellHasFocus) {

	        JComponent comp = (JComponent) super.getListCellRendererComponent(list, value, index, isSelected, cellHasFocus);

	        if (-1 < index && null != value && null != tooltips) {
	                    list.setToolTipText(tooltips.get(index));
	                }
	        return comp;
	    }
		
	}
}
