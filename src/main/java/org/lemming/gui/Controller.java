package org.lemming.gui;

import ij.IJ;
import ij.ImagePlus;
import ij.WindowManager;
import ij.io.FileInfo;
import ij.io.TiffDecoder;
import ij.plugin.FileInfoVirtualStack;
import ij.plugin.FolderOpener;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.NumericType;

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
import javax.swing.JFileChooser;
import javax.swing.JLabel;
import javax.swing.JList;
import javax.swing.LayoutStyle.ComponentPlacement;
import javax.swing.DefaultListCellRenderer;
import javax.swing.JComboBox;
import javax.swing.DefaultComboBoxModel;
import javax.swing.JCheckBox;
import javax.swing.JSpinner;
import javax.swing.UIManager;
import javax.swing.UnsupportedLookAndFeelException;

import org.japura.gui.CheckComboBox;
import org.japura.gui.event.ListCheckListener;
import org.japura.gui.event.ListEvent;
import org.japura.gui.model.ListCheckModel;
import org.lemming.factories.DetectorFactory;
import org.lemming.factories.FitterFactory;
import org.lemming.factories.PreProcessingFactory;
import org.lemming.factories.RendererFactory;
import org.lemming.modules.ImageLoader;
import org.lemming.modules.ImageMath;
import org.lemming.modules.ImageMath.operators;
import org.lemming.modules.SaveFittedLocalizations;
import org.lemming.modules.StoreLoader;
import org.lemming.modules.TableLoader;
import org.lemming.pipeline.AbstractModule;
import org.lemming.pipeline.ExtendableTable;
import org.lemming.pipeline.Manager;
import org.lemming.providers.ActionProvider;
import org.lemming.providers.DetectorProvider;
import org.lemming.providers.FitterProvider;
import org.lemming.providers.PreProcessingProvider;
import org.lemming.providers.RendererProvider;

import java.beans.PropertyChangeListener;
import java.io.File;
import java.io.IOException;
import java.beans.PropertyChangeEvent;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.awt.Component;

import javax.swing.SpinnerNumberModel;
import java.awt.Dimension;
import java.awt.event.ContainerEvent;
import java.awt.event.ContainerListener;

public class Controller<T extends NumericType<T> & NativeType<T>> extends JFrame implements ActionListener,PropertyChangeListener,ListCheckListener,ContainerListener {

	private static final long serialVersionUID = -2596199192028890712L;

	private JPanel contentPane;
	
	private JTabbedPane tabbedPane;
	
	private JPanel panelLoc;
	
	private JPanel panelRecon;
	
	private JPanel panelPreview;

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
	private JLabel lblFitter;

	private FitterProvider fitterProvider;
	private JComboBox<String> comboBoxFitter;

	private RendererProvider rendererProvider;
	private JPanel panel;
	private JLabel lblRenderer;
	private JComboBox<String> comboBoxRenderer;
	private JCheckBox chkboxFilter;

	private ActionProvider actionProvider;

	private List<String> checksPreprocessing;

	private PreProcessingProvider preProcessingProvider;

	private PreProcessingFactory preProcessingFactory;

	private FitterFactory fitterFactory;

	private RendererFactory rendererFactory;

	private ConfigurationPanel panelReconDown;
	private JLabel lblFile;

	private Manager manager;

	private ExtendableTable table;

	private ImageLoader<T> tif;

	private StoreLoader storeLoader;
	
	private Map<String,Object> settings;

	private File saveFile; 

	/**
	 * Create the frame.
	 * @param imp 
	 */
	public Controller(ImagePlus imp) {
		try {
			UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
		} catch (ClassNotFoundException | InstantiationException | IllegalAccessException | UnsupportedLookAndFeelException e1) {
			IJ.error(e1.getMessage());
		}
		setTitle("Lemming");
		setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
		setBounds(100, 100, 650, 470);
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
		panelLoc.addContainerListener(this);
		panelLoc.setBorder(UIManager.getBorder("List.focusCellHighlightBorder"));
		tabbedPane.addTab("Localize", null, panelLoc, null);
		GridBagLayout gbl_panelLoc = new GridBagLayout();
		gbl_panelLoc.columnWidths = new int[] {300, 0};
		gbl_panelLoc.rowHeights = new int[] {140, 20, 210};
		gbl_panelLoc.columnWeights = new double[]{1.0, Double.MIN_VALUE};
		gbl_panelLoc.rowWeights = new double[]{1.0, 0.0, 1.0};
		panelLoc.setLayout(gbl_panelLoc);
		
		JPanel panelUpper = new JPanel();
		
		lblPreprocessing = new JLabel("Preprocessing");
		
		jComboBoxPreprocessing = new CheckComboBox();
		ListCheckModel preprocessingModel = jComboBoxPreprocessing.getModel();
		preprocessingModel.addListCheckListener(this);
		
		lblPeakDet = new JLabel("Peak Detector");
		GridBagConstraints gbc_panelUpper = new GridBagConstraints();
		gbc_panelUpper.anchor = GridBagConstraints.NORTHWEST;
		gbc_panelUpper.insets = new Insets(0, 0, 5, 0);
		gbc_panelUpper.gridx = 0;
		gbc_panelUpper.gridy = 0;
		panelLoc.add(panelUpper, gbc_panelUpper);
		
		comboBoxPeakDet = new JComboBox<>();
		comboBoxPeakDet.setPreferredSize(new Dimension(32, 26));
		
		lblFitter = new JLabel("Fitter");
		
		comboBoxFitter = new JComboBox<>();
		comboBoxFitter.setPreferredSize(new Dimension(32, 26));
		comboBoxFitter.addActionListener(this);
		
		lblFile = new JLabel("File");
		GroupLayout gl_panelUpper = new GroupLayout(panelUpper);
		gl_panelUpper.setHorizontalGroup(
			gl_panelUpper.createParallelGroup(Alignment.LEADING)
				.addGroup(gl_panelUpper.createSequentialGroup()
					.addContainerGap()
					.addGroup(gl_panelUpper.createParallelGroup(Alignment.LEADING)
						.addComponent(lblFile, GroupLayout.DEFAULT_SIZE, 290, Short.MAX_VALUE)
						.addGroup(gl_panelUpper.createSequentialGroup()
							.addComponent(lblPreprocessing)
							.addPreferredGap(ComponentPlacement.UNRELATED)
							.addComponent(jComboBoxPreprocessing, GroupLayout.PREFERRED_SIZE, 170, GroupLayout.PREFERRED_SIZE))
						.addGroup(gl_panelUpper.createSequentialGroup()
							.addGroup(gl_panelUpper.createParallelGroup(Alignment.LEADING)
								.addComponent(lblPeakDet)
								.addComponent(lblFitter))
							.addPreferredGap(ComponentPlacement.RELATED)
							.addGroup(gl_panelUpper.createParallelGroup(Alignment.TRAILING)
								.addComponent(comboBoxFitter, 0, 177, Short.MAX_VALUE)
								.addComponent(comboBoxPeakDet, Alignment.LEADING, 0, 177, Short.MAX_VALUE))))
					.addContainerGap())
		);
		gl_panelUpper.setVerticalGroup(
			gl_panelUpper.createParallelGroup(Alignment.LEADING)
				.addGroup(gl_panelUpper.createSequentialGroup()
					.addComponent(lblFile)
					.addPreferredGap(ComponentPlacement.RELATED)
					.addGroup(gl_panelUpper.createParallelGroup(Alignment.BASELINE)
						.addComponent(jComboBoxPreprocessing, GroupLayout.PREFERRED_SIZE, GroupLayout.DEFAULT_SIZE, GroupLayout.PREFERRED_SIZE)
						.addComponent(lblPreprocessing, GroupLayout.PREFERRED_SIZE, 27, GroupLayout.PREFERRED_SIZE))
					.addPreferredGap(ComponentPlacement.RELATED)
					.addGroup(gl_panelUpper.createParallelGroup(Alignment.LEADING)
						.addComponent(comboBoxPeakDet, GroupLayout.DEFAULT_SIZE, 27, Short.MAX_VALUE)
						.addComponent(lblPeakDet, GroupLayout.PREFERRED_SIZE, 27, GroupLayout.PREFERRED_SIZE))
					.addPreferredGap(ComponentPlacement.UNRELATED)
					.addGroup(gl_panelUpper.createParallelGroup(Alignment.BASELINE)
						.addComponent(comboBoxFitter, GroupLayout.PREFERRED_SIZE, GroupLayout.DEFAULT_SIZE, GroupLayout.PREFERRED_SIZE)
						.addComponent(lblFitter))
					.addGap(21))
		);
		panelUpper.setLayout(gl_panelUpper);
		comboBoxPeakDet.addActionListener(this);
		
		panelMiddle = new JPanel();
		FlowLayout flowLayout = (FlowLayout) panelMiddle.getLayout();
		flowLayout.setVgap(0);
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
		panelRecon.setBorder(UIManager.getBorder("List.focusCellHighlightBorder"));
		tabbedPane.addTab("Reconstruct", null, panelRecon, null);
		GridBagLayout gbl_panelRecon = new GridBagLayout();
		gbl_panelRecon.columnWidths = new int[] {300, 0};
		gbl_panelRecon.rowHeights = new int[] {70, 280};
		gbl_panelRecon.columnWeights = new double[]{1.0};
		gbl_panelRecon.rowWeights = new double[]{1.0, 0.0};
		panelRecon.setLayout(gbl_panelRecon);
		
		panel = new JPanel();
		GridBagConstraints gbc_panel = new GridBagConstraints();
		gbc_panel.insets = new Insets(0, 0, 5, 0);
		gbc_panel.fill = GridBagConstraints.BOTH;
		gbc_panel.gridx = 0;
		gbc_panel.gridy = 0;
		panelRecon.add(panel, gbc_panel);
		
		lblRenderer = new JLabel("Renderer");
		
		comboBoxRenderer = new JComboBox<>();
		comboBoxRenderer.addActionListener(this);
		
		chkboxFilter = new JCheckBox("Filter");
		GroupLayout gl_panel = new GroupLayout(panel);
		gl_panel.setHorizontalGroup(
			gl_panel.createParallelGroup(Alignment.LEADING)
				.addGroup(gl_panel.createSequentialGroup()
					.addContainerGap()
					.addGroup(gl_panel.createParallelGroup(Alignment.LEADING)
						.addGroup(gl_panel.createSequentialGroup()
							.addComponent(lblRenderer)
							.addGap(39)
							.addComponent(comboBoxRenderer, GroupLayout.PREFERRED_SIZE, 165, GroupLayout.PREFERRED_SIZE))
						.addComponent(chkboxFilter))
					.addContainerGap(39, Short.MAX_VALUE))
		);
		gl_panel.setVerticalGroup(
			gl_panel.createParallelGroup(Alignment.LEADING)
				.addGroup(gl_panel.createSequentialGroup()
					.addContainerGap()
					.addGroup(gl_panel.createParallelGroup(Alignment.BASELINE)
						.addComponent(comboBoxRenderer, GroupLayout.PREFERRED_SIZE, GroupLayout.DEFAULT_SIZE, GroupLayout.PREFERRED_SIZE)
						.addComponent(lblRenderer))
					.addPreferredGap(ComponentPlacement.RELATED)
					.addComponent(chkboxFilter)
					.addContainerGap(24, Short.MAX_VALUE))
		);
		panel.setLayout(gl_panel);
		
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
		createFitterProvider();
		createRendererProvider();
		createActionProvider();
		try {
			saveFile = File.createTempFile("Lemming", ".tmp");
		} catch (IOException e) {
			IJ.error(e.getMessage());
		}
		manager = new Manager();
	}

////Overrides
	
	@SuppressWarnings("rawtypes")
	@Override
	public void actionPerformed(ActionEvent e) {
		Object s = e.getSource();
		
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
		
		if (s == this.comboBoxFitter){
			if (panelDown != null)				// remove panel if one exists
				panelLoc.remove(panelDown);
			chooseFitter();
		}
		
		if (s == this.comboBoxRenderer){
			if (panelReconDown != null)				// remove panel if one exists
				panelRecon.remove(panelReconDown);
			chooseRenderer();
		}
		
		if (s == this.btnProcess){ // Manager
			// TODO sanity checks
			
			if (tabbedPane.getSelectedIndex()==0){
				if (panelDown != null){
					Map<String, Object> curSet = panelDown.getSettings();
					for (String key : curSet.keySet())
						settings.put(key, curSet.get(key));
			} else if (tabbedPane.getSelectedIndex()==1){
				if (panelReconDown != null){
					Map<String, Object> curSet = panelReconDown.getSettings();
					for (String key : curSet.keySet())
						settings.put(key, curSet.get(key));
				}
			}
				
			if (tif==null) {
				IJ.error("Please load images first!");
				return;
			}
			detectorFactory.setAndCheckSettings(settings);
			AbstractModule detector = detectorFactory.getDetector();
			manager.add(detector);
			
			if (!checksPreprocessing.isEmpty()){
				AbstractModule pp = preProcessingFactory.getModule();
				manager.add(pp);
				manager.linkModules(tif, pp);
				operators op = preProcessingFactory.getOperator();
				ImageMath math = new ImageMath();
				math.setOperator(op);
				manager.add(math);
				manager.linkModules(tif, math);
				manager.linkModules(pp, math);
				manager.linkModules(math, detector);
			} else {
				manager.linkModules(tif, detector, true);
			}			
			/*UnpackElements unpacker = new UnpackElements();
			manager.add(unpacker);
			manager.linkModules(detector, unpacker);*/
			fitterFactory.setAndCheckSettings(settings);
			AbstractModule fitter = fitterFactory.getFitter();
			manager.add(fitter);
			manager.linkModules(tif, fitter);
			manager.linkModules(detector, fitter);				
			rendererFactory.setAndCheckSettings(settings);
			AbstractModule renderer = rendererFactory.getRenderer();
			manager.add(renderer);
			manager.linkModules(fitter, renderer, false);
			SaveFittedLocalizations saver = new SaveFittedLocalizations(saveFile);
			manager.add(saver);
			manager.linkModules(fitter, saver, false);
			}			
		}
		
		if (s == this.btnLoad){
			if (tabbedPane.getSelectedIndex()==0)
				loadImages();
			else
				loadLocalizations();
		}
		
		if (s == this.btnSave){
			if (chkboxFilter.isSelected()){
				
			} else {
				JFileChooser fc = new JFileChooser(System.getProperty("user.home")+"/ownCloud/storm");
		    	fc.setFileSelectionMode(JFileChooser.FILES_ONLY);
		    	fc.setDialogTitle("Save Data");
		    	 
		        if (fc.showSaveDialog(this) != JFileChooser.APPROVE_OPTION)
		        	return;
		        fc.getSelectedFile();
				
			}
		}
	}

	@Override
	public void propertyChange(PropertyChangeEvent evt) {
		
	}
	
	@Override
	public void addCheck(ListEvent event) {
		ListCheckModel source = event.getSource();
		if (source == this.jComboBoxPreprocessing.getModel()){
			if (panelDown != null)
				panelLoc.remove(panelDown);
			List<Object> checks = jComboBoxPreprocessing.getModel().getCheckeds();
			String selected = null;
			for (Object o : checks){
				String s = (String) o;
				if (!checksPreprocessing.contains(s)){
					selected = s;
					checksPreprocessing.add(s);
				}				
			}
			List<String> visibleKeys = preProcessingProvider.getVisibleKeys();
			String selectedKey=null;
			for ( final String key : visibleKeys ){
				String currentName = preProcessingProvider.getFactory( key ).getName();
				if(currentName.contains(selected))
					selectedKey = key;
			}
			if (selectedKey==null) return;
			
			System.out.println("preProcessing: " + selected);
			preProcessingFactory = preProcessingProvider.getFactory(selectedKey);
			panelDown = preProcessingFactory.getConfigurationPanel();
			GridBagConstraints gbc_panelDown = new GridBagConstraints();
			gbc_panelDown.anchor = GridBagConstraints.NORTHWEST;
			gbc_panelDown.gridx = 0;
			gbc_panelDown.gridy = 2;
			panelLoc.add(panelDown, gbc_panelDown);
			this.validate();
			this.repaint();
		}
	}

	@Override
	public void removeCheck(ListEvent event) {
		ListCheckModel source = event.getSource();
		if (source == this.jComboBoxPreprocessing.getModel()){
			if (panelDown != null)
				panelLoc.remove(panelDown);
			List<Object> checks = jComboBoxPreprocessing.getModel().getCheckeds();
			String removalString = null;
			for (String s : checksPreprocessing){
				if(!checks.contains(s))
					removalString = s;
			}
			if (removalString!=null)
				checksPreprocessing.remove(removalString);	
			this.validate();
			this.repaint();
		}
	}
	
	@Override
	public void componentAdded(ContainerEvent e) {
		
	}

	@Override
	public void componentRemoved(ContainerEvent e) {
		if (tabbedPane.getSelectedIndex()==0){
			Map<String, Object> curSet = panelDown.getSettings();
			for (String key : curSet.keySet())
				settings.put(key, curSet.get(key));
		} else if (tabbedPane.getSelectedIndex()==1){
			Map<String, Object> curSet = panelReconDown.getSettings();
			for (String key : curSet.keySet())
				settings.put(key, curSet.get(key));
		}		
	}
	
	//// Private Methods
	
	private void loadImages() {
	    manager.reset();
		ImagePlus loc_im = WindowManager.getCurrentImage();
		
		if (loc_im!=null) {
			this.lblFile.setText(loc_im.getTitle());
			return;
		}
		
		JFileChooser fc = new JFileChooser(System.getProperty("user.home")+"/ownCloud/storm");
    	fc.setFileSelectionMode(JFileChooser.FILES_AND_DIRECTORIES);
    	fc.setDialogTitle("Import Images");
    	
    	int returnVal = fc.showOpenDialog(this);
    	 
        if (returnVal != JFileChooser.APPROVE_OPTION)
        	return;
        
        File file = fc.getSelectedFile();
        
        
		if (file.isDirectory()){
        	FolderOpener fo = new FolderOpener();
        	fo.openAsVirtualStack(true);
        	loc_im = fo.openFolder(file.getAbsolutePath());
        }
        
        if (file.isFile()){
        	File dir = file.getParentFile();
        	TiffDecoder td = new TiffDecoder(dir.getAbsolutePath(), file.getName());
        	FileInfo[] info;
			try {info = td.getTiffInfo();}
    		catch (IOException e) {
    			String msg = e.getMessage();
    			if (msg==null||msg.equals("")) msg = ""+e;
    			IJ.error("TiffDecoder", msg);
    			return;
    		}
    		if (info==null || info.length==0) {
    			IJ.error("Virtual Stack", "This does not appear to be a TIFF stack");
    			return;
    		}
        	FileInfoVirtualStack fivs = new FileInfoVirtualStack(info[0], false);
        	loc_im = new ImagePlus("",fivs);
        }
        
        tif = new ImageLoader<>(loc_im);
        
        manager.add(tif);
        
        this.lblFile.setText(file.getName());
        this.btnLoad.setEnabled(false);
	}
	
	private void loadLocalizations() {
	    manager.reset();
		JFileChooser fc = new JFileChooser(System.getProperty("user.home")+"/ownCloud/storm");
    	fc.setFileSelectionMode(JFileChooser.FILES_ONLY);
    	fc.setDialogTitle("Import Data");
    	
    	int returnVal = fc.showOpenDialog(this);
    	 
        if (returnVal != JFileChooser.APPROVE_OPTION)
        	return;
        
        File file = fc.getSelectedFile();

		if (this.chkboxFilter.isSelected()){
        	TableLoader tl = new TableLoader(file,",");
        	table = tl.getTable();
		}
        else {
        	storeLoader = new StoreLoader(file,",");
        	manager.add(storeLoader);
        }
		
		this.lblFile.setText(file.getName());
		this.btnLoad.setEnabled(false);
	}

	
	private void chooseDetector(){
		final int index = comboBoxPeakDet.getSelectedIndex();
		final String key = detectorProvider.getVisibleKeys().get( index );
		
		detectorFactory = detectorProvider.getFactory( key );
		panelDown = detectorFactory.getConfigurationPanel();
		System.out.println("Detector_"+index+" : "+key);
		
		GridBagConstraints gbc_panelDown = new GridBagConstraints();
		gbc_panelDown.anchor = GridBagConstraints.NORTHWEST;
		gbc_panelDown.gridx = 0;
		gbc_panelDown.gridy = 2;
		panelLoc.add(panelDown, gbc_panelDown);
		this.validate();
		this.repaint();
	}
	
	private void chooseFitter() {
		final int index = comboBoxFitter.getSelectedIndex();
		final String key = fitterProvider.getVisibleKeys().get( index );
		
		fitterFactory = fitterProvider.getFactory( key );
		panelDown = fitterFactory.getConfigurationPanel();
		System.out.println("Fitter_"+index+" : "+key);
		
		GridBagConstraints gbc_panelDown = new GridBagConstraints();
		gbc_panelDown.anchor = GridBagConstraints.NORTHWEST;
		gbc_panelDown.gridx = 0;
		gbc_panelDown.gridy = 2;
		panelLoc.add(panelDown, gbc_panelDown);
		this.validate();
		this.repaint();
	}
	
	private void chooseRenderer() {
		final int index = comboBoxRenderer.getSelectedIndex();
		final String key = rendererProvider.getVisibleKeys().get( index );
		rendererFactory = rendererProvider.getFactory( key );
		System.out.println("Renderer_"+index+" : "+key);
		panelReconDown = rendererFactory.getConfigurationPanel();
		
		GridBagConstraints gbc_panelDown = new GridBagConstraints();
		gbc_panelDown.anchor = GridBagConstraints.NORTHWEST;
		gbc_panelDown.gridx = 0;
		gbc_panelDown.gridy = 1;
		panelRecon.add(panelReconDown, gbc_panelDown);
		this.validate();
		this.repaint();		
	}
	
	private void createDetectorProvider(){
		detectorProvider = new DetectorProvider();
		final List< String > visibleKeys = detectorProvider.getVisibleKeys();
		final List< String > detectorNames = new ArrayList<>( visibleKeys.size() );
		final List< String > infoTexts = new ArrayList<>( visibleKeys.size() );
		for ( final String key : visibleKeys ){
			detectorNames.add( detectorProvider.getFactory( key ).getName() );
			infoTexts.add( detectorProvider.getFactory( key ).getInfoText() );
		}
		String[] names = detectorNames.toArray(new String[] {});
		comboBoxPeakDet.setModel(new DefaultComboBoxModel<>(names));
		comboBoxPeakDet.setRenderer(new ToolTipRenderer(infoTexts));
		if (comboBoxPeakDet.getSelectedObjects().length>0) comboBoxPeakDet.setSelectedIndex(0);
	}
	
	private void createPreProcessingProvider() {
		preProcessingProvider = new PreProcessingProvider();
		checksPreprocessing = new ArrayList<>();
		jComboBoxPreprocessing.setTextFor(CheckComboBox.NONE, "none"); 
		jComboBoxPreprocessing.setTextFor(CheckComboBox.MULTIPLE, "multiple"); 
		jComboBoxPreprocessing.setTextFor(CheckComboBox.ALL, "all");
		ListCheckModel model = jComboBoxPreprocessing.getModel();
		final List< String > visibleKeys = preProcessingProvider.getVisibleKeys();
		final List< String > infoTexts = new ArrayList<>( visibleKeys.size() );
		for ( final String key : visibleKeys ){
			String currentName = preProcessingProvider.getFactory( key ).getName();
			infoTexts.add( preProcessingProvider.getFactory( key ).getInfoText() );
			model.addElement(currentName);
		}
	}
	
	private void createFitterProvider() {
		fitterProvider = new FitterProvider();
		final List< String > visibleKeys = fitterProvider.getVisibleKeys();
		final List< String > fitterNames = new ArrayList<>( visibleKeys.size() );
		final List< String > infoTexts = new ArrayList<>( visibleKeys.size() );
		for ( final String key : visibleKeys ){
			fitterNames.add( fitterProvider.getFactory( key ).getName() );
			infoTexts.add( fitterProvider.getFactory( key ).getInfoText() );
		}
		String[] names = fitterNames.toArray(new String[] {});
		comboBoxFitter.setModel(new DefaultComboBoxModel<>(names));
		comboBoxFitter.setRenderer(new ToolTipRenderer(infoTexts));
		if (comboBoxFitter.getSelectedObjects().length>0) comboBoxFitter.setSelectedIndex(0);
	}
	
	private void createRendererProvider() {
		rendererProvider = new RendererProvider();
		final List< String > visibleKeys = rendererProvider.getVisibleKeys();
		final List< String > fitterNames = new ArrayList<>( visibleKeys.size() );
		final List< String > infoTexts = new ArrayList<>( visibleKeys.size() );
		for ( final String key : visibleKeys ){
			fitterNames.add( rendererProvider.getFactory( key ).getName() );
			infoTexts.add( rendererProvider.getFactory( key ).getInfoText() );
		}
		String[] names = fitterNames.toArray(new String[] {});
		comboBoxRenderer.setModel(new DefaultComboBoxModel<>(names));
		comboBoxRenderer.setRenderer(new ToolTipRenderer(infoTexts));
		if (comboBoxRenderer.getSelectedObjects().length>0) comboBoxRenderer.setSelectedIndex(0);
	}
	
	private void createActionProvider() {
		actionProvider = new ActionProvider();
		final List< String > visibleKeys = actionProvider.getVisibleKeys();
		final List< String > actionNames = new ArrayList<>( visibleKeys.size() );
		final List< String > infoTexts = new ArrayList<>( visibleKeys.size() );
		for ( final String key : visibleKeys ){
			actionNames.add( actionProvider.getFactory( key ).getName() );
			infoTexts.add( actionProvider.getFactory( key ).getInfoText() );
		}
		String[] names = actionNames.toArray(new String[] {});
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
