package org.lemming.gui;

import ij.IJ;
import ij.ImageListener;
import ij.ImagePlus;
import ij.WindowManager;
import ij.gui.ImageWindow;
import ij.gui.PointRoi;
import ij.gui.StackWindow;
import ij.io.FileInfo;
import ij.io.TiffDecoder;
import ij.plugin.FileInfoVirtualStack;
import ij.plugin.FolderOpener;
import ij.plugin.frame.ContrastAdjuster;
import ij.process.FloatPolygon;
import ij.process.ImageProcessor;
import net.imglib2.img.Img;
import net.imglib2.img.display.imagej.ImageJFunctions;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.NumericType;
import net.imglib2.type.numeric.RealType;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.ComponentEvent;
import java.awt.event.ComponentListener;
import java.awt.event.KeyEvent;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;

import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.border.EmptyBorder;
import javax.swing.JTabbedPane;

import java.awt.GridBagLayout;
import java.awt.GridBagConstraints;
import java.awt.Insets;
import java.awt.Rectangle;

import javax.swing.JButton;

import java.awt.FlowLayout;

import javax.swing.GroupLayout;
import javax.swing.GroupLayout.Alignment;
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
import org.japura.gui.renderer.CheckListRenderer;
import org.lemming.factories.DetectorFactory;
import org.lemming.factories.FitterFactory;
import org.lemming.factories.PreProcessingFactory;
import org.lemming.factories.RendererFactory;
import org.lemming.interfaces.Element;
import org.lemming.interfaces.Frame;
import org.lemming.interfaces.Store;
import org.lemming.modules.DataTable;
import org.lemming.modules.Fitter;
import org.lemming.modules.ImageLoader;
import org.lemming.modules.ImageMath;
import org.lemming.modules.ImageMath.operators;
import org.lemming.modules.Renderer;
import org.lemming.modules.TableLoader;
import org.lemming.pipeline.AbstractModule;
import org.lemming.pipeline.ExtendableTable;
import org.lemming.pipeline.FastStore;
import org.lemming.pipeline.FrameElements;
import org.lemming.pipeline.ImgLib2Frame;
import org.lemming.pipeline.Manager;
import org.lemming.providers.ActionProvider;
import org.lemming.providers.DetectorProvider;
import org.lemming.providers.FitterProvider;
import org.lemming.providers.PreProcessingProvider;
import org.lemming.providers.RendererProvider;
import org.lemming.tools.LemmingUtils;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.awt.Component;

import javax.swing.SpinnerNumberModel;

import java.awt.Dimension;
import java.awt.event.ContainerEvent;
import java.awt.event.ContainerListener;
import java.awt.event.KeyAdapter;

import javax.swing.event.ChangeListener;
import javax.swing.event.ChangeEvent;

import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.beans.PropertyChangeEvent;
import java.beans.PropertyChangeListener;

import javax.swing.JProgressBar;

import java.awt.CardLayout;
import java.awt.event.ComponentAdapter;


public class Controller<T extends NumericType<T> & NativeType<T> & RealType<T>  , F extends Frame<T>> extends JFrame implements 
	ActionListener,ListCheckListener {

	private static final long serialVersionUID = -2596199192028890712L;
	private JTabbedPane tabbedPane;
	private JPanel panelLoc;
	private JPanel panelRecon;
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
	private DetectorFactory detectorFactory;
	private JLabel lblFitter;
	private FitterProvider fitterProvider;
	private JComboBox<String> comboBoxFitter;
	private RendererProvider rendererProvider;
	private JLabel lblRenderer;
	private JComboBox<String> comboBoxRenderer;
	private JCheckBox chkboxFilter;
	private ActionProvider actionProvider;
	private List<String> checksPreprocessing;
	private PreProcessingProvider preProcessingProvider;
	private PreProcessingFactory preProcessingFactory;
	private FitterFactory fitterFactory;
	private RendererFactory rendererFactory;
	private JLabel lblFile;
	private Manager manager;
	private ExtendableTable table;
	private ImageLoader<T> tif;
	private Map<String,Object> settings;
	private File saveFile;
	private StackWindow previewerWindow;
	private AbstractModule detector;
	private Fitter<T,F> fitter;
	protected ContrastAdjuster contrastAdjuster;
	private Renderer renderer;
	private FrameElements<T> detResults;
	private List<Element> fitResults;
	private ImageWindow rendererWindow;
	protected int widgetSelection = 0;
	private boolean processed = false;
	protected ExtendableTable filteredTable = null;
	protected static int DETECTOR = 1;
	protected static int FITTER = 2;
	private JProgressBar progressBar;
	private JButton btnReset;
	private JPanel panelLower;
	private JPanel panelFilter;
	private CardLayout cardsFirst;
	private CardLayout cardsSecond;

	/**
	 * Create the frame.
	 * @param imp 
	 */
	@SuppressWarnings("serial")
	public Controller() {
		addWindowListener(new WindowAdapter() {
			@Override
			public void windowClosing(WindowEvent e) {
				if (previewerWindow != null)
					previewerWindow.close();
				if (contrastAdjuster !=null)
					contrastAdjuster.close();
				if (rendererWindow != null)
					rendererWindow.close();
			}
		});
		try {
			UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
		} catch (ClassNotFoundException | InstantiationException | IllegalAccessException | UnsupportedLookAndFeelException e1) {
			IJ.error(e1.getMessage());
		}
		setTitle("Lemming");
		setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
		setBounds(100, 100, 330, 500);
		JPanel contentPane = new JPanel();
		setContentPane(contentPane);

		GridBagLayout gbl_contentPane = new GridBagLayout();
		gbl_contentPane.columnWidths = new int[] {315, 0};
		gbl_contentPane.rowHeights = new int[] {400, 0, 30, 0};
		gbl_contentPane.columnWeights = new double[]{1.0, Double.MIN_VALUE};
		gbl_contentPane.rowWeights = new double[]{1.0, 0.0, 0.0, Double.MIN_VALUE};
		contentPane.setLayout(gbl_contentPane);
		
		tabbedPane = new JTabbedPane(JTabbedPane.TOP);
		tabbedPane.addChangeListener(new ChangeListener() {
			public void stateChanged(ChangeEvent e) {
				saveSettings(panelLower);
				saveSettings(panelFilter);
			}
		});
		tabbedPane.setBorder(new EmptyBorder(0, 0, 0, 0));
		
		panelLoc = new JPanel();
		panelLoc.setBorder(UIManager.getBorder("List.focusCellHighlightBorder"));
		tabbedPane.addTab("Localize", null, panelLoc, null);
		GridBagLayout gbl_panelLoc = new GridBagLayout();
		gbl_panelLoc.columnWidths = new int[] {250};
		gbl_panelLoc.rowHeights = new int[] {150, 20, 189};
		gbl_panelLoc.columnWeights = new double[]{1.0};
		gbl_panelLoc.rowWeights = new double[]{1.0, 0.0, 1.0};
		panelLoc.setLayout(gbl_panelLoc);
		
		JPanel panelUpper = new JPanel();
		
		lblPreprocessing = new JLabel("Preprocessing");
		
		jComboBoxPreprocessing = new CheckComboBox();
		ListCheckModel preprocessingModel = jComboBoxPreprocessing.getModel();
		preprocessingModel.addListCheckListener(this);
		lblPeakDet = new JLabel("Peak Detector");
		GridBagConstraints gbc_panelUpper = new GridBagConstraints();
		gbc_panelUpper.insets = new Insets(0, 0, 5, 0);
		gbc_panelUpper.anchor = GridBagConstraints.NORTHWEST;
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
		
		JLabel lblDataSource = new JLabel("Data source");
		GroupLayout gl_panelUpper = new GroupLayout(panelUpper);
		gl_panelUpper.setHorizontalGroup(
			gl_panelUpper.createParallelGroup(Alignment.LEADING)
				.addGroup(gl_panelUpper.createSequentialGroup()
					.addContainerGap()
					.addGroup(gl_panelUpper.createParallelGroup(Alignment.LEADING)
						.addGroup(gl_panelUpper.createSequentialGroup()
							.addComponent(lblPreprocessing)
							.addPreferredGap(ComponentPlacement.UNRELATED)
							.addComponent(jComboBoxPreprocessing, GroupLayout.DEFAULT_SIZE, 174, Short.MAX_VALUE))
						.addGroup(gl_panelUpper.createSequentialGroup()
							.addGroup(gl_panelUpper.createParallelGroup(Alignment.LEADING)
								.addComponent(lblPeakDet)
								.addComponent(lblFitter))
							.addPreferredGap(ComponentPlacement.RELATED)
							.addGroup(gl_panelUpper.createParallelGroup(Alignment.LEADING)
								.addComponent(comboBoxPeakDet, GroupLayout.PREFERRED_SIZE, 199, GroupLayout.PREFERRED_SIZE)
								.addComponent(comboBoxFitter, 0, 181, Short.MAX_VALUE)))
						.addGroup(gl_panelUpper.createSequentialGroup()
							.addComponent(lblDataSource)
							.addPreferredGap(ComponentPlacement.UNRELATED)
							.addComponent(lblFile, GroupLayout.PREFERRED_SIZE, 187, GroupLayout.PREFERRED_SIZE)))
					.addContainerGap())
		);
		gl_panelUpper.setVerticalGroup(
			gl_panelUpper.createParallelGroup(Alignment.LEADING)
				.addGroup(gl_panelUpper.createSequentialGroup()
					.addContainerGap()
					.addGroup(gl_panelUpper.createParallelGroup(Alignment.BASELINE)
						.addComponent(lblDataSource)
						.addComponent(lblFile))
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
		GridBagConstraints gbc_panelMiddle = new GridBagConstraints();
		gbc_panelMiddle.insets = new Insets(0, 0, 5, 0);
		gbc_panelMiddle.gridx = 0;
		gbc_panelMiddle.gridy = 1;
		panelLoc.add(panelMiddle, gbc_panelMiddle);
		
		chckbxROI = new JCheckBox("use ROI");
		chckbxROI.addActionListener(this);
		
		lblSkipFrames = new JLabel("Skip frames");
		
		spinnerSkipFrames = new JSpinner();
		spinnerSkipFrames.addChangeListener(new ChangeListener() {
			public void stateChanged(ChangeEvent e) {
			}
		});
		spinnerSkipFrames.setPreferredSize(new Dimension(40, 28));
		spinnerSkipFrames.setModel(new SpinnerNumberModel(new Integer(0), new Integer(0), null, new Integer(1)));
		GroupLayout gl_panelMiddle = new GroupLayout(panelMiddle);
		gl_panelMiddle.setHorizontalGroup(
			gl_panelMiddle.createParallelGroup(Alignment.LEADING)
				.addGroup(gl_panelMiddle.createSequentialGroup()
					.addContainerGap()
					.addComponent(chckbxROI)
					.addPreferredGap(ComponentPlacement.UNRELATED)
					.addComponent(lblSkipFrames)
					.addGap(18)
					.addComponent(spinnerSkipFrames, GroupLayout.PREFERRED_SIZE, 63, GroupLayout.PREFERRED_SIZE)
					.addContainerGap())
		);
		gl_panelMiddle.setVerticalGroup(
			gl_panelMiddle.createParallelGroup(Alignment.LEADING)
				.addGroup(gl_panelMiddle.createParallelGroup(Alignment.BASELINE)
					.addComponent(spinnerSkipFrames, GroupLayout.PREFERRED_SIZE, GroupLayout.DEFAULT_SIZE, GroupLayout.PREFERRED_SIZE)
					.addComponent(lblSkipFrames)
					.addComponent(chckbxROI))
		);
		panelMiddle.setLayout(gl_panelMiddle);
		
		panelLower = new JPanel();
		GridBagConstraints gbc_panelLower = new GridBagConstraints();
		gbc_panelLower.fill = GridBagConstraints.BOTH;
		gbc_panelLower.gridx = 0;
		gbc_panelLower.gridy = 2;
		panelLoc.add(panelLower, gbc_panelLower);
		cardsFirst=new CardLayout(0, 0);
		panelLower.setLayout(cardsFirst);
		
		ConfigurationPanel panelFirst = new ConfigurationPanel(){
			@Override
			public void setSettings(Map<String, Object> settings) {
			}

			@Override
			public Map<String, Object> getSettings() {
				return null;
			}
		};
		panelLower.add(panelFirst, "FIRST");
		
		panelRecon = new JPanel();
		panelRecon.setBorder(UIManager.getBorder("List.focusCellHighlightBorder"));
		tabbedPane.addTab("Reconstruct", null, panelRecon, null);
		GridBagLayout gbl_panelRecon = new GridBagLayout();
		gbl_panelRecon.columnWidths = new int[] {300};
		gbl_panelRecon.rowHeights = new int[] {80, 310};
		gbl_panelRecon.columnWeights = new double[]{1.0};
		gbl_panelRecon.rowWeights = new double[]{1.0, 1.0};
		panelRecon.setLayout(gbl_panelRecon);
		
		JPanel panelRenderer = new JPanel();
		GridBagConstraints gbc_panelRenderer = new GridBagConstraints();
		gbc_panelRenderer.insets = new Insets(0, 0, 5, 0);
		gbc_panelRenderer.fill = GridBagConstraints.BOTH;
		gbc_panelRenderer.gridx = 0;
		gbc_panelRenderer.gridy = 0;
		panelRecon.add(panelRenderer, gbc_panelRenderer);
		
		lblRenderer = new JLabel("Renderer");
		
		comboBoxRenderer = new JComboBox<>();
		comboBoxRenderer.addActionListener(this);
		
		chkboxFilter = new JCheckBox("Filter");
		chkboxFilter.addActionListener(this);
		
		btnReset = new JButton("Reset");
		btnReset.addActionListener(this);
		GroupLayout gl_panelRenderer = new GroupLayout(panelRenderer);
		gl_panelRenderer.setHorizontalGroup(
			gl_panelRenderer.createParallelGroup(Alignment.LEADING)
				.addGroup(gl_panelRenderer.createSequentialGroup()
					.addContainerGap()
					.addGroup(gl_panelRenderer.createParallelGroup(Alignment.LEADING)
						.addGroup(gl_panelRenderer.createSequentialGroup()
							.addComponent(lblRenderer)
							.addGap(39)
							.addComponent(comboBoxRenderer, GroupLayout.PREFERRED_SIZE, 165, GroupLayout.PREFERRED_SIZE))
						.addGroup(gl_panelRenderer.createSequentialGroup()
							.addComponent(chkboxFilter)
							.addPreferredGap(ComponentPlacement.RELATED, 117, Short.MAX_VALUE)
							.addComponent(btnReset))))
		);
		gl_panelRenderer.setVerticalGroup(
			gl_panelRenderer.createParallelGroup(Alignment.LEADING)
				.addGroup(gl_panelRenderer.createSequentialGroup()
					.addContainerGap()
					.addGroup(gl_panelRenderer.createParallelGroup(Alignment.BASELINE)
						.addComponent(comboBoxRenderer, GroupLayout.PREFERRED_SIZE, GroupLayout.DEFAULT_SIZE, GroupLayout.PREFERRED_SIZE)
						.addComponent(lblRenderer))
					.addPreferredGap(ComponentPlacement.RELATED)
					.addGroup(gl_panelRenderer.createParallelGroup(Alignment.TRAILING)
						.addComponent(chkboxFilter)
						.addComponent(btnReset))
					.addContainerGap(GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
		);
		panelRenderer.setLayout(gl_panelRenderer);
		
		panelFilter = new JPanel();
		GridBagConstraints gbc_panelFilter = new GridBagConstraints();
		gbc_panelFilter.fill = GridBagConstraints.BOTH;
		gbc_panelFilter.gridx = 0;
		gbc_panelFilter.gridy = 1;
		panelRecon.add(panelFilter, gbc_panelFilter);
		cardsSecond = new CardLayout(0, 0);
		panelFilter.setLayout(cardsSecond);
		
		ConfigurationPanel panelSecond = new ConfigurationPanel(){
			@Override
			public void setSettings(Map<String, Object> settings) {
			}

			@Override
			public Map<String, Object> getSettings() {
				return null;
			}
		};
		panelFilter.add(panelSecond, "SECOND");
		
		spinnerSkipFrames.setVisible(false);
		lblSkipFrames.setVisible(false);
		GridBagConstraints gbc_tabbedPane = new GridBagConstraints();
		gbc_tabbedPane.anchor = GridBagConstraints.NORTHWEST;
		gbc_tabbedPane.fill = GridBagConstraints.VERTICAL;
		gbc_tabbedPane.gridx = 0;
		gbc_tabbedPane.gridy = 0;
		contentPane.add(tabbedPane, gbc_tabbedPane);
		
		JPanel panelButtons = new JPanel();
		panelButtons.setBorder(null);
		
		btnProcess = new JButton("Process");
		btnProcess.addActionListener(this);
		
		btnLoad = new JButton("Load");
		btnLoad.addActionListener(this);
		panelButtons.setLayout(new FlowLayout(FlowLayout.CENTER, 5, 5));
		panelButtons.add(btnLoad);
		
		btnSave = new JButton("Save");
		btnSave.addActionListener(this);
		
		progressBar = new JProgressBar();
		progressBar.setStringPainted(true);
		GridBagConstraints gbc_progressBar = new GridBagConstraints();
		gbc_progressBar.fill = GridBagConstraints.HORIZONTAL;
		gbc_progressBar.insets = new Insets(0, 10, 0, 10);
		gbc_progressBar.gridx = 0;
		gbc_progressBar.gridy = 1;
		contentPane.add(progressBar, gbc_progressBar);
		panelButtons.add(btnSave);
		panelButtons.add(btnProcess);
		GridBagConstraints gbc_panelButtons = new GridBagConstraints();
		gbc_panelButtons.anchor = GridBagConstraints.NORTH;
		gbc_panelButtons.gridx = 0;
		gbc_panelButtons.gridy = 2;
		contentPane.add(panelButtons, gbc_panelButtons);
		init();
	}
	
	private void init(){
		settings = new HashMap<>();
		manager = new Manager();
		manager.addPropertyChangeListener(new PropertyChangeListener(){
			@Override
			public void propertyChange(PropertyChangeEvent evt) {
				if (evt.getPropertyName().equals("progress")){
					int value = (int) evt.getNewValue();
					progressBar.setValue(value);
				}
			}});
		createDetectorProvider();
		createPreProcessingProvider();
		createFitterProvider();
		createRendererProvider();
		createActionProvider();
		cardsFirst.show(panelLower, "FIRST");
		cardsSecond.show(panelFilter, "SECOND");
//		try {
//			saveFile = File.createTempFile("Lemming", ".tmp");
//		} catch (IOException e) {
//			IJ.error(e.getMessage());
//		}
		
	}

////Overrides
	
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
			chooseDetector();
		}
		
		if (s == this.comboBoxFitter){
			chooseFitter();
		}
		
		if (s == this.comboBoxRenderer){
			chooseRenderer();
		}
		
		if (s == this.chkboxFilter){
			if (this.chkboxFilter.isSelected())
				filterTable();
			validate();
		}
		
		if (s == this.btnReset){ 
			if (rendererFactory!=null){
				Map<String, Object> initialMap = ((RendererPanel) rendererFactory.getConfigurationPanel()).getInitialSettings();
				rendererShow(initialMap);
			}
		}
		
		if (s == this.btnProcess){ 			
			process(true);		
		}
		
		if (s == this.btnLoad){
			if (tabbedPane.getSelectedIndex()==0)
				loadImages();
			else
				loadLocalizations();
		}
		
		if (s == this.btnSave){
			if (!processed) {
				// TODO
			}
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

	private void process(boolean b) {
		// Manager

		if (tif == null) {
			IJ.error("Please load images first!");
			return;
		}

		if (detector == null) {
			IJ.error("Please choose detector first!");
			return;
		}
		manager.add(detector);

		ImageMath<T,F> math = null;
		if (!checksPreprocessing.isEmpty()) {
			AbstractModule pp = preProcessingFactory.getModule();
			manager.add(pp);
			manager.linkModules(tif, pp);
			operators op = preProcessingFactory.getOperator();
			math = new ImageMath<>();
			math.setOperator(op);
			manager.add(math);
			manager.linkModules(tif, math);
			manager.linkModules(pp, math);
			manager.linkModules(math, detector);
		} else {
			manager.linkModules(tif, detector);
		}
		
		if (fitter != null) {
			manager.add(fitter);
			manager.linkModules(detector, fitter);
		}
		if (b) {
			if (renderer != null) {
				manager.add(renderer);
				manager.linkModules(fitter, renderer,false);
			}
		
			manager.execute();
			processed = true;
		}
	}

	@Override
	public void addCheck(ListEvent event) {
		ListCheckModel source = event.getSource();
		if (source == this.jComboBoxPreprocessing.getModel()){
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
			if (selectedKey==null || tif == null) return;
			
			System.out.println("preProcessing: " + selected);
			preProcessingFactory = preProcessingProvider.getFactory(selectedKey);
			ConfigurationPanel panelDown = preProcessingFactory.getConfigurationPanel();
			
			panelDown.addPropertyChangeListener(ConfigurationPanel.propertyName, new PropertyChangeListener(){

				@SuppressWarnings("unchecked")
				@Override
				public void propertyChange(PropertyChangeEvent evt) {
					Map<String, Object> value = (Map<String, Object>) evt.getNewValue();
					ppPreview(value);
				}
			});
			validate();
		}
	}

	@Override
	public void removeCheck(ListEvent event) {
		ListCheckModel source = event.getSource();
		if (source == this.jComboBoxPreprocessing.getModel()){
			List<Object> checks = jComboBoxPreprocessing.getModel().getCheckeds();
			String removalString = null;
			for (String s : checksPreprocessing){
				if(!checks.contains(s))
					removalString = s;
			}
			if (removalString!=null)
				checksPreprocessing.remove(removalString);	
			this.repaint();
		}
	}
	
	//// Private Methods
	
	@SuppressWarnings("unchecked")
	private void ppPreview(Map<String, Object> map) {
		preProcessingFactory.setAndCheckSettings(map);
		AbstractModule preProcessor = detectorFactory.getDetector();
		int frameNumber = previewerWindow.getImagePlus().getSlice();
		List<Element> list = new ArrayList<>();
		
		for (int i = frameNumber; i < frameNumber + preProcessingFactory.processingFrames(); i++){
			ImageProcessor ip = previewerWindow.getImagePlus().getStack().getProcessor(i);
			Img<T> curImage = LemmingUtils.wrap(ip);
			Frame<T> curFrame = new ImgLib2Frame<>(frameNumber, (int)curImage.dimension(0), (int)curImage.dimension(1), curImage);
			list.add(curFrame);
		}
		
		FastStore ppResults = new FastStore();;
		if (!list.isEmpty()){
			FastStore previewStore = new FastStore();
	 		Element last = list.remove(list.size()-1);
			for (Element entry : list)	
				previewStore.put(entry);
			last.setLast(true);
			previewStore.put(last);
			preProcessor.setInput(previewStore);
			preProcessor.setOutput(ppResults);
			Thread preProcessingThread = new Thread(preProcessor,preProcessor.getClass().getSimpleName());
			preProcessingThread.start();
			try {
				preProcessingThread.join();
			} catch (InterruptedException e1) {
				e1.printStackTrace();
			}
		}
		rendererWindow.repaint();
		
		if(ppResults.isEmpty()) return;
		
		for (int i = frameNumber; i < frameNumber + preProcessingFactory.processingFrames(); i++){
			Frame<T> resFrame = (Frame<T>) ppResults.get();
			if (resFrame == null) continue;
			ImageProcessor ip = ImageJFunctions.wrap(resFrame.getPixels(), "").getProcessor();
			previewerWindow.getImagePlus().getStack().setProcessor(ip, i);
		}
	}
	
	private void loadImages() {
	    manager.reset();
		ImagePlus loc_im = WindowManager.getCurrentImage();
		
		if (loc_im==null) {	
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
	        	loc_im = new ImagePlus(file.getName(),fivs);
	        }
		}
        if (loc_im !=null){
		    tif = new ImageLoader<>(loc_im);
		    manager.add(tif);
		    
		    previewerWindow = new StackWindow(loc_im,loc_im.getCanvas());
		    previewerWindow.addKeyListener(new KeyAdapter(){

				@Override
				public void keyPressed(KeyEvent e) {
					if (e.getKeyChar() == 'C'){
						contrastAdjuster = new ContrastAdjuster();
						contrastAdjuster.run("B&C");
					}
				}
			});
		    previewerWindow.setVisible(true);
		    lblFile.setText(loc_im.getTitle());
		    validate();
		    repaint();
        }
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

    	TableLoader tl = new TableLoader(file);
    	tl.readCSV(',');
    	table = tl.getTable();
		
		lblFile.setText(file.getName());
		processed = true;
	}
	
	private static ConfigurationPanel getConfigSettings(JPanel cardPanel){
		ConfigurationPanel s = null;
		for (Component comp : cardPanel.getComponents()) {
	           if (comp.isVisible()) 
	                s = ((ConfigurationPanel) comp);
		}
		return s;
	}

	private void saveSettings(JPanel cardPanel){
	if (cardPanel == null) return;
	for (Component comp : cardPanel.getComponents()) {
		Map<String, Object> s = ((ConfigurationPanel) comp).getSettings();
		if (s!=null)
			for (String key : s.keySet())
				settings.put(key, s.get(key));
			}	
	}
	
	private void chooseDetector(){
		final int index = comboBoxPeakDet.getSelectedIndex() - 1;
		if (index<0 || tif == null){
			if (previewerWindow != null)
				previewerWindow.getImagePlus().killRoi();
			detector = null;
			widgetSelection = 0;
			return;
		}
		widgetSelection = DETECTOR;
		
		final String key = detectorProvider.getVisibleKeys().get( index );
		detectorFactory = detectorProvider.getFactory(key);
		cardsFirst.show(panelLower, key);
		validate();
		System.out.println("Detector_"+index+" : "+key);
		
		ConfigurationPanel panelDown = getConfigSettings(panelLower);
		panelDown.addPropertyChangeListener(ConfigurationPanel.propertyName, new PropertyChangeListener(){

			@SuppressWarnings("unchecked")
			@Override
			public void propertyChange(PropertyChangeEvent evt) {
				Map<String, Object> value = (Map<String, Object>) evt.getNewValue();
				detectorPreview(value);
			}
		});
		
		ImagePlus.addImageListener(new ImageListener(){

			@Override
			public void imageClosed(ImagePlus ip) {	}

			@Override
			public void imageOpened(ImagePlus ip) {	}

			@Override
			public void imageUpdated(ImagePlus ip) {
				if (ip == previewerWindow.getImagePlus() && widgetSelection==DETECTOR)
					detectorPreview(getConfigSettings(panelLower).getSettings());
				if (ip == previewerWindow.getImagePlus() && widgetSelection==FITTER)
					fitterPreview(getConfigSettings(panelLower).getSettings());
			}
		});		
		detectorPreview(panelDown.getSettings());
	}
	
	@SuppressWarnings("unchecked")
	private void detectorPreview(Map<String, Object> map){
		detectorFactory.setAndCheckSettings(map);
		detector = detectorFactory.getDetector();
		int frameNumber = previewerWindow.getImagePlus().getSlice();
		Img<T> curImage = LemmingUtils.wrap(previewerWindow.getImagePlus().getStack().getProcessor(frameNumber));
		ImgLib2Frame<T> curFrame = new ImgLib2Frame<>(frameNumber, (int)curImage.dimension(0), (int)curImage.dimension(1), curImage);
		
		detResults = (FrameElements<T>) detector.preview(curFrame);
		FloatPolygon points = LemmingUtils.convertToPoints(detResults.getList());
		PointRoi roi = new PointRoi(points);
		previewerWindow.getImagePlus().setRoi(roi);
	}
	
	private void chooseFitter() {
		final int index = comboBoxFitter.getSelectedIndex() - 1;
		if (index<0 || tif == null || detector == null){ 
			fitter = null;
			return;
		}
		widgetSelection = FITTER;
		
		final String key = fitterProvider.getVisibleKeys().get( index );
		fitterFactory = fitterProvider.getFactory(key);
		cardsFirst.show(panelLower, key);
		validate();
		System.out.println("Fitter_"+index+" : "+key);
		ConfigurationPanel panelDown = getConfigSettings(panelLower);
		panelDown.addPropertyChangeListener(ConfigurationPanel.propertyName, new PropertyChangeListener(){
			@SuppressWarnings("unchecked")
			@Override
			public void propertyChange(PropertyChangeEvent evt) {
				Map<String, Object> value = (Map<String, Object>) evt.getNewValue();
				fitterPreview(value);
			}
		});
				
		if (panelDown.getSettings() != null) 
			fitterPreview(panelDown.getSettings());
	}
	
	private void fitterPreview(Map<String, Object> map){
		if (!fitterFactory.setAndCheckSettings(map))
			return;
		fitter = fitterFactory.getFitter();
		
		previewerWindow.getImagePlus().killRoi();
		int frameNumber = previewerWindow.getImagePlus().getSlice();
		Img<T> curImage = LemmingUtils.wrap(previewerWindow.getImagePlus().getStack().getProcessor(frameNumber));
		ImgLib2Frame<T> curFrame = new ImgLib2Frame<>(frameNumber, (int)curImage.dimension(0), (int)curImage.dimension(1), curImage);
		
		fitResults = fitter.fit(detResults.getList(), curFrame.getPixels(), fitter.getWindowSize(), frameNumber);
		FloatPolygon points = LemmingUtils.convertToPoints(fitResults);
		PointRoi roi = new PointRoi(points);
		previewerWindow.getImagePlus().setRoi(roi);
		previewerWindow.repaint();
	}
	
	private void chooseRenderer() { 
		final int index = comboBoxRenderer.getSelectedIndex() - 1;
		if (index<0) return;
		
		final String key = rendererProvider.getVisibleKeys().get( index );
		rendererFactory = rendererProvider.getFactory(key);
		System.out.println("Renderer_"+index+" : "+key);
		cardsSecond.show(panelFilter, key);
		validate();
		ConfigurationPanel panelReconDown = getConfigSettings(panelFilter);
		panelReconDown.addPropertyChangeListener(ConfigurationPanel.propertyName, new PropertyChangeListener(){

			@SuppressWarnings("unchecked")
			@Override
			public void propertyChange(PropertyChangeEvent evt) {
				Map<String, Object> map = (Map<String, Object>) evt.getNewValue();
				rendererPreview(map);
			}
		});
		
		rendererFactory.setAndCheckSettings(panelReconDown.getSettings());
		renderer = rendererFactory.getRenderer();
		
		if(rendererWindow == null){
			rendererWindow = new ImageWindow(renderer.getImage()); // set a new Renderer
			rendererWindow.addKeyListener(new KeyAdapter(){
		
				@Override
				public void keyPressed(KeyEvent e) {
					if (e.getKeyChar() == 'C'){
						contrastAdjuster = new ContrastAdjuster();
						contrastAdjuster.run("B&C");
					}
				}
			});
		}
		if (processed)
			rendererShow(settings);
		else
			rendererPreview(settings);
	}
	
	private void rendererPreview(Map<String, Object> map){
		rendererFactory.setAndCheckSettings(map);
		
		rendererWindow.getCanvas().fitToWindow();
		if (!rendererWindow.isVisible())
			rendererWindow.setVisible(true);
		
		List<Element> list = fitResults;
		if (list != null && !list.isEmpty()){
			renderer.preview(list);
		}
	}
	
	private void rendererShow(Map<String, Object> map){	
		rendererFactory.setAndCheckSettings(map);
		renderer = rendererFactory.getRenderer();
		rendererWindow.setImage(renderer.getImage());
		rendererWindow.getCanvas().fitToWindow();
		ExtendableTable tableToRender = filteredTable == null ? table : filteredTable;	
		
		rendererWindow.getCanvas().addMouseListener(new MouseAdapter(){
			@Override
		    public void mouseClicked(MouseEvent e){
		        if(e.getClickCount()==2){
		        	try{
		        		Rectangle rect = renderer.getImage().getRoi().getBounds();
		        	
			        	final double xmin = (double) settings.get(RendererFactory.KEY_xmin);
						final double xmax = (double) settings.get(RendererFactory.KEY_xmax);
						final double ymin = (double) settings.get(RendererFactory.KEY_ymin);
						final double ymax = (double) settings.get(RendererFactory.KEY_ymax);
						final int xbins = (int) settings.get(RendererFactory.KEY_xBins);
						final int ybins = (int) settings.get(RendererFactory.KEY_yBins);
						
						double new_xmin = (xmax-xmin)*rect.getMinX()/xbins+xmin;
						double new_ymin = (ymax-ymin)*rect.getMinY()/ybins+ymin;
						double new_xmax = (xmax-xmin)*rect.getMaxX()/xbins+xmin;
						double new_ymax = (ymax-ymin)*rect.getMaxY()/ybins+ymin;
						double factx = rect.getWidth()/rect.getHeight();
						double facty = rect.getHeight()/rect.getWidth();
						double ar = Math.min(factx, facty);
						int new_xbins =  (int)(Math.round(xbins*ar));
						int new_ybins =  (int)(Math.round(ybins*ar));
						
			        	settings.put(RendererFactory.KEY_xmin,new_xmin);
			        	settings.put(RendererFactory.KEY_ymin,new_ymin);
			        	settings.put(RendererFactory.KEY_xmax,new_xmax);
			        	settings.put(RendererFactory.KEY_ymax,new_ymax);
			        	settings.put(RendererFactory.KEY_xBins,new_xbins);
			        	settings.put(RendererFactory.KEY_yBins,new_ybins);
			        	
		        	} catch (NullPointerException ne){}
		        	rendererShow(settings);
		        }
			}
		});
		if (tableToRender != null && tableToRender.columnNames().size()>0){
			Store previewStore = tableToRender.getFIFO();
			System.out.println("Rendering " + tableToRender.getNumberOfRows() + " elements");
			renderer.setInput(previewStore);
			renderer.run();		
			rendererWindow.repaint();
		}
	}
	
	private void filterTable() {
		// TODO
		if (!processed){
			if (IJ.showMessageWithCancel("Filter", "Pipeline not yet processed.\nDo you want to process it now?"))
				process(false);
			else
				return;
			if(fitter == null) return;
			DataTable dt = new DataTable();
			manager.add(dt);
			manager.linkModules(fitter, dt);
			manager.execute();
			processed = true;
			table = dt.getTable();
		}
		if (table!=null || !table.columnNames().isEmpty()){
			FilterPanel panelReconDown = new FilterPanel(table);
			cardsSecond.addLayoutComponent(panelReconDown, "FILTER");
			panelReconDown.addPropertyChangeListener(ConfigurationPanel.propertyName, new PropertyChangeListener(){
				@Override
				public void propertyChange(PropertyChangeEvent evt) {
					if (table.filtersCollection.isEmpty()){
						filteredTable = null;
					} else {
						filteredTable = table.filter();
					}
					if (renderer != null)
						rendererShow(settings);
				}
			});
			Map<String, Object> curSet = panelReconDown.getSettings();
			if (curSet != null)
				for (String key : curSet.keySet())
					settings.put(key, curSet.get(key));
			if (renderer != null)
				rendererShow(settings);
			validate();
		}		
	}
	
	private void createDetectorProvider(){
		detectorProvider = new DetectorProvider();
		final List< String > visibleKeys = detectorProvider.getVisibleKeys();
		final List< String > detectorNames = new ArrayList<>();
		final List< String > infoTexts = new ArrayList<>();
		detectorNames.add("none");
		for ( final String key : visibleKeys ){
			DetectorFactory factory = detectorProvider.getFactory( key );
			detectorNames.add( detectorProvider.getFactory( key ).getName() );
			infoTexts.add( detectorProvider.getFactory( key ).getInfoText() );
			panelLower.add(factory.getConfigurationPanel(),key);
		}
		String[] names = detectorNames.toArray(new String[] {});
		comboBoxPeakDet.setModel(new DefaultComboBoxModel<>(names));
		comboBoxPeakDet.setRenderer(new ToolTipRenderer(infoTexts));
	}
	
	private void createPreProcessingProvider() {
		preProcessingProvider = new PreProcessingProvider();
		checksPreprocessing = new ArrayList<>();
		jComboBoxPreprocessing.setTextFor(CheckComboBox.NONE, "none"); 
		jComboBoxPreprocessing.setTextFor(CheckComboBox.MULTIPLE, "multiple"); 
		jComboBoxPreprocessing.setTextFor(CheckComboBox.ALL, "all");
		ListCheckModel model = jComboBoxPreprocessing.getModel();
		final List< String > visibleKeys = preProcessingProvider.getVisibleKeys();
		final List< String > infoTexts = new ArrayList<>();
		for ( final String key : visibleKeys ){
			PreProcessingFactory factory = preProcessingProvider.getFactory( key );
			String currentName = factory.getName();
			infoTexts.add( factory.getInfoText() );
			model.addElement(currentName);
			panelLower.add(factory.getConfigurationPanel(),key);
		}
		jComboBoxPreprocessing.setRenderer(new ToolTipCheckListRenderer(infoTexts));
	}
	
	private void createFitterProvider() {
		fitterProvider = new FitterProvider();
		final List< String > visibleKeys = fitterProvider.getVisibleKeys();
		final List< String > fitterNames = new ArrayList<>();
		final List< String > infoTexts = new ArrayList<>();
		fitterNames.add("none");
		for ( final String key : visibleKeys ){
			FitterFactory factory = fitterProvider.getFactory( key );
			fitterNames.add( factory.getName() );
			infoTexts.add( factory.getInfoText() );
			panelLower.add(factory.getConfigurationPanel(),key);
		}
		String[] names = fitterNames.toArray(new String[] {});
		comboBoxFitter.setModel(new DefaultComboBoxModel<>(names));
		comboBoxFitter.setRenderer(new ToolTipRenderer(infoTexts));
	}
	
	private void createRendererProvider() {
		rendererProvider = new RendererProvider();
		final List< String > visibleKeys = rendererProvider.getVisibleKeys();
		final List< String > rendererNames = new ArrayList<>();
		final List< String > infoTexts = new ArrayList<>();
		rendererNames.add("none");
		for ( final String key : visibleKeys ){
			RendererFactory factory = rendererProvider.getFactory( key );
			rendererNames.add( factory.getName() );
			infoTexts.add( factory.getInfoText() );
			panelFilter.add(factory.getConfigurationPanel(),key);
		}
		String[] names = rendererNames.toArray(new String[] {});
		comboBoxRenderer.setModel(new DefaultComboBoxModel<>(names));
		comboBoxRenderer.setRenderer(new ToolTipRenderer(infoTexts));
		panelFilter.add(new FilterPanel(table),"FILTER");
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
	
	private class ToolTipCheckListRenderer extends CheckListRenderer{
		private static final long serialVersionUID = 1L;
		
		List<String> tooltips;
		
		public ToolTipCheckListRenderer(List<String> tooltips){
			 this.tooltips = tooltips;
		}
		
		@SuppressWarnings("rawtypes")
		@Override
		public Component getListCellRendererComponent(JList list, Object value, int index, boolean isSelected, boolean cellHasFocus){
			
			JComponent comp = (JComponent) super.getListCellRendererComponent(list, value, index, isSelected, cellHasFocus);

	        if (0 < index && null != value && null != tooltips) 
	                    list.setToolTipText(tooltips.get(index-1));
	        return comp;
		}
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

	        if (0 < index && null != value && null != tooltips) 
	                    list.setToolTipText(tooltips.get(index-1));
	        return comp;
	    }
		
	}

}
