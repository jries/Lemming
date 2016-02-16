package org.lemming.gui;

import ij.IJ;
import ij.ImageListener;
import ij.ImagePlus;
import ij.ImageStack;
import ij.WindowManager;
import ij.gui.ImageWindow;
import ij.gui.PointRoi;
import ij.gui.Roi;
import ij.gui.StackWindow;
import ij.io.OpenDialog;
import ij.io.SaveDialog;
import ij.measure.Calibration;
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
import javax.swing.JLabel;
import javax.swing.JList;
import javax.swing.LayoutStyle.ComponentPlacement;
import javax.swing.SwingWorker.StateValue;
import javax.swing.DefaultListCellRenderer;
import javax.swing.JComboBox;
import javax.swing.DefaultComboBoxModel;
import javax.swing.JCheckBox;
import javax.swing.JSpinner;
import javax.swing.UIManager;
import javax.swing.UnsupportedLookAndFeelException;

import org.lemming.factories.DetectorFactory;
import org.lemming.factories.FitterFactory;
import org.lemming.factories.PreProcessingFactory;
import org.lemming.factories.RendererFactory;
import org.lemming.interfaces.Element;
import org.lemming.interfaces.Frame;
import org.lemming.interfaces.Store;
import org.lemming.math.CentroidFitterIP;
import org.lemming.modules.DataTable;
import org.lemming.modules.Detector;
import org.lemming.modules.Fitter;
import org.lemming.modules.ImageLoader;
import org.lemming.modules.ImageMath;
import org.lemming.modules.SaveLocalizations;
import org.lemming.modules.ImageMath.operators;
import org.lemming.modules.Renderer;
import org.lemming.modules.StoreSaver;
import org.lemming.modules.TableLoader;
import org.lemming.pipeline.AbstractModule;
import org.lemming.pipeline.ExtendableTable;
import org.lemming.pipeline.FrameElements;
import org.lemming.pipeline.ImgLib2Frame;
import org.lemming.pipeline.LinkedStore;
import org.lemming.pipeline.Manager;
import org.lemming.plugins.FastMedianFilter;
import org.lemming.providers.ActionProvider;
import org.lemming.providers.DetectorProvider;
import org.lemming.providers.FitterProvider;
import org.lemming.providers.PreProcessingProvider;
import org.lemming.providers.RendererProvider;
import org.lemming.tools.LemmingUtils;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.awt.Component;

import javax.swing.SpinnerNumberModel;

import java.awt.Dimension;
import java.awt.event.KeyAdapter;

import javax.swing.event.ChangeListener;
import javax.swing.event.ChangeEvent;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.beans.PropertyChangeEvent;
import java.beans.PropertyChangeListener;

import javax.swing.JProgressBar;

import java.awt.CardLayout;

import javax.swing.border.TitledBorder;
import javax.swing.border.LineBorder;

import java.awt.Color;

/**
 * The GUI main class controlling all user interactions
 * 
 * @author Ronny Sczech
 *
 * @param <T> - data type
 * @param <F> - frame type
 */
public class Controller<T extends NumericType<T> & NativeType<T> & RealType<T>> extends JFrame implements ActionListener {

	private static final long serialVersionUID = -2596199192028890712L;
	private JTabbedPane tabbedPane;
	private JPanel panelLoc;
	private JPanel panelRecon;
	private JCheckBox checkboxPP;
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
	private PreProcessingProvider preProcessingProvider;
	private PreProcessingFactory preProcessingFactory;
	private FitterFactory fitterFactory;
	private RendererFactory rendererFactory;
	private JLabel lblFile;
	private Manager manager;
	private ExtendableTable table;
	private ImageLoader<T> tif;
	private Map<String, Object> settings;
	private StackWindow previewerWindow;
	private Detector<T> detector;
	private Fitter<T> fitter;
	private ContrastAdjuster contrastAdjuster;
	private Renderer renderer;
	private FrameElements<T> detResults;
	private List<Element> fitResults;
	private ImageWindow rendererWindow;
	private int widgetSelection = 0;
	private boolean processed = false;
	private ExtendableTable filteredTable = null;
	private static int DETECTOR = 1;
	private static int FITTER = 2;
	private JProgressBar progressBar;
	private JButton btnReset;
	private JPanel panelLower;
	private JPanel panelFilter;
	private Locale curLocale;
	private JPanel panelProgress;
	private JLabel lblEta;
	private long start;
	private AbstractModule saver;
	public static String lastDir = System.getProperty("user.home"); 

	/**
	 * Create the frame.
	 * 
	 * @param imp
	 */
	public Controller() {
		addWindowListener(new WindowAdapter() {
			@Override
			public void windowClosing(WindowEvent e) {
				if (previewerWindow != null)
					previewerWindow.close();
				if (contrastAdjuster != null)
					contrastAdjuster.close();
				if (rendererWindow != null)
					rendererWindow.close();
				Locale.setDefault(curLocale);
			}
		});
		try {
			UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
		} catch (ClassNotFoundException | InstantiationException | IllegalAccessException | UnsupportedLookAndFeelException e1) {
			IJ.error(e1.getMessage());
		}
		setTitle("Lemming");
		setBounds(100, 100, 320, 500);
		JPanel contentPane = new JPanel();
		setContentPane(contentPane);

		GridBagLayout gbl_contentPane = new GridBagLayout();
		gbl_contentPane.columnWidths = new int[] { 315, 0 };
		gbl_contentPane.rowHeights = new int[] { 400, 0, 28, 0 };
		gbl_contentPane.columnWeights = new double[] { 1.0, Double.MIN_VALUE };
		gbl_contentPane.rowWeights = new double[] { 1.0, 0.0, 0.0, Double.MIN_VALUE };
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
		panelLoc.setBorder(null);
		tabbedPane.addTab("Localize", null, panelLoc, null);
		GridBagLayout gbl_panelLoc = new GridBagLayout();
		gbl_panelLoc.columnWidths = new int[] { 250 };
		gbl_panelLoc.rowHeights = new int[] {140, 25, 200};
		gbl_panelLoc.columnWeights = new double[] { 1.0 };
		gbl_panelLoc.rowWeights = new double[] { 0.0, 0.0, 1.0 };
		panelLoc.setLayout(gbl_panelLoc);

		JPanel panelUpper = new JPanel();

		checkboxPP = new JCheckBox("Background Substraction");
		checkboxPP.addActionListener(this);

		lblPeakDet = new JLabel("Peak Detector");
		GridBagConstraints gbc_panelUpper = new GridBagConstraints();
		gbc_panelUpper.fill = GridBagConstraints.HORIZONTAL;
		gbc_panelUpper.anchor = GridBagConstraints.NORTH;
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
					.addGroup(gl_panelUpper.createParallelGroup(Alignment.LEADING, false)
						.addComponent(checkboxPP)
						.addGroup(gl_panelUpper.createParallelGroup(Alignment.TRAILING)
							.addGroup(gl_panelUpper.createSequentialGroup()
								.addGroup(gl_panelUpper.createParallelGroup(Alignment.LEADING)
									.addComponent(lblPeakDet)
									.addComponent(lblFitter))
								.addPreferredGap(ComponentPlacement.RELATED)
								.addGroup(gl_panelUpper.createParallelGroup(Alignment.LEADING)
									.addComponent(comboBoxPeakDet, 0, 177, Short.MAX_VALUE)
									.addComponent(comboBoxFitter, 0, 177, Short.MAX_VALUE))
								.addPreferredGap(ComponentPlacement.RELATED))
							.addGroup(Alignment.LEADING, gl_panelUpper.createSequentialGroup()
								.addComponent(lblDataSource)
								.addPreferredGap(ComponentPlacement.UNRELATED)
								.addComponent(lblFile, GroupLayout.PREFERRED_SIZE, 187, GroupLayout.PREFERRED_SIZE))))
					.addGap(1))
		);
		gl_panelUpper.setVerticalGroup(
			gl_panelUpper.createParallelGroup(Alignment.LEADING)
				.addGroup(gl_panelUpper.createSequentialGroup()
					.addContainerGap()
					.addGroup(gl_panelUpper.createParallelGroup(Alignment.BASELINE)
						.addComponent(lblDataSource)
						.addComponent(lblFile))
					.addPreferredGap(ComponentPlacement.RELATED)
					.addComponent(checkboxPP, GroupLayout.PREFERRED_SIZE, 27, GroupLayout.PREFERRED_SIZE)
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
		gbc_panelMiddle.fill = GridBagConstraints.HORIZONTAL;
		gbc_panelMiddle.anchor = GridBagConstraints.NORTH;
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
		gl_panelMiddle.setHorizontalGroup(gl_panelMiddle.createParallelGroup(Alignment.LEADING)
				.addGroup(gl_panelMiddle.createSequentialGroup().addContainerGap().addComponent(chckbxROI)
						.addPreferredGap(ComponentPlacement.UNRELATED).addComponent(lblSkipFrames).addGap(18)
						.addComponent(spinnerSkipFrames, GroupLayout.PREFERRED_SIZE, 63, GroupLayout.PREFERRED_SIZE).addContainerGap()));
		gl_panelMiddle
				.setVerticalGroup(
						gl_panelMiddle.createParallelGroup(Alignment.LEADING)
								.addGroup(gl_panelMiddle
										.createParallelGroup(Alignment.BASELINE).addComponent(spinnerSkipFrames, GroupLayout.PREFERRED_SIZE,
												GroupLayout.DEFAULT_SIZE, GroupLayout.PREFERRED_SIZE)
										.addComponent(lblSkipFrames).addComponent(chckbxROI)));
		panelMiddle.setLayout(gl_panelMiddle);

		panelLower = new JPanel();
		panelLower.setBorder(new TitledBorder(new LineBorder(new Color(0, 0, 205)), "none", 
				TitledBorder.LEADING, TitledBorder.TOP, null, new Color(0, 0, 205)));
		
		GridBagConstraints gbc_panelLower = new GridBagConstraints();
		gbc_panelLower.fill = GridBagConstraints.BOTH;
		gbc_panelLower.gridx = 0;
		gbc_panelLower.gridy = 2;
		panelLoc.add(panelLower, gbc_panelLower);
		panelLower.setLayout(new CardLayout(0, 0));

		panelRecon = new JPanel();
		panelRecon.setBorder(null);
		tabbedPane.addTab("Reconstruct", null, panelRecon, null);
		GridBagLayout gbl_panelRecon = new GridBagLayout();
		gbl_panelRecon.columnWidths = new int[] { 300 };
		gbl_panelRecon.rowHeights = new int[] {75, 305};
		gbl_panelRecon.columnWeights = new double[] { 1.0 };
		gbl_panelRecon.rowWeights = new double[] { 0.0, 1.0 };
		panelRecon.setLayout(gbl_panelRecon);

		JPanel panelRenderer = new JPanel();
		panelRenderer.setBorder(null);
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
		gl_panelRenderer.setHorizontalGroup(gl_panelRenderer.createParallelGroup(Alignment.LEADING)
				.addGroup(gl_panelRenderer.createSequentialGroup().addContainerGap()
						.addGroup(gl_panelRenderer.createParallelGroup(Alignment.LEADING)
								.addGroup(gl_panelRenderer.createSequentialGroup().addComponent(lblRenderer).addGap(39).addComponent(comboBoxRenderer,
										GroupLayout.PREFERRED_SIZE, 165, GroupLayout.PREFERRED_SIZE))
						.addGroup(gl_panelRenderer.createSequentialGroup().addComponent(chkboxFilter)
								.addPreferredGap(ComponentPlacement.RELATED, 117, Short.MAX_VALUE).addComponent(btnReset)))));
		gl_panelRenderer
				.setVerticalGroup(gl_panelRenderer.createParallelGroup(Alignment.LEADING)
						.addGroup(gl_panelRenderer.createSequentialGroup().addContainerGap()
								.addGroup(gl_panelRenderer.createParallelGroup(Alignment.BASELINE)
										.addComponent(comboBoxRenderer, GroupLayout.PREFERRED_SIZE, GroupLayout.DEFAULT_SIZE,
												GroupLayout.PREFERRED_SIZE)
										.addComponent(lblRenderer))
								.addPreferredGap(ComponentPlacement.RELATED)
								.addGroup(gl_panelRenderer.createParallelGroup(Alignment.TRAILING).addComponent(chkboxFilter).addComponent(btnReset))
								.addContainerGap(GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)));
		panelRenderer.setLayout(gl_panelRenderer);

		panelFilter = new JPanel();
		panelFilter.setBorder(new TitledBorder(new LineBorder(new Color(0, 0, 205)), "none", 
				TitledBorder.LEADING, TitledBorder.TOP, null, new Color(0, 0, 205)));
		GridBagConstraints gbc_panelFilter = new GridBagConstraints();
		gbc_panelFilter.fill = GridBagConstraints.BOTH;
		gbc_panelFilter.gridx = 0;
		gbc_panelFilter.gridy = 1;
		panelRecon.add(panelFilter, gbc_panelFilter);
		panelFilter.setLayout(new CardLayout(0, 0));

		GridBagConstraints gbc_tabbedPane = new GridBagConstraints();
		gbc_tabbedPane.insets = new Insets(0, 0, 5, 0);
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
		
		panelProgress = new JPanel();
		GridBagConstraints gbc_panelProgress = new GridBagConstraints();
		gbc_panelProgress.fill = GridBagConstraints.BOTH;
		gbc_panelProgress.insets = new Insets(0, 0, 5, 0);
		gbc_panelProgress.gridx = 0;
		gbc_panelProgress.gridy = 1;
		contentPane.add(panelProgress, gbc_panelProgress);
		GridBagLayout gbl_panelProgress = new GridBagLayout();
		gbl_panelProgress.columnWidths = new int[]{315, 0, 0};
		gbl_panelProgress.rowHeights = new int[]{0, 0};
		gbl_panelProgress.columnWeights = new double[]{1.0, 0.0, Double.MIN_VALUE};
		gbl_panelProgress.rowWeights = new double[]{0.0, Double.MIN_VALUE};
		panelProgress.setLayout(gbl_panelProgress);

		progressBar = new JProgressBar();
		GridBagConstraints gbc_progressBar = new GridBagConstraints();
		gbc_progressBar.insets = new Insets(1, 10, 1, 10);
		gbc_progressBar.fill = GridBagConstraints.HORIZONTAL;
		gbc_progressBar.gridx = 0;
		gbc_progressBar.gridy = 0;
		panelProgress.add(progressBar, gbc_progressBar);
		progressBar.setStringPainted(true);
		
		lblEta = new JLabel("   0 sec");
		GridBagConstraints gbc_lblEta = new GridBagConstraints();
		gbc_lblEta.insets = new Insets(0, 0, 0, 10);
		gbc_lblEta.gridx = 1;
		gbc_lblEta.gridy = 0;
		panelProgress.add(lblEta, gbc_lblEta);
		panelButtons.add(btnSave);
		panelButtons.add(btnProcess);
		GridBagConstraints gbc_panelButtons = new GridBagConstraints();
		gbc_panelButtons.anchor = GridBagConstraints.SOUTH;
		gbc_panelButtons.gridx = 0;
		gbc_panelButtons.gridy = 2;
		contentPane.add(panelButtons, gbc_panelButtons);
		init();
	}

	private void init() {
		this.curLocale = Locale.getDefault();
		final Locale usLocale = new Locale("en", "US"); // setting us locale
		Locale.setDefault(usLocale);
		settings = new HashMap<>();
		table = new ExtendableTable();
		manager = new Manager();
		manager.addPropertyChangeListener(new PropertyChangeListener() {
			@Override
			public void propertyChange(PropertyChangeEvent evt) {
				if (evt.getPropertyName().equals("progress")) {
					int value = (Integer) evt.getNewValue();
					progressBar.setValue(value);
					long current = System.currentTimeMillis();
					long eta = 0;
					if (value > 0)
						eta = Math.round((current - start) * value / 1000);
					lblEta.setText(String.valueOf(eta) + "sec");
					start = current;
				}
				if (evt.getPropertyName().equals("state")) {
					StateValue value = (StateValue) evt.getNewValue();
					if (value == StateValue.DONE) {
						processed = true;
						if (rendererWindow != null) rendererWindow.repaint();
					}
				}
			}
		});
		createInitialPanels();
		createDetectorProvider();
		createPreProcessingProvider();
		createFitterProvider();
		createRendererProvider();
		createActionProvider();
		ImagePlus.addImageListener(new ImageListener() {

			@Override
			public void imageClosed(ImagePlus ip) {
			}

			@Override
			public void imageOpened(ImagePlus ip) {
			}

			@Override
			public void imageUpdated(ImagePlus ip) {
				if (widgetSelection == 0) return;
				else
				if (widgetSelection == DETECTOR && ip == previewerWindow.getImagePlus() )
					detectorPreview(getConfigSettings(panelLower).getSettings());
				else
				if (widgetSelection == FITTER && ip == previewerWindow.getImagePlus() )
					fitterPreview();
			}
		});
		OpenDialog.setDefaultDirectory(lastDir);
	}

	//// Overrides

	@Override
	public void actionPerformed(ActionEvent e) {
		Object s = e.getSource();

		if (s == this.chckbxROI) {
			setRoi();
		}

		if (s == this.comboBoxPeakDet) {
			chooseDetector();
		}

		if (s == this.comboBoxFitter) {
			chooseFitter();
		}

		if (s == this.comboBoxRenderer) {
			chooseRenderer();
		}

		if (s == this.checkboxPP) {
			if (checkboxPP.isSelected())
				choosePP();
			else
				((CardLayout) panelLower.getLayout()).first(panelLower);
		}

		if (s == this.chkboxFilter) {
			if (this.chkboxFilter.isSelected())
				filterTable();
		}

		if (s == this.btnReset) {
			if (rendererFactory != null) {
				Map<String, Object> initialMap = rendererFactory.getInitialSettings();
				if (previewerWindow != null) {
					initialMap.put(RendererFactory.KEY_xmax, previewerWindow.getImagePlus().getWidth()
							* previewerWindow.getImagePlus().getCalibration().pixelDepth);
					initialMap.put(RendererFactory.KEY_ymax, previewerWindow.getImagePlus().getHeight()
							* previewerWindow.getImagePlus().getCalibration().pixelDepth);
				}
				filteredTable = null;
				settings.putAll(initialMap);
				rendererShow(settings);
			}
		}

		if (s == this.btnProcess) {
			process(true);
		}

		if (s == this.btnLoad) {
			if (tabbedPane.getSelectedIndex() == 0)
				loadImages();
			else
				loadLocalizations();
		}

		if (s == this.btnSave) {
			saveLocalizations();
		}
	}

	private void process(boolean b) {
		// Manager
		final int elements = previewerWindow != null ? previewerWindow.getImagePlus().getStackSize() : 100;
		if (tif == null) {
			IJ.error("Please load images first!");
			return;
		}

		if (detector == null) {
			IJ.error("Please choose detector first!");
			return;
		}
		manager.add(detector);

		ImageMath<T> math = null;
		if (checkboxPP.isSelected()) {
			AbstractModule pp = preProcessingFactory.getModule();
			manager.add(pp);
			manager.linkModules(tif, pp, true, elements);
			operators op = preProcessingFactory.getOperator();
			math = new ImageMath<>(elements);
			math.setOperator(op);
			manager.add(math);
			manager.linkModules(tif, math, true, elements);
			manager.linkModules(pp, math);
			manager.linkModules(math, detector);
		} else {
			manager.linkModules(tif, detector, true, elements);
		}

		if (fitter != null) {
			manager.add(fitter);
			manager.linkModules(detector, fitter);
			DataTable dt = new DataTable();
			manager.add(dt);
			manager.linkModules(fitter, dt);
			table = dt.getTable();
		}
		if (b) {
			if (saver == null) {
				saveLocalizations();
			}
		}
		if (renderer != null) {
			manager.add(renderer);
			manager.linkModules(fitter, renderer, false, elements);
		}
		start = System.currentTimeMillis();
		manager.execute();
	}

	//// Private Methods

	private void setRoi() {
		if (previewerWindow == null)
			return;
		final ImagePlus curImage = previewerWindow.getImagePlus();
		if (chckbxROI.isSelected()) {
			Roi roi = curImage.getRoi();
			if (roi == null) {
				final Rectangle r = curImage.getProcessor().getRoi();
				final int iWidth = r.width / 2;
				final int iHeight = r.height / 2;
				final int iXROI = r.x + r.width / 4;
				final int iYROI = r.y + r.height / 4;
				curImage.setRoi(iXROI, iYROI, iWidth, iHeight);
			}
		} else {
			curImage.killRoi();
		}
	}
	
	private void convertPhotons(ImageProcessor ip2, List<Double> cameraSettings){
		double offset = cameraSettings.get(0);
		double em_gain = cameraSettings.get(1);
		double conversion = cameraSettings.get(2);
		ip2.subtract(offset);
		ip2.multiply(conversion/em_gain);
	}

	private void loadImages() {
		manager.reset();
		ImagePlus loc_im = WindowManager.getCurrentImage();

		if (loc_im == null) {
			final OpenDialog od = new OpenDialog("Import Images");
			if(od.getFileName()==null) return;
			final File file = new File(od.getDirectory()+od.getFileName());
			lastDir = od.getDirectory();
			OpenDialog.setDefaultDirectory(lastDir);

			if (file.isDirectory()) {
				final FolderOpener fo = new FolderOpener();
				fo.openAsVirtualStack(true);
				loc_im = fo.openFolder(file.getAbsolutePath());
			}

			if (file.isFile()) {
				loc_im = FileInfoVirtualStack.openVirtual(file.getAbsolutePath());
				loc_im.setOpenAsHyperStack(true);
				final int[] dims = loc_im.getDimensions();
				if (dims[4] == 1 && dims[3] > 1) { // swap Z With T
					loc_im.setDimensions(dims[2], dims[4], dims[3]);
					final Calibration calibration = loc_im.getCalibration();
					calibration.frameInterval = 1;
					calibration.setTimeUnit("frame");
				}
			}
		}
		if (loc_im != null) {
			
			tif = new ImageLoader<>(loc_im, LemmingUtils.readCameraSettings("camera.props"));
			manager.add(tif);

			previewerWindow = new StackWindow(loc_im, loc_im.getCanvas());
			previewerWindow.addKeyListener(new KeyAdapter() {

				@Override
				public void keyPressed(KeyEvent e) {
					if (e.getKeyChar() == 'C') {
						contrastAdjuster = new ContrastAdjuster();
						contrastAdjuster.run("B&C");
					}
				}
			});
			previewerWindow.setVisible(true);
			lblFile.setText(loc_im.getTitle());
			validate();
		}
	}

	private void loadLocalizations() {
		manager.reset();
		final OpenDialog od = new OpenDialog("Import Data");
		if(od.getFileName()==null) return;
		File file = new File(od.getDirectory()+od.getFileName());
		lastDir = od.getDirectory();
		OpenDialog.setDefaultDirectory(lastDir);

		TableLoader tl = new TableLoader(file);
		tl.readCSV(',');
		table = new ExtendableTable(tl.getTable());

		lblFile.setText(file.getName());
		processed = true;
	}

	private static ConfigurationPanel getConfigSettings(JPanel cardPanel) {
		ConfigurationPanel s = null;
		for (Component comp : cardPanel.getComponents()) {
			if (comp.isVisible()){
				s = ((ConfigurationPanel) comp);
				break;
			}
		}
		return s;
	}

	private void saveSettings(JPanel cardPanel) {
		if (cardPanel == null)
			return;
		for (Component comp : cardPanel.getComponents()) {
			Map<String, Object> s = ((ConfigurationPanel) comp).getSettings();
			if (s != null)
				for (String key : s.keySet())
					settings.put(key, s.get(key));
		}
	}

	private void choosePP() {
		if (tif == null) {
			if (previewerWindow != null)
				previewerWindow.getImagePlus().killRoi();
			return;
		}
		preProcessingFactory = preProcessingProvider.getFactory(FastMedianFilter.KEY);
		((TitledBorder) panelLower.getBorder()).setTitle(preProcessingFactory.getName());
		((CardLayout) panelLower.getLayout()).show(panelLower, FastMedianFilter.KEY);
		ConfigurationPanel panelDown = getConfigSettings(panelLower);
		ppPreview(panelDown.getSettings());
	}

	@SuppressWarnings("unchecked")
	private void ppPreview(Map<String, Object> map) {
		preProcessingFactory.setAndCheckSettings(map);
		final AbstractModule preProcessor = preProcessingFactory.getModule();
		final int frameNumber = previewerWindow.getImagePlus().getSlice();
		final List<Element> list = new ArrayList<>();
		final ImageStack stack = previewerWindow.getImagePlus().getImageStack();
		final int stackSize = stack.getSize();
		final double pixelSize = previewerWindow.getImagePlus().getCalibration().pixelDepth;

		for (int i = frameNumber; i < frameNumber + preProcessingFactory.processingFrames(); i++) {
			if (i < stackSize) {
				Object ip = stack.getPixels(i);
				Img<T> curImage = LemmingUtils.wrap(ip, new long[]{stack.getWidth(), stack.getHeight()});
				Frame<T> curFrame = new ImgLib2Frame<>(frameNumber, (int) curImage.dimension(0), (int) curImage.dimension(1), pixelSize, curImage);
				list.add(curFrame);
			}
		}

		final Store ppResults = new LinkedStore(128);
		if (!list.isEmpty()) {
			final Store previewStore = new LinkedStore(128);
			final Element last = list.remove(list.size() - 1);
			try {
				for (Element entry : list)
					previewStore.put(entry);
				last.setLast(true);
				previewStore.put(last);
			} catch (InterruptedException e) {}
			preProcessor.setInput(previewStore);
			preProcessor.setOutput(ppResults);
			final ExecutorService service = Executors.newSingleThreadExecutor();
			Future<?> future = service.submit(preProcessor);
			try {
				future.get();
			} catch (InterruptedException | ExecutionException e1) {
				e1.printStackTrace();
			}
		}

		if (ppResults.isEmpty())
			return;

		for (int i = frameNumber; i < frameNumber + preProcessingFactory.processingFrames(); i++) {
			final Frame<T> resFrame = (Frame<T>) ppResults.poll();
			if (resFrame == null)
				continue;
			final ImageProcessor ip = ImageJFunctions.wrap(resFrame.getPixels(), "").getProcessor();
			if (i < stackSize) {
				final ImageProcessor stackip = stack.getProcessor(i);
				stackip.setPixels(ip.getPixels());
			}
		}
		preProcessor.reset();
		previewerWindow.repaint();
	}

	private void chooseDetector() {
		final int index = comboBoxPeakDet.getSelectedIndex() - 1;
		if (index < 0 || tif == null) {
			if (previewerWindow != null)
				previewerWindow.getImagePlus().killRoi();
			detector = null;
			widgetSelection = 0;
			((TitledBorder) panelLower.getBorder()).setTitle("none");
			((CardLayout) panelLower.getLayout()).first(panelLower);
			return;
		}
		widgetSelection = DETECTOR;

		final String key = detectorProvider.getVisibleKeys().get(index);
		detectorFactory = detectorProvider.getFactory(key);
		((TitledBorder) panelLower.getBorder()).setTitle(detectorFactory.getName());
		((CardLayout) panelLower.getLayout()).show(panelLower, key);
		System.out.println("Detector_" + index + " : " + key);

		ConfigurationPanel panelDown = getConfigSettings(panelLower);
		detectorPreview(panelDown.getSettings());
		settings.putAll(panelDown.getSettings());
		panelLower.repaint();
	}

	@SuppressWarnings("unchecked")
	private void detectorPreview(Map<String, Object> map) {
		// TODO constrain to current ROI
		settings.putAll(map);
		detectorFactory.setAndCheckSettings(map);
		detector = detectorFactory.getDetector();
		final ImagePlus img = previewerWindow.getImagePlus();
		final int frameNumber = img.getCurrentSlice();
		final double pixelSize = img.getCalibration().pixelDepth;
		Roi currentRoi = previewerWindow.getImagePlus().getRoi();
		ImageProcessor ip = img.getStack().getProcessor(frameNumber);
		if (currentRoi != null){
			ip.setRoi(currentRoi.getBounds());
			ip = ip.crop();
		} else{
			currentRoi = new Roi(0,0,ip.getWidth(),ip.getHeight());
		}
		
		convertPhotons(ip,LemmingUtils.readCameraSettings("camera.props"));

		Img<T> curImage = LemmingUtils.wrap(ip.getPixels(), new long[]{ip.getWidth(), ip.getHeight()});
		ImgLib2Frame<T> curFrame = new ImgLib2Frame<>(frameNumber, (int) curImage.dimension(0), (int) curImage.dimension(1), pixelSize, curImage);

		detResults = (FrameElements<T>) detector.preview(curFrame);
		if (detResults==null) return;
		final FloatPolygon points = LemmingUtils.convertToPoints(detResults.getList(), currentRoi.getBounds(), pixelSize);
		final PointRoi roi = new PointRoi(points);
		img.setRoi(roi);
	}

	private void chooseFitter() {
		final int index = comboBoxFitter.getSelectedIndex() - 1;
		if (index < 0 || tif == null || detector == null) {
			((TitledBorder) panelLower.getBorder()).setTitle("none");
			((CardLayout) panelLower.getLayout()).show(panelLower, "FIRST");
			widgetSelection = 0;
			fitter = null;
			return;
		}
		widgetSelection = FITTER;

		final String key = fitterProvider.getVisibleKeys().get(index);
		fitterFactory = fitterProvider.getFactory(key);
		((TitledBorder) panelLower.getBorder()).setTitle(fitterFactory.getName());
		((CardLayout) panelLower.getLayout()).show(panelLower, key);
		System.out.println("Fitter_" + index + " : " + key);
	
		fitterPreview();
		panelLower.repaint();
	}

	@SuppressWarnings("unchecked")
	private void fitterPreview() { // use only CentroidFitter for performance reasons
		final ImagePlus img = previewerWindow.getImagePlus();
		img.killRoi();
		final int frameNumber = img.getCurrentSlice();
		final float pixelSize = (float) img.getCalibration().pixelDepth;
		Roi currentRoi = img.getRoi();
		final ImageStack stack = img.getStack();
		ImageProcessor ip = stack.getProcessor(frameNumber);
		if (currentRoi != null){
			ip.setRoi(currentRoi.getBounds());
			ip = ip.crop();
		} else{
			currentRoi = new Roi(0,0,ip.getWidth(),ip.getHeight());
		}
		final Img<T> curImage = LemmingUtils.wrap(ip.getPixels(), new long[]{ip.getWidth(), ip.getHeight()});
		final ImgLib2Frame<T> curFrame = new ImgLib2Frame<>(frameNumber, (int) curImage.dimension(0), (int) curImage.dimension(1), pixelSize, curImage);
		detResults = (FrameElements<T>) detector.preview(curFrame);
		fitResults = CentroidFitterIP.fit(detResults.getList(), ip, fitterFactory.getHalfKernel(), pixelSize);

		if (fitResults == null) return;
		final FloatPolygon points = LemmingUtils.convertToPoints(fitResults, currentRoi.getBounds(), pixelSize);
		final PointRoi roi = new PointRoi(points);
		img.setRoi(roi);
	}
	
	// set a new Renderer
	private void initRenderer() {
		rendererWindow = new ImageWindow(renderer.getImage());
		rendererWindow.addKeyListener(new KeyAdapter() {
			@Override
			public void keyPressed(KeyEvent e) {
				if (e.getKeyChar() == 'C') {
					contrastAdjuster = new ContrastAdjuster();
					contrastAdjuster.run("B&C");
				}
			}
		});
		rendererWindow.getCanvas().addMouseListener(new MouseAdapter() {
			@Override
			public void mouseClicked(MouseEvent e) {
				if (e.getClickCount() == 2) {
					if (!processed) {
						IJ.showMessage(getTitle(), "Zoom only works if you press Process first!");
						return;
					}
					try {
						Rectangle rect = renderer.getImage().getRoi().getBounds();

						final double xmin = (Double) settings.get(RendererFactory.KEY_xmin);
						final double xmax = (Double) settings.get(RendererFactory.KEY_xmax);
						final double ymin = (Double) settings.get(RendererFactory.KEY_ymin);
						final double ymax = (Double) settings.get(RendererFactory.KEY_ymax);
						final int xbins = (Integer) settings.get(RendererFactory.KEY_xBins);
						final int ybins = (Integer) settings.get(RendererFactory.KEY_yBins);

						final double new_xmin = (xmax - xmin) * rect.getMinX() / xbins + xmin;
						final double new_ymin = (ymax - ymin) * rect.getMinY() / ybins + ymin;
						final double new_xmax = (xmax - xmin) * rect.getMaxX() / xbins + xmin;
						final double new_ymax = (ymax - ymin) * rect.getMaxY() / ybins + ymin;
						final double factx = rect.getWidth() / rect.getHeight();
						final double facty = rect.getHeight() / rect.getWidth();
						final double ar = Math.min(factx, facty);
						final int new_xbins = (int) (Math.round(xbins * ar));
						final int new_ybins = (int) (Math.round(ybins * ar));

						settings.put(RendererFactory.KEY_xmin, new_xmin);
						settings.put(RendererFactory.KEY_ymin, new_ymin);
						settings.put(RendererFactory.KEY_xmax, new_xmax);
						settings.put(RendererFactory.KEY_ymax, new_ymax);
						settings.put(RendererFactory.KEY_xBins, new_xbins);
						settings.put(RendererFactory.KEY_yBins, new_ybins);
						rendererShow(settings);
					} catch (NullPointerException ne) {
					}
				}
			}
		});
	}

	private void chooseRenderer() {
		final int index = comboBoxRenderer.getSelectedIndex() - 1;
		if (index < 0)
			return;

		final String key = rendererProvider.getVisibleKeys().get(index);
		rendererFactory = rendererProvider.getFactory(key);
		System.out.println("Renderer_" + index + " : " + key);
		((TitledBorder) panelFilter.getBorder()).setTitle(rendererFactory.getName());
		((CardLayout) panelFilter.getLayout()).show(panelFilter, key);
		
		Map<String, Object> rendererSettings = rendererFactory.getConfigurationPanel().getSettings();
		if (previewerWindow != null) {
			rendererSettings.put(RendererFactory.KEY_xmax, previewerWindow.getImagePlus().getWidth()
					* previewerWindow.getImagePlus().getCalibration().pixelDepth); 
			rendererSettings.put(RendererFactory.KEY_ymax, previewerWindow.getImagePlus().getHeight()
					* previewerWindow.getImagePlus().getCalibration().pixelDepth);
		}
		settings.putAll(rendererSettings);
		rendererFactory.setAndCheckSettings(rendererSettings);
		renderer = rendererFactory.getRenderer();
		panelFilter.repaint();

		initRenderer();
		
		if (processed)
			rendererShow(settings);
		else
			rendererPreview(settings);
	}

	private void rendererPreview(Map<String, Object> map) {
		rendererFactory.setAndCheckSettings(map);
		rendererWindow.getCanvas().fitToWindow();

		List<Element> list = fitResults;
		if (list != null && !list.isEmpty()) {
			renderer.preview(list);
		}
	}

	private void rendererShow(Map<String, Object> map) {
		rendererFactory.setAndCheckSettings(map);
		renderer = rendererFactory.getRenderer();
		rendererWindow.setImage(renderer.getImage());
		rendererWindow.getCanvas().fitToWindow();
		final ExtendableTable tableToRender = filteredTable == null ? table : filteredTable;

		if (tableToRender != null && tableToRender.columnNames().size() > 0) {
			final Store previewStore = tableToRender.getFIFO();
			System.out.println("Rendering " + tableToRender.getNumberOfRows() + " elements");
			renderer.setInput(previewStore);
			renderer.run();
			rendererWindow.repaint();
		}
	}

	private void filterTable() {
		// TODO  
		if (!processed) {
			if (IJ.showMessageWithCancel("Filter", "Pipeline not yet processed.\nDo you want to process it now?"))
				process(false);
			else
				return;
			if (table == null)
				return;
			final ExecutorService executor = Executors.newSingleThreadExecutor();
			start = System.currentTimeMillis();
			Future<?> f = executor.submit(manager);
			try {
				f.get();
			} catch (InterruptedException | ExecutionException e) {
				e.printStackTrace();return;
			}
		}
		 
		((TitledBorder) panelFilter.getBorder()).setTitle("Filter");
		((CardLayout) panelFilter.getLayout()).show(panelFilter, FilterPanel.KEY);
		panelFilter.repaint();
		FilterPanel comp = (FilterPanel) getConfigSettings(panelFilter);
		comp.setTable(table);
		
		if (renderer != null)
			rendererShow(settings);
		else
			IJ.showMessage(getTitle(), "No renderer chosen!\n No data will be displayed.");
	}

	private void saveLocalizations() {
		final SaveDialog sd = new SaveDialog("Import Images", "Results", ".csv");
		if(sd.getFileName()==null) return;
		lastDir = sd.getDirectory();
		OpenDialog.setDefaultDirectory(lastDir);
		final File file = new File(sd.getDirectory()+sd.getFileName());
		if (this.chkboxFilter.isSelected()) {
			ExtendableTable tableToProcess = filteredTable == null ? table : filteredTable;
			final Store s = tableToProcess.getFIFO();
			final StoreSaver tSaver = new StoreSaver(file);
			tSaver.putMetadata(settings);
			tSaver.setInput(s);
			tSaver.run();
		} else {
			if (fitter != null) {
				saver = new SaveLocalizations(file);
				if (!manager.getMap().containsKey(fitter.hashCode())) manager.add(fitter);
				manager.add(saver);
				manager.linkModules(fitter, saver, false, 100);
			} else {
				IJ.showMessage(getTitle(), "No Fitter chosen!");
			}
		}
	}
	
	private void createInitialPanels(){
		final ConfigurationPanel panelFirst = new ConfigurationPanel() {
			/**
			 * 
			 */
			private static final long serialVersionUID = 1L;

			@Override
			public void setSettings(Map<String, Object> settings) {
			}

			@Override
			public Map<String, Object> getSettings() {
				return null;
			}
		};
		panelLower.add(panelFirst, "FIRST");
		final ConfigurationPanel panelSecond = new ConfigurationPanel() {
			/**
			 * 
			 */
			private static final long serialVersionUID = 1L;

			@Override
			public void setSettings(Map<String, Object> settings) {
			}

			@Override
			public Map<String, Object> getSettings() {
				return null;
			}
		};
		panelFilter.add(panelSecond, "SECOND");
		final FilterPanel fp = new FilterPanel();
		fp.addPropertyChangeListener(ConfigurationPanel.propertyName, new PropertyChangeListener() {
			@Override
			public void propertyChange(PropertyChangeEvent evt) {
				if (table.filtersCollection.isEmpty()) {
					filteredTable = null;
				} else {
					filteredTable = table.filter();
				}
				if (renderer != null)
					rendererShow(settings);
			}
		});
		panelFilter.add(fp, FilterPanel.KEY);
	}

	private void createDetectorProvider() {
		detectorProvider = new DetectorProvider();
		final List<String> visibleKeys = detectorProvider.getVisibleKeys();
		final List<String> detectorNames = new ArrayList<>();
		final List<String> infoTexts = new ArrayList<>();
		detectorNames.add("none");
		for (final String key : visibleKeys) {
			final DetectorFactory factory = detectorProvider.getFactory(key);
			detectorNames.add(detectorProvider.getFactory(key).getName());
			infoTexts.add(detectorProvider.getFactory(key).getInfoText());
			final ConfigurationPanel panelDown = factory.getConfigurationPanel();
			panelDown.addPropertyChangeListener(ConfigurationPanel.propertyName, new PropertyChangeListener() {
				@SuppressWarnings("unchecked")
				@Override
				public void propertyChange(PropertyChangeEvent evt) {
					Map<String, Object> value = (Map<String, Object>) evt.getNewValue();
					detectorPreview(value);
				}
			});
			panelLower.add(panelDown, key);
		}
		final String[] names = detectorNames.toArray(new String[] {});
		comboBoxPeakDet.setModel(new DefaultComboBoxModel<>(names));
		comboBoxPeakDet.setRenderer(new ToolTipRenderer(infoTexts));
	}

	private void createPreProcessingProvider() {
		preProcessingProvider = new PreProcessingProvider();
		final List<String> visibleKeys = preProcessingProvider.getVisibleKeys();
		final List<String> infoTexts = new ArrayList<>();
		for (final String key : visibleKeys) {
			final PreProcessingFactory factory = preProcessingProvider.getFactory(key);
			infoTexts.add(factory.getInfoText());
			final ConfigurationPanel panelDown = factory.getConfigurationPanel();
			panelDown.addPropertyChangeListener(ConfigurationPanel.propertyName, new PropertyChangeListener() {
				@SuppressWarnings("unchecked")
				@Override
				public void propertyChange(PropertyChangeEvent evt) {
					Map<String, Object> value = (Map<String, Object>) evt.getNewValue();
					ppPreview(value);
				}
			});
			panelLower.add(panelDown, key);
		}
	}

	private void createFitterProvider() {
		fitterProvider = new FitterProvider();
		final List<String> visibleKeys = fitterProvider.getVisibleKeys();
		final List<String> fitterNames = new ArrayList<>();
		final List<String> infoTexts = new ArrayList<>();
		fitterNames.add("none");
		for (final String key : visibleKeys) {
			final FitterFactory factory = fitterProvider.getFactory(key);
			fitterNames.add(factory.getName());
			infoTexts.add(factory.getInfoText());
			final ConfigurationPanel panelDown = factory.getConfigurationPanel();
			panelDown.addPropertyChangeListener(ConfigurationPanel.propertyName, new PropertyChangeListener() {
				@Override
				public void propertyChange(PropertyChangeEvent evt) {
					fitterPreview();
				}
			});
			panelLower.add(panelDown, key);
		}
		final String[] names = fitterNames.toArray(new String[] {});
		comboBoxFitter.setModel(new DefaultComboBoxModel<>(names));
		comboBoxFitter.setRenderer(new ToolTipRenderer(infoTexts));
	}

	private void createRendererProvider() {
		rendererProvider = new RendererProvider();
		final List<String> visibleKeys = rendererProvider.getVisibleKeys();
		final List<String> rendererNames = new ArrayList<>();
		final List<String> infoTexts = new ArrayList<>();
		rendererNames.add("none");
		for (final String key : visibleKeys) {
			RendererFactory factory = rendererProvider.getFactory(key);
			rendererNames.add(factory.getName());
			infoTexts.add(factory.getInfoText());
			final ConfigurationPanel panelDown = factory.getConfigurationPanel();
			panelDown.addPropertyChangeListener(ConfigurationPanel.propertyName, new PropertyChangeListener() {
				@SuppressWarnings("unchecked")
				@Override
				public void propertyChange(PropertyChangeEvent evt) {
					Map<String, Object> map = (Map<String, Object>) evt.getNewValue();
					if (processed)
						rendererShow(map);
					else
						rendererPreview(map);
				}
			});
			panelFilter.add(panelDown, key);
		}
		String[] names = rendererNames.toArray(new String[] {});
		comboBoxRenderer.setModel(new DefaultComboBoxModel<>(names));
		comboBoxRenderer.setRenderer(new ToolTipRenderer(infoTexts));
	}

	@SuppressWarnings("unused")
	private void createActionProvider() {
		actionProvider = new ActionProvider();
		final List<String> visibleKeys = actionProvider.getVisibleKeys();
		final List<String> actionNames = new ArrayList<>(visibleKeys.size());
		final List<String> infoTexts = new ArrayList<>(visibleKeys.size());
		for (final String key : visibleKeys) {
			actionNames.add(actionProvider.getFactory(key).getName());
			infoTexts.add(actionProvider.getFactory(key).getInfoText());
		}
		String[] names = actionNames.toArray(new String[] {});
		//System.out.println(names.toString());
	}

	
class ToolTipRenderer extends DefaultListCellRenderer {
		private static final long serialVersionUID = 1L;
		List<String> tooltips;

		public ToolTipRenderer(List<String> tooltips) {
			this.tooltips = tooltips;
		}

		@Override
		public Component getListCellRendererComponent(JList<?> list, Object value, int index, boolean isSelected, boolean cellHasFocus) {

			JComponent comp = (JComponent) super.getListCellRendererComponent(list, value, index, isSelected, cellHasFocus);

			if (0 < index && null != value && null != tooltips)
				list.setToolTipText(tooltips.get(index - 1));
			return comp;
		}
	}

}
