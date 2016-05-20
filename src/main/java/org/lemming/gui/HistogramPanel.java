package org.lemming.gui;

import javax.swing.JPanel;
import javax.swing.WindowConstants;

import java.awt.GridBagLayout;

import javax.swing.border.LineBorder;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.IntervalMarker;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.StandardXYBarPainter;
import org.jfree.chart.renderer.xy.XYBarRenderer;
import org.lemming.modules.TableLoader;
import org.lemming.pipeline.ExtendableTable;
import org.lemming.tools.LogHistogramDataset;
import org.lemming.tools.NumericHistogram;
import org.lemming.tools.XYTextSimpleAnnotation;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Font;

import javax.swing.JComboBox;
import javax.swing.JFrame;

import java.awt.GridBagConstraints;
import java.awt.Insets;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.event.MouseMotionListener;
import java.awt.geom.Rectangle2D;
import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.awt.event.FocusEvent;
import java.awt.event.FocusListener;

import javax.swing.ComboBoxModel;
import javax.swing.DefaultComboBoxModel;

public class HistogramPanel extends JPanel
{

	private static final long serialVersionUID = 6491446979855209762L;
	private static final Font FONT = new Font( "Arial", Font.PLAIN, 11 );
	private static final Font SMALL_FONT = FONT.deriveFont( 10f );
	private static final Color annotationColor = new java.awt.Color( 252, 117, 0 );
	private static final String DATA_SERIES_NAME = "Data";
	private static final int maxCount = 1000;
	private static final int maxBins = 80;
	private final ChangeEvent CHANGE_EVENT = new ChangeEvent( this );
	JComboBox<String> jComboBoxFeature;
	private ChartPanel chartPanel;
	private LogHistogramDataset dataset;
	private XYPlot plot;
	private IntervalMarker intervalMarker;
	private XYTextSimpleAnnotation annotationUpper;
	private XYTextSimpleAnnotation annotationLower;
	private String key;
	private final List< String > allKeys;
	private final ArrayList< ChangeListener > listeners = new ArrayList<>();
	private double threshold;
	private double upperThreshold;
	private boolean upperSelected;
	protected double offset;
	protected boolean lowerDragging;
	protected boolean upperDragging;
	private final ExtendableTable table;
	private final Random rnd;

	/**
	 * CONSTRUCTOR
	 */
	HistogramPanel(ExtendableTable table) {
		super();
		allKeys = new ArrayList<>();
		this.table = table;
		allKeys.addAll(table.getNames().keySet());
		this.rnd = new Random(System.currentTimeMillis());
		initGUI();
		jComboBoxFeature.setSelectedIndex( 0 );
	}

	/**
	 * Set the threshold currently selected for the data displayed in this
	 * panel.
	 *
	 * @see #isAboveThreshold()
	 */
	public void setThreshold( final double value )
	{
		// Compute new value 
		double minimum = plot.getDomainAxis().getLowerBound();
		threshold = Math.max(minimum, value);
	}
	
	public void setUpperThreshold(final double value) {
		// Compute new value
		double maximum = plot.getDomainAxis().getUpperBound();
		upperThreshold = Math.min(maximum, value);
	}

	
	/**
	 * Return the threshold currently selected for the data displayed in this
	 * panel.
	 */
	public double getThreshold()
	{
		return threshold;
	}
	
	public double getUpperThreshold()
	{
		return upperThreshold;
	}
	
	/**
	 * Return the Enum constant selected in this panel.
	 */
	public String getKey()
	{
		return key;
	}

	/**
	 * Add an {@link ChangeListener} to this panel. The {@link ChangeListener}
	 * will be notified when a change happens to the threshold displayed by this
	 * panel, whether due to the slider being move, the auto-threshold button
	 * being pressed, or the combo-box selection being changed.
	 */
	public void addChangeListener( final ChangeListener listener )
	{
		listeners.add( listener );
	}

	/**
	 * Remove an ChangeListener.
	 *
	 * @return true if the listener was in listener collection of this instance.
	 */
	public boolean removeChangeListener( final ChangeListener listener )
	{
		return listeners.remove( listener );
	}

	/**
	 * Refreshes the histogram content. Call this method when the values in the
	 * {@link #valuesMap} changed to update histogram display.
	 */
	public void refresh()
	{
		final double old = getThreshold();
		final double oldUpper = getUpperThreshold();
		key = allKeys.get( jComboBoxFeature.getSelectedIndex() );
		NumericHistogram histogram = new NumericHistogram();
		histogram.allocate(maxBins);
		final List<Number> col = table.getColumn(key); 
		final int nRows = table.getNumberOfRows();
		for (int i = 0 ; i < Math.min(maxCount, nRows); i++){
			Number rowD = col.get(rnd.nextInt(nRows));
			histogram.add(rowD.doubleValue());
		}

		if (  null == col || 0 == col.size()  )	{
			dataset = new LogHistogramDataset();
			annotationUpper.setLocation( 0.5f, 0.5f );
			annotationUpper.setText( "No data" );
		}
		else {
			dataset = new LogHistogramDataset();
			dataset.addSeries( DATA_SERIES_NAME, histogram.getCounts(), maxBins, histogram.quantile(0), histogram.quantile(1));
		}
		plot.setDataset( dataset );
		setThreshold(old);
		setUpperThreshold(oldUpper);
		chartPanel.repaint();
	}

	
	private void fireThresholdChanged()
	{
		for ( final ChangeListener al : listeners )
			al.stateChanged( CHANGE_EVENT );
	}

	private void comboBoxSelectionChanged()
	{
		// long start = System.currentTimeMillis();
		final int index = jComboBoxFeature.getSelectedIndex();
		key = allKeys.get( index );
		NumericHistogram histogram = new NumericHistogram();
		histogram.allocate(maxBins);
		final List<Number> col = table.getColumn(key);
		final int nRows = table.getNumberOfRows();
		for (int i = 0 ; i < Math.min(maxCount, nRows); i++){ 			// random portion of the whole data set
			Number rowD = col.get(rnd.nextInt(nRows));
			if (rowD != null)
				histogram.add(rowD.doubleValue());
		}
		Number rowD = col.get(0);
		if (rowD != null)
			histogram.add(rowD.doubleValue());
		rowD = col.get(nRows-1);
		if (rowD != null)							// set first and last to get the whole range
			histogram.add(rowD.doubleValue());		// in sequential data
		
		if ( 0 == col.size() )
		{
			dataset = new LogHistogramDataset();
			setThreshold(Double.NaN);
			setUpperThreshold(Double.NaN);
			annotationUpper.setLocation( 0.5f, 0.5f );
			annotationUpper.setText( "No data" );
			plot.setDataset( dataset );
			fireThresholdChanged();
		}
		else
		{
			dataset = new LogHistogramDataset();
			dataset.addSeries( DATA_SERIES_NAME, histogram.getCounts(), maxBins, histogram.quantile(0), histogram.quantile(1));
			
			plot.setDataset( dataset );
			final double length = plot.getDomainAxis().getRange().getLength();
			setThreshold(plot.getDomainAxis().getRange().getCentralValue()-0.25*length);
			setUpperThreshold(plot.getDomainAxis().getRange().getCentralValue()+0.25*length);
		}
		resetAxes();
		redrawThresholdMarker();
		// System.out.println("Histogram created in :" + (System.currentTimeMillis()-start) + "ms");
	}

	private void initGUI()
	{
		final Dimension panelSize = new java.awt.Dimension( 280, 120 );
		final Dimension panelMaxSize = new java.awt.Dimension( 1000, 120 );
		try
		{
			final GridBagLayout gridBagLayout = new GridBagLayout();
			gridBagLayout.columnWidths = new int[]{50, 0};
			gridBagLayout.rowHeights = new int[]{0, 0, 0};
			gridBagLayout.columnWeights = new double[]{1.0, Double.MIN_VALUE};
			gridBagLayout.rowWeights = new double[]{0.0, 1.0, Double.MIN_VALUE};
			this.setLayout( gridBagLayout );
			this.setPreferredSize( panelSize );
			this.setMaximumSize( panelMaxSize );
			this.setBorder( new LineBorder( annotationColor, 1, true ) );

			final ComboBoxModel<String> jComboBoxFeatureModel = new DefaultComboBoxModel<>( table.getNames().keySet().toArray( new String[] {} ) );
			jComboBoxFeature = new JComboBox<>();
			GridBagConstraints gbc_comboBox = new GridBagConstraints();
			gbc_comboBox.insets = new Insets(0, 0, 1, 0);
			gbc_comboBox.gridx = 0;
			gbc_comboBox.gridy = 0;
			this.add( jComboBoxFeature, gbc_comboBox );
			jComboBoxFeature.setPreferredSize(new Dimension(80, 27));
			jComboBoxFeature.setModel( jComboBoxFeatureModel );
			jComboBoxFeature.setFont( FONT );
			jComboBoxFeature.addActionListener(new ActionListener(){
				@Override
				public void actionPerformed(ActionEvent e){comboBoxSelectionChanged();}});

			createHistogramPlot();
			chartPanel.setPreferredSize( new Dimension( 0, 0 ) );
			this.add( chartPanel, new GridBagConstraints( 0, 1, 3, 1, 0.0, 0.0, GridBagConstraints.CENTER, GridBagConstraints.BOTH, new Insets( 0, 0, 0, 0 ), 0, 0 ) );
			chartPanel.setOpaque( false );

		}
		catch ( final Exception e )
		{
			e.printStackTrace();
		}
	}

	/**
	 * Instantiate and configure the histogram chart.
	 */
	private void createHistogramPlot()
	{
		dataset = new LogHistogramDataset();
		JFreeChart chart = ChartFactory.createHistogram(null, null, null, dataset, PlotOrientation.VERTICAL, false, false, false);

		plot = chart.getXYPlot();
		final XYBarRenderer renderer = ( XYBarRenderer ) plot.getRenderer();
		renderer.setShadowVisible( false );
		renderer.setMargin( 0 );
		renderer.setBarPainter( new StandardXYBarPainter() );
		renderer.setDrawBarOutline( true );
		renderer.setSeriesOutlinePaint( 0, Color.black );
		renderer.setSeriesPaint( 0, new Color( 1, 1, 1, 0 ) );

		plot.setBackgroundPaint( new Color( 1, 1, 1, 0 ) );
		plot.setOutlineVisible( false );
		plot.setDomainCrosshairVisible( false );
		plot.setDomainGridlinesVisible( false );
		plot.setRangeCrosshairVisible( false );
		plot.setRangeGridlinesVisible( false );
		plot.getRangeAxis().setVisible( false );
		plot.getDomainAxis().setVisible( false );

		chart.setBorderVisible( false );
		chart.setBackgroundPaint( new Color( 0.8f, 0.8f, 0.9f ) );

		intervalMarker = new IntervalMarker( 0, 0, new Color( 0.3f, 0.5f, 0.8f ), new BasicStroke(), new Color( 0, 0, 0.5f ), new BasicStroke( 1.5f ), 0.5f );
		plot.addDomainMarker( intervalMarker );

		chartPanel = new ChartPanel(chart);
		final MouseListener[] mls = chartPanel.getMouseListeners();
		for ( final MouseListener ml : mls )
			chartPanel.removeMouseListener( ml );

		chartPanel.addMouseListener( new MouseListener()
		{
			@Override
			public void mouseReleased( final MouseEvent e ){
				lowerDragging = false;
				upperDragging = false;
				fireThresholdChanged();
			}

			@Override
			public void mousePressed( final MouseEvent e ){
				chartPanel.requestFocusInWindow();
				final double x = getXFromChartEvent( e );
				final double length = plot.getDomainAxis().getRange().getLength()*0.05;
				
				boolean lowerPressed = false;
				boolean upperPressed = false;
				
				if (upperSelected){
					if (x > getUpperThreshold()-length && x < getUpperThreshold()+length)
						upperPressed = true;
					else if (x > getThreshold()-length && x < getThreshold()+length)
						lowerPressed = true;
				} else {
					if (x > getThreshold()-length && x < getThreshold()+length)
						lowerPressed = true;
					else if (x > getUpperThreshold()-length && x < getUpperThreshold()+length)
						upperPressed = true;
				}
				
				if (lowerPressed) {
					offset = x - getThreshold();
					upperSelected = false;
					lowerDragging = true;
					return;
				}
				lowerDragging = false;
				
				if (upperPressed) {
					offset = x - getUpperThreshold();
					upperSelected = true;
					upperDragging = true;
					return;
				}
				upperDragging = false;				
			}

			@Override
			public void mouseExited( final MouseEvent e )
			{}

			@Override
			public void mouseEntered( final MouseEvent e )
			{}

			@Override
			public void mouseClicked( final MouseEvent e )
			{}
		} );
		chartPanel.addMouseMotionListener( new MouseMotionListener()
		{
			@Override
			public void mouseMoved( final MouseEvent e )
			{}

			@Override
			public void mouseDragged( final MouseEvent e )
			{
				double left = getXFromChartEvent( e ) - offset;
				
				if (lowerDragging) {
					double hMax = getUpperThreshold();
					left = Math.min(left, hMax);
					setThreshold(left);
					
				} else if (upperDragging) {
					double hMin = getThreshold();
					left = Math.max(left, hMin);
					setUpperThreshold(left);					
				}
				redrawThresholdMarker();
			}
		} );
		chartPanel.setFocusable( true );
		chartPanel.addFocusListener( new FocusListener()
		{

			@Override
			public void focusLost( final FocusEvent arg0 )
			{
				annotationUpper.setColor( annotationColor.darker() );
				annotationLower.setColor( annotationColor.darker() );
			}

			@Override
			public void focusGained( final FocusEvent arg0 )
			{
				annotationUpper.setColor( Color.RED.darker() );
				annotationLower.setColor( Color.RED.darker() );
			}
		} );

		annotationUpper = new XYTextSimpleAnnotation( chartPanel );
		annotationUpper.setFont( SMALL_FONT.deriveFont( Font.BOLD ) );
		annotationUpper.setColor( annotationColor.darker() );
		plot.addAnnotation( annotationUpper );
		annotationLower = new XYTextSimpleAnnotation( chartPanel );
		annotationLower.setFont( SMALL_FONT.deriveFont( Font.BOLD ) );
		annotationLower.setColor( annotationColor.darker() );
		plot.addAnnotation( annotationLower );
	}

	private double getXFromChartEvent( final MouseEvent mouseEvent )
	{
		final Rectangle2D plotArea = chartPanel.getScreenDataArea();
		return plot.getDomainAxis().java2DToValue( mouseEvent.getX(), plotArea, plot.getDomainAxisEdge() );
	}

	private void redrawThresholdMarker()
	{
		final String selectedFeature = allKeys.get( jComboBoxFeature.getSelectedIndex() );
		List<Number> col = table.getColumn(selectedFeature);
		if ( null == col )
			return;

		intervalMarker.setStartValue( getThreshold() );
		intervalMarker.setEndValue( getUpperThreshold() );
		
		float x, y;
		if ( getUpperThreshold() > 0.85 * plot.getDomainAxis().getUpperBound() )
			x = ( float ) ( getUpperThreshold() - 0.15 * plot.getDomainAxis().getRange().getLength() );
		else
			x = ( float ) ( getUpperThreshold()+ 0.05 * plot.getDomainAxis().getRange().getLength() );
		

		y = ( float ) ( 0.9 * plot.getRangeAxis().getUpperBound() );
		annotationUpper.setText( String.format( "%.2f", getUpperThreshold() ) );
		annotationUpper.setLocation( x, y );
		
		if ( getThreshold() < 0.15 * plot.getDomainAxis().getLowerBound() )
			x = ( float ) ( getThreshold() + 0.05 * plot.getDomainAxis().getRange().getLength() );
		else
			x = ( float ) ( getThreshold() - 0.15 * plot.getDomainAxis().getRange().getLength() );
		
		y = ( float ) ( 0.8 * plot.getRangeAxis().getUpperBound() );
		annotationLower.setText( String.format( "%.2f", getThreshold() ) );
		annotationLower.setLocation( x, y );
	}

	private void resetAxes()
	{
		plot.getRangeAxis().setLowerMargin( 0 );
		plot.getRangeAxis().setUpperMargin( 0 );
		plot.getDomainAxis().setLowerMargin( 0 );
		plot.getDomainAxis().setUpperMargin( 0 );
	}

	/*
	 * MAIN METHOD
	 */

	/**
	 * Display this JPanel inside a new JFrame.
	 */
	public static void main( final String[] args )
	{
		TableLoader loader = new TableLoader(new File("/Users/ronny/Documents/testTable.csv"));
		//loader.readObjects();
		loader.readCSV(',');
		
		// Create GUI
		final HistogramPanel tp = new HistogramPanel( loader.getTable());
		tp.resetAxes();
		final JFrame frame = new JFrame();
		frame.getContentPane().add( tp );
		frame.setDefaultCloseOperation( WindowConstants.DISPOSE_ON_CLOSE );
		frame.pack();
		frame.setVisible( true );
	}

}
