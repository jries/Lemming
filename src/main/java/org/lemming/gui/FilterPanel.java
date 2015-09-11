package org.lemming.gui;

import javax.swing.JPanel;
import javax.swing.WindowConstants;

import java.awt.GridBagLayout;
import javax.swing.border.LineBorder;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

import org.apache.commons.lang.ArrayUtils;
import org.apache.commons.math3.stat.descriptive.rank.Percentile;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.IntervalMarker;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.StandardXYBarPainter;
import org.jfree.chart.renderer.xy.XYBarRenderer;
import org.lemming.pipeline.ExtendableTable;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Font;

import javax.swing.JComboBox;
import javax.swing.JFrame;

import java.awt.GridBagConstraints;
import java.awt.event.ActionListener;
import java.awt.event.ActionEvent;
import java.awt.Insets;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.event.MouseMotionListener;
import java.awt.geom.Rectangle2D;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Random;
import java.awt.event.FocusEvent;
import java.awt.event.FocusListener;
import javax.swing.ComboBoxModel;
import javax.swing.DefaultComboBoxModel;

public class FilterPanel extends JPanel
{

	static final Font FONT = new Font( "Arial", Font.PLAIN, 11 );

	static final Font SMALL_FONT = FONT.deriveFont( 10f );

	private static final Color annotationColor = new java.awt.Color( 252, 117, 0 );

	private static final long serialVersionUID = 1L;

	private static final String DATA_SERIES_NAME = "Data";

	private final ChangeEvent CHANGE_EVENT = new ChangeEvent( this );

	JComboBox<String> jComboBoxFeature;

	private ChartPanel chartPanel;

	private LogHistogramDataset dataset;

	private JFreeChart chart;

	private XYPlot plot;

	private IntervalMarker intervalMarker;

	//private final Map< String, double[] > valuesMap;

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

	private ExtendableTable table;


	/*
	 * CONSTRUCTOR
	 */

	/*public FilterPanel(  final Map< String, double[] > valuesMap, final int selectedKey )
	{
		super();
		this.valuesMap = valuesMap;
		allKeys = new ArrayList<>();
		for (String k : valuesMap.keySet())
			allKeys.add(k);
		initGUI();
		jComboBoxFeature.setSelectedIndex( selectedKey );
	}

	public FilterPanel( final Map< String, double[] > valuesMap)
	{
		this( valuesMap, 0 );
	}*/

	public FilterPanel(ExtendableTable table, final int selectedKey) {
		super();
		allKeys = new ArrayList<>();
		this.table = table;
		for (String k : table.getNames().keySet())
			allKeys.add(k);
		initGUI();
		jComboBoxFeature.setSelectedIndex( selectedKey );
	}

	/*
	 * PUBLIC METHODS
	 */

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

	public Collection< ChangeListener > getChangeListeners()
	{
		return listeners;
	}

	/**
	 * Refreshes the histogram content. Call this method when the values in the
	 * {@link #valuesMap} changed to update histogram display.
	 */
	public void refresh()
	{
		final double old = getThreshold();
		key = allKeys.get( jComboBoxFeature.getSelectedIndex() );
		Double[] col = table.getColumn(key).toArray(new Double[]{});
		final double[] values = ArrayUtils.toPrimitive(col);

		if ( null == values || 0 == values.length )	{
			dataset = new LogHistogramDataset();
			annotationUpper.setLocation( 0.5f, 0.5f );
			annotationUpper.setText( "No data" );
		}
		else {
			final int nBins = getNBins( values, 8, 100 );
			dataset = new LogHistogramDataset();
			if ( nBins > 1 ){
				dataset.addSeries( DATA_SERIES_NAME, values, nBins );
			}
		}
		plot.setDataset( dataset );
		setThreshold(old);
		chartPanel.repaint();
	}

	/**
	 * Return the optimal bin number for a histogram of the data given in array,
	 * using the Freedman and Diaconis rule (bin_space = 2*IQR/n^(1/3)). It is
	 * ensured that the bin number returned is not smaller and no bigger than
	 * the bounds given in argument.
	 */
	private static final int getNBins( final double[] values, final int minBinNumber, final int maxBinNumber )
	{
		Percentile p = new Percentile();
		
		final int size = values.length;
		final double q1 = p.evaluate( values, 0.25 );
		final double q3 = p.evaluate( values, 0.75 );
		final double iqr = q3 - q1;
		final double binWidth = 2 * iqr * Math.pow( size, -0.333 );
		final double[] range = getRange( values );
		int nBin = ( int ) ( range[ 0 ] / binWidth + 1 );
		if ( nBin > maxBinNumber ){
			nBin = maxBinNumber;
		}
		else if ( nBin < minBinNumber ){
			nBin = minBinNumber;
		}
		return nBin;
	}
	
	/**
	 * Returns <code>[range, min, max]</code> of the given double array.
	 * 
	 * @return A double[] of length 3, where index 0 is the range, index 1 is
	 *         the min, and index 2 is the max.
	 */
	private static final double[] getRange( final double[] data )
	{
		double min = Double.POSITIVE_INFINITY;
		double max = Double.NEGATIVE_INFINITY;
		double value;
		for ( int i = 0; i < data.length; i++ )	{
			value = data[ i ];
			if ( value < min ){
				min = value;
			}
			if ( value > max ){
				max = value;
			}
		}
		return new double[] { ( max - min ), min, max };
	}

	private void fireThresholdChanged()
	{
		for ( final ChangeListener al : listeners )
			al.stateChanged( CHANGE_EVENT );
	}

	private void comboBoxSelectionChanged()
	{
		final int index = jComboBoxFeature.getSelectedIndex();
		key = allKeys.get( index );
		Double[] col = table.getColumn(key).toArray(new Double[]{});
		final double[] values = ArrayUtils.toPrimitive(col);
		if ( null == values || 0 == values.length )
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
			final int nBins = getNBins( values, 8, 100 );
			dataset = new LogHistogramDataset();
			if ( nBins > 1 ){
				dataset.addSeries( DATA_SERIES_NAME, values, nBins );
			}
			plot.setDataset( dataset );
			final double length = plot.getDomainAxis().getRange().getLength();
			setThreshold(plot.getDomainAxis().getRange().getCentralValue()-0.25*length);
			setUpperThreshold(plot.getDomainAxis().getRange().getCentralValue()+0.25*length);
		}
		resetAxes();
		redrawThresholdMarker();
	}

	private void initGUI()
	{
		final Dimension panelSize = new java.awt.Dimension( 250, 140 );
		final Dimension panelMaxSize = new java.awt.Dimension( 1000, 140 );
		try
		{
			final GridBagLayout thisLayout = new GridBagLayout();
			thisLayout.rowWeights = new double[] { 0.0, 1.0, 0.0 };
			thisLayout.rowHeights = new int[] { 10, 7, 15 };
			thisLayout.columnWeights = new double[] { 0.0, 0.0, 1.0 };
			thisLayout.columnWidths = new int[] { 7, 20, 7 };
			this.setLayout( thisLayout );
			this.setPreferredSize( panelSize );
			this.setMaximumSize( panelMaxSize );
			this.setBorder( new LineBorder( annotationColor, 1, true ) );
			{
				final ComboBoxModel<String> jComboBoxFeatureModel = new DefaultComboBoxModel<>( table.getNames().keySet().toArray( new String[] {} ) );
				jComboBoxFeature = new JComboBox<>();
				this.add( jComboBoxFeature, new GridBagConstraints( 0, 0, 3, 1, 0.0, 0.0, GridBagConstraints.CENTER, GridBagConstraints.HORIZONTAL, new Insets( 2, 5, 2, 5 ), 0, 0 ) );
				jComboBoxFeature.setModel( jComboBoxFeatureModel );
				jComboBoxFeature.setFont( FONT );
				jComboBoxFeature.addActionListener( new ActionListener()
				{
					@Override
					public void actionPerformed( final ActionEvent e1 )
					{
						comboBoxSelectionChanged();
					}
				} );
			}
			{
				createHistogramPlot();
				chartPanel.setPreferredSize( new Dimension( 0, 0 ) );
				this.add( chartPanel, new GridBagConstraints( 0, 1, 3, 1, 0.0, 0.0, GridBagConstraints.CENTER, GridBagConstraints.BOTH, new Insets( 0, 0, 0, 0 ), 0, 0 ) );
				chartPanel.setOpaque( false );
			}
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
		chart = ChartFactory.createHistogram( null, null, null, dataset, PlotOrientation.VERTICAL, false, false, false );

		plot = chart.getXYPlot();
		final XYBarRenderer renderer = ( XYBarRenderer ) plot.getRenderer();
		renderer.setShadowVisible( false );
		renderer.setMargin( 0 );
		renderer.setBarPainter( new StandardXYBarPainter() );
		renderer.setDrawBarOutline( true );
		renderer.setSeriesOutlinePaint( 0, Color.BLACK );
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
		chart.setBackgroundPaint( new Color( 0.6f, 0.6f, 0.7f ) );

		intervalMarker = new IntervalMarker( 0, 0, new Color( 0.3f, 0.5f, 0.8f ), new BasicStroke(), new Color( 0, 0, 0.5f ), new BasicStroke( 1.5f ), 0.5f );
		plot.addDomainMarker( intervalMarker );

		chartPanel = new ChartPanel( chart );
		final MouseListener[] mls = chartPanel.getMouseListeners();
		for ( final MouseListener ml : mls )
			chartPanel.removeMouseListener( ml );

		chartPanel.addMouseListener( new MouseListener()
		{
			@Override
			public void mouseReleased( final MouseEvent e ){
				lowerDragging = false;
				upperDragging = false;
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
			}

			@Override
			public void focusGained( final FocusEvent arg0 )
			{
				annotationUpper.setColor( Color.RED.darker() );
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
		Double[] col = table.getColumn(selectedFeature).toArray(new Double[]{});
		final double[] values = ArrayUtils.toPrimitive(col);
		if ( null == values )
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
		fireThresholdChanged();
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
		// Prepare fake data
		final int N_ITEMS = 100;
		final Random ran = new Random();
		double mean;
		
		ExtendableTable table = new ExtendableTable();
		table.addNewMember("Contrast");
		table.addNewMember("Morphology");
		table.addNewMember("Mean intensity");
		
		for (String k: table.getNames().keySet()){
			List<Object> col = table.getColumn(k);
			mean = ran.nextDouble() * 10;
			for ( int j = 0; j < N_ITEMS; j++ )
				col.add(ran.nextGaussian() + 5 + mean);
		}

		/*final String[] features = new String[] { "Contrast", "Morphology", "Mean intensity" };

		final Map< String, double[] > fv = new HashMap<>();
		for ( final String feature : features )
		{
			final double[] val = new double[ N_ITEMS ];
			mean = ran.nextDouble() * 10;
			for ( int j = 0; j < val.length; j++ )
				val[ j ] = ran.nextGaussian() + 5 + mean;
			fv.put( feature, val );
		}*/
		
		

		// Create GUI
		final FilterPanel tp = new FilterPanel( table, 0 );
		tp.resetAxes();
		final JFrame frame = new JFrame();
		frame.getContentPane().add( tp );
		frame.setDefaultCloseOperation( WindowConstants.DISPOSE_ON_CLOSE );
		frame.pack();
		frame.setVisible( true );
	}

}
