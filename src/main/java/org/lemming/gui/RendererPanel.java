package org.lemming.gui;

import java.util.HashMap;
import java.util.Hashtable;
import java.util.Map;

import javax.swing.GroupLayout;
import javax.swing.GroupLayout.Alignment;
import javax.swing.JLabel;
import javax.swing.LayoutStyle.ComponentPlacement;
import javax.swing.JFrame;
import javax.swing.JSpinner;
import javax.swing.SpinnerNumberModel;
import javax.swing.WindowConstants;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

import org.lemming.tools.WaitForKeyListener;

import java.awt.event.MouseWheelListener;
import java.awt.event.MouseWheelEvent;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;

public class RendererPanel extends ConfigurationPanel {

	/**
	 * 
	 */
	private static final long serialVersionUID = -3031663211936690561L;
	public static final String KEY_xBins = "xBins";
	public static final String KEY_yBins = "yBins";
	public static final String KEY_xmin = "xmin";
	public static final String KEY_xmax = "xmax";
	public static final String KEY_ymin = "ymin";
	public static final String KEY_ymax = "ymax";
	private RangeSlider rangeSliderX;
	private RangeSlider rangeSliderY;
	private JSpinner spinnerXBins;
	private JSpinner spinnerYBins;
	private int xOldValue=0;
	private int xOldUpperValue=100;
	private int yOldValue=0;
	private int yOldUpperValue=100;
	private int factor=100;
	private boolean changed=false;

	public RendererPanel() {
		setBorder(null);
		
		rangeSliderX = new RangeSlider(0,100);
		rangeSliderX.addMouseListener(new MouseAdapter() {
			@Override
			public void mouseReleased(MouseEvent e) {
				if(changed){
					fireChanged();
					changed=false;
				}
			}
		});
		rangeSliderX.addMouseWheelListener(new MouseWheelListener() {
			public void mouseWheelMoved(MouseWheelEvent e) {
				if( rangeSliderY.getMaximum()<100)
					factor = 10;
				else if( rangeSliderY.getMaximum()<10)
					factor = 1;
				else 
					factor = 100;
				int width = rangeSliderY.getMaximum() + (e.getWheelRotation() * factor);
				if (width < 1)
					width = 0;
				rangeSliderX.setMaximum(width);
				rangeSliderX.setMajorTickSpacing(width);
				rangeSliderX.setMinorTickSpacing(width/4);
				Hashtable<Integer,JLabel> dict = new Hashtable<>();
				dict.put(0, new JLabel(String.valueOf(0)));
				dict.put(width/2, new JLabel(String.valueOf(width/2)));
				dict.put(width, new JLabel(String.valueOf(width)));
				rangeSliderX.setLabelTable(dict);
				rangeSliderX.revalidate();
				e.consume();
			}
		});
		rangeSliderX.addChangeListener(new ChangeListener() {
			@Override
			public void stateChanged(ChangeEvent e) {
				
				if(rangeSliderX.getValue()!=xOldValue || rangeSliderX.getUpperValue()!=xOldUpperValue){
					xOldValue = rangeSliderX.getValue();
					xOldUpperValue = rangeSliderX.getUpperValue();
					changed = true;
				}
			}
		});
		rangeSliderX.setMinorTickSpacing(10);
		rangeSliderX.setMajorTickSpacing(50);
		rangeSliderX.setPaintTicks(true);
		rangeSliderX.setPaintLabels(true);
		
		JLabel lblX = new JLabel("X");
		
		JLabel lblY = new JLabel("Y");
		
		rangeSliderY = new RangeSlider(0,100);
		rangeSliderY.addMouseListener(new MouseAdapter() {
			@Override
			public void mouseReleased(MouseEvent e) {
				if(changed){
					fireChanged();
					changed=false;
				}
			}
		});
		rangeSliderY.addMouseWheelListener(new MouseWheelListener() {
			public void mouseWheelMoved(MouseWheelEvent e) {
				if( rangeSliderY.getMaximum()<100)
					factor = 10;
				else if( rangeSliderY.getMaximum()<10)
					factor = 1;
				else 
					factor = 100;
				int width = rangeSliderY.getMaximum() + (e.getWheelRotation() * factor);
				if (width < 1)
					width = 0;
				rangeSliderY.setMaximum(width);
				rangeSliderY.setMajorTickSpacing(width);
				rangeSliderY.setMinorTickSpacing(width/5);
				Hashtable<Integer,JLabel> dict = new Hashtable<>();
				dict.put(0, new JLabel(String.valueOf(0)));
				dict.put(width/2, new JLabel(String.valueOf(width/2)));
				dict.put(width, new JLabel(String.valueOf(width)));
				rangeSliderY.setLabelTable(dict);
				rangeSliderY.revalidate();
				e.consume();
			}
		});
		rangeSliderY.addChangeListener(new ChangeListener() {
			@Override
			public void stateChanged(ChangeEvent e) {
				if(rangeSliderY.getValue()!=yOldValue || rangeSliderY.getUpperValue()!=yOldUpperValue){
					yOldValue = rangeSliderY.getValue();
					yOldUpperValue = rangeSliderY.getUpperValue();
					changed = true;
				}
			}
		});
		rangeSliderY.setMinorTickSpacing(10);
		rangeSliderY.setMajorTickSpacing(50);
		rangeSliderY.setPaintTicks(true);
		rangeSliderY.setPaintLabels(true);
		
		JLabel lblXBins = new JLabel("X Bins");
		
		JLabel lblYBins = new JLabel("Y Bins");
		
		spinnerXBins = new JSpinner();
		spinnerXBins.addKeyListener(new WaitForKeyListener(1000, new Runnable(){
			@Override
			public void run() {
				fireChanged();
			}
		}));
		spinnerXBins.addChangeListener(new ChangeListener() {
			public void stateChanged(ChangeEvent e) {
				fireChanged();
			}
		});
		spinnerXBins.setModel(new SpinnerNumberModel(new Integer(100), null, null, new Integer(1)));
		
		spinnerYBins = new JSpinner();
		spinnerYBins.addKeyListener(new WaitForKeyListener(1000, new Runnable(){
			@Override
			public void run() {
				fireChanged();
			}
		}));
		spinnerYBins.addChangeListener(new ChangeListener() {
			public void stateChanged(ChangeEvent e) {
				fireChanged();
			}
		});
		spinnerYBins.setModel(new SpinnerNumberModel(new Integer(100), null, null, new Integer(1)));
		GroupLayout groupLayout = new GroupLayout(this);
		groupLayout.setHorizontalGroup(
			groupLayout.createParallelGroup(Alignment.LEADING)
				.addGroup(groupLayout.createSequentialGroup()
					.addContainerGap()
					.addGroup(groupLayout.createParallelGroup(Alignment.LEADING, false)
						.addGroup(groupLayout.createSequentialGroup()
							.addComponent(lblX)
							.addPreferredGap(ComponentPlacement.RELATED)
							.addComponent(rangeSliderX, GroupLayout.PREFERRED_SIZE, 243, GroupLayout.PREFERRED_SIZE))
						.addGroup(groupLayout.createSequentialGroup()
							.addComponent(lblY)
							.addPreferredGap(ComponentPlacement.RELATED)
							.addComponent(rangeSliderY, GroupLayout.DEFAULT_SIZE, GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
						.addGroup(groupLayout.createParallelGroup(Alignment.TRAILING, false)
							.addGroup(groupLayout.createSequentialGroup()
								.addComponent(lblYBins)
								.addPreferredGap(ComponentPlacement.RELATED)
								.addComponent(spinnerYBins))
							.addGroup(Alignment.LEADING, groupLayout.createSequentialGroup()
								.addComponent(lblXBins)
								.addPreferredGap(ComponentPlacement.RELATED)
								.addComponent(spinnerXBins, GroupLayout.PREFERRED_SIZE, 70, GroupLayout.PREFERRED_SIZE))))
					.addContainerGap(174, Short.MAX_VALUE))
		);
		groupLayout.setVerticalGroup(
			groupLayout.createParallelGroup(Alignment.LEADING)
				.addGroup(groupLayout.createSequentialGroup()
					.addContainerGap()
					.addGroup(groupLayout.createParallelGroup(Alignment.LEADING)
						.addComponent(lblX)
						.addComponent(rangeSliderX, GroupLayout.PREFERRED_SIZE, GroupLayout.DEFAULT_SIZE, GroupLayout.PREFERRED_SIZE))
					.addPreferredGap(ComponentPlacement.RELATED)
					.addGroup(groupLayout.createParallelGroup(Alignment.LEADING)
						.addComponent(lblY)
						.addComponent(rangeSliderY, GroupLayout.PREFERRED_SIZE, GroupLayout.DEFAULT_SIZE, GroupLayout.PREFERRED_SIZE))
					.addPreferredGap(ComponentPlacement.RELATED)
					.addGroup(groupLayout.createParallelGroup(Alignment.BASELINE)
						.addComponent(lblXBins)
						.addComponent(spinnerXBins, GroupLayout.PREFERRED_SIZE, GroupLayout.DEFAULT_SIZE, GroupLayout.PREFERRED_SIZE))
					.addPreferredGap(ComponentPlacement.RELATED)
					.addGroup(groupLayout.createParallelGroup(Alignment.BASELINE)
						.addComponent(spinnerYBins, GroupLayout.PREFERRED_SIZE, GroupLayout.DEFAULT_SIZE, GroupLayout.PREFERRED_SIZE)
						.addComponent(lblYBins))
					.addContainerGap(160, Short.MAX_VALUE))
		);
		setLayout(groupLayout);
	}

	@Override
	public void setSettings(Map<String, Object> settings) {
		rangeSliderX.setValue((int) settings.get(KEY_xmin));
		if ((int) settings.get(KEY_xmax) > rangeSliderX.getMaximum())
			rangeSliderX.setMaximum((int) settings.get(KEY_xmax));
		rangeSliderX.setUpperValue((int) settings.get(KEY_xmax));
		rangeSliderY.setValue((int) settings.get(KEY_ymin));
		if ((int) settings.get(KEY_ymax) > rangeSliderY.getMaximum())
			rangeSliderY.setMaximum((int) settings.get(KEY_ymax));
		rangeSliderY.setUpperValue((int) settings.get(KEY_ymax));
		spinnerXBins.setValue(settings.get(KEY_xBins));
		spinnerYBins.setValue(settings.get(KEY_yBins));
		validate();
		repaint();
	}

	@Override
	public Map<String, Object> getSettings() {
		HashMap<String, Object> settings = new HashMap<>();
		settings.put(KEY_xmin, rangeSliderX.getValue());
		settings.put(KEY_xmax, rangeSliderX.getUpperValue());
		settings.put(KEY_ymin, rangeSliderY.getValue());
		settings.put(KEY_ymax, rangeSliderY.getUpperValue());
		settings.put(KEY_xBins, spinnerXBins.getValue());
		settings.put(KEY_yBins, spinnerYBins.getValue());
		return settings;
	}
	
	/**
	 * Display this JPanel inside a new JFrame.
	 */
	public static void main( final String[] args )
	{
		
		// Create GUI
		final RendererPanel tp = new RendererPanel( );
		final JFrame frame = new JFrame();
		frame.getContentPane().add( tp );
		frame.setDefaultCloseOperation( WindowConstants.DISPOSE_ON_CLOSE );
		frame.pack();
		frame.setVisible( true );
	}
	
}
