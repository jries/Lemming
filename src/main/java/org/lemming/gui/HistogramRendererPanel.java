package org.lemming.gui;

import java.util.HashMap;
import java.util.Map;

import javax.swing.GroupLayout;
import javax.swing.GroupLayout.Alignment;
import javax.swing.JLabel;
import javax.swing.LayoutStyle.ComponentPlacement;
import javax.swing.JFrame;
import javax.swing.WindowConstants;
import javax.swing.JTextField;

import org.lemming.factories.RendererFactory;
import org.lemming.tools.WaitForKeyListener;

import javax.swing.SwingConstants;

public class HistogramRendererPanel extends ConfigurationPanel {

	/**
	 * 
	 */
	private static final long serialVersionUID = -3031663211936690561L;
	private JTextField textXBins;
	private JTextField textYBins;
	private JLabel lblX;
	private JLabel lblY;
	private JLabel labelX2;
	private JLabel labelY2;
	private Map<String, Object> settings = new HashMap<>();
	private Map<String, Object> initialSettings;;

	public HistogramRendererPanel() {
		setBorder(null);
		
		lblX = new JLabel("0");
		
		lblY = new JLabel("0");
		
		JLabel lblXBins = new JLabel("X Bins");
		
		JLabel lblYBins = new JLabel("Y Bins");
		
		textXBins = new JTextField();
		textXBins.setHorizontalAlignment(SwingConstants.TRAILING);
		textXBins.setText("500");
		textXBins.addKeyListener(new WaitForKeyListener(1000, new Runnable(){
			@Override
			public void run() {
				fireChanged();
			}
		}));
		
		textYBins = new JTextField();
		textYBins.setHorizontalAlignment(SwingConstants.TRAILING);
		textYBins.setText("500");
		textYBins.addKeyListener(new WaitForKeyListener(1000, new Runnable(){
			@Override
			public void run() {
				fireChanged();
			}
		}));
		
		labelX2 = new JLabel("100");
		
		labelY2 = new JLabel("100");
		GroupLayout groupLayout = new GroupLayout(this);
		groupLayout.setHorizontalGroup(
			groupLayout.createParallelGroup(Alignment.TRAILING)
				.addGroup(groupLayout.createSequentialGroup()
					.addContainerGap()
					.addGroup(groupLayout.createParallelGroup(Alignment.LEADING, false)
						.addGroup(groupLayout.createSequentialGroup()
							.addComponent(lblYBins)
							.addPreferredGap(ComponentPlacement.RELATED)
							.addComponent(textYBins))
						.addGroup(groupLayout.createSequentialGroup()
							.addComponent(lblXBins)
							.addPreferredGap(ComponentPlacement.RELATED)
							.addComponent(textXBins, GroupLayout.PREFERRED_SIZE, 70, GroupLayout.PREFERRED_SIZE)))
					.addContainerGap(330, Short.MAX_VALUE))
				.addGroup(groupLayout.createSequentialGroup()
					.addGap(53)
					.addGroup(groupLayout.createParallelGroup(Alignment.LEADING)
						.addComponent(lblX, GroupLayout.DEFAULT_SIZE, 38, Short.MAX_VALUE)
						.addComponent(lblY, GroupLayout.DEFAULT_SIZE, 38, Short.MAX_VALUE))
					.addPreferredGap(ComponentPlacement.RELATED)
					.addGroup(groupLayout.createParallelGroup(Alignment.LEADING, false)
						.addComponent(labelY2, GroupLayout.DEFAULT_SIZE, GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
						.addComponent(labelX2, GroupLayout.DEFAULT_SIZE, 39, Short.MAX_VALUE))
					.addGap(314))
		);
		groupLayout.setVerticalGroup(
			groupLayout.createParallelGroup(Alignment.LEADING)
				.addGroup(groupLayout.createSequentialGroup()
					.addContainerGap()
					.addGroup(groupLayout.createParallelGroup(Alignment.BASELINE)
						.addComponent(lblX)
						.addComponent(labelX2))
					.addPreferredGap(ComponentPlacement.RELATED)
					.addGroup(groupLayout.createParallelGroup(Alignment.BASELINE)
						.addComponent(lblY)
						.addComponent(labelY2))
					.addPreferredGap(ComponentPlacement.UNRELATED)
					.addGroup(groupLayout.createParallelGroup(Alignment.BASELINE)
						.addComponent(lblXBins)
						.addComponent(textXBins, GroupLayout.PREFERRED_SIZE, GroupLayout.DEFAULT_SIZE, GroupLayout.PREFERRED_SIZE))
					.addPreferredGap(ComponentPlacement.RELATED)
					.addGroup(groupLayout.createParallelGroup(Alignment.BASELINE)
						.addComponent(textYBins, GroupLayout.PREFERRED_SIZE, GroupLayout.DEFAULT_SIZE, GroupLayout.PREFERRED_SIZE)
						.addComponent(lblYBins))
					.addContainerGap(182, Short.MAX_VALUE))
		);
		setLayout(groupLayout);
		settings.put(RendererFactory.KEY_xmin, new Double(0));
		settings.put(RendererFactory.KEY_ymin, new Double(0));
		settings.put(RendererFactory.KEY_xmax, new Double(100));
		settings.put(RendererFactory.KEY_ymax, new Double(100));
		settings.put(RendererFactory.KEY_xBins,new Integer(500));
		settings.put(RendererFactory.KEY_yBins,new Integer(500));
		initialSettings = new HashMap<>(settings);
	}

	@Override
	public void setSettings(Map<String, Object> settings) {
		lblX.setText(String.format("%.4f",settings.get(RendererFactory.KEY_xmin)));
		lblY.setText(String.format("%.4f",settings.get(RendererFactory.KEY_ymin)));
		labelX2.setText(String.format("%.4f",settings.get(RendererFactory.KEY_xmax)));
		labelY2.setText(String.format("%.4f",settings.get(RendererFactory.KEY_ymax)));
		textXBins.setText(String.valueOf(settings.get(RendererFactory.KEY_xBins)));
		textYBins.setText(String.valueOf(settings.get(RendererFactory.KEY_yBins)));
		for (String key : settings.keySet())
			this.settings.put(key, settings.get(key));
		revalidate();
	}

	@Override
	public Map<String, Object> getSettings() {
		return settings;
	}
	
	public Map<String, Object> getInitialSettings(){
		return initialSettings;
	}
	/**
	 * Display this JPanel inside a new JFrame.
	 */
	public static void main( final String[] args )
	{
		
		// Create GUI
		final HistogramRendererPanel tp = new HistogramRendererPanel( );
		final JFrame frame = new JFrame();
		frame.getContentPane().add( tp );
		frame.setDefaultCloseOperation( WindowConstants.DISPOSE_ON_CLOSE );
		frame.pack();
		frame.setVisible( true );
	}
}
