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

public class GaussRendererPanel extends ConfigurationPanel {

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
	private Map<String, Object> initialSettings;
	private JLabel lblRanges;;

	public GaussRendererPanel() {
		setBorder(null);
		
		lblX = new JLabel("0");
		
		lblY = new JLabel("0");
		
		JLabel lblXBins = new JLabel("xBins");
		
		JLabel lblYBins = new JLabel("yBins");
		
		textXBins = new JTextField();
		textXBins.setHorizontalAlignment(SwingConstants.TRAILING);
		textXBins.setText("512");
		textXBins.addKeyListener(new WaitForKeyListener(500, new Runnable(){
			@Override
			public void run() {
				fireChanged();
			}
		}));
		
		textYBins = new JTextField();
		textYBins.setHorizontalAlignment(SwingConstants.TRAILING);
		textYBins.setText("512");
		textYBins.addKeyListener(new WaitForKeyListener(500, new Runnable(){
			@Override
			public void run() {
				fireChanged();
			}
		}));
		
		labelX2 = new JLabel("100");
		
		labelY2 = new JLabel("100");
		
		lblRanges = new JLabel("Ranges");
		GroupLayout groupLayout = new GroupLayout(this);
		groupLayout.setHorizontalGroup(
			groupLayout.createParallelGroup(Alignment.TRAILING)
				.addGroup(groupLayout.createSequentialGroup()
					.addContainerGap()
					.addGroup(groupLayout.createParallelGroup(Alignment.LEADING)
						.addGroup(groupLayout.createSequentialGroup()
							.addGroup(groupLayout.createParallelGroup(Alignment.TRAILING)
								.addGroup(groupLayout.createParallelGroup(Alignment.LEADING, false)
									.addGroup(groupLayout.createSequentialGroup()
										.addComponent(lblYBins)
										.addPreferredGap(ComponentPlacement.RELATED)
										.addComponent(textYBins))
									.addGroup(groupLayout.createSequentialGroup()
										.addComponent(lblXBins)
										.addPreferredGap(ComponentPlacement.RELATED)
										.addComponent(textXBins, GroupLayout.PREFERRED_SIZE, 70, GroupLayout.PREFERRED_SIZE)))
								.addGroup(groupLayout.createSequentialGroup()
									.addGap(41)
									.addGroup(groupLayout.createParallelGroup(Alignment.LEADING)
										.addComponent(lblY, GroupLayout.DEFAULT_SIZE, 46, Short.MAX_VALUE)
										.addComponent(lblX, GroupLayout.DEFAULT_SIZE, 46, Short.MAX_VALUE))
									.addPreferredGap(ComponentPlacement.UNRELATED)
									.addGroup(groupLayout.createParallelGroup(Alignment.LEADING, false)
										.addComponent(labelY2, GroupLayout.DEFAULT_SIZE, GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
										.addComponent(labelX2, GroupLayout.PREFERRED_SIZE, 74, GroupLayout.PREFERRED_SIZE))))
							.addGap(265))
						.addGroup(groupLayout.createSequentialGroup()
							.addComponent(lblRanges)
							.addContainerGap(383, Short.MAX_VALUE))))
		);
		groupLayout.setVerticalGroup(
			groupLayout.createParallelGroup(Alignment.LEADING)
				.addGroup(groupLayout.createSequentialGroup()
					.addContainerGap()
					.addComponent(lblRanges)
					.addGap(7)
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
					.addContainerGap(159, Short.MAX_VALUE))
		);
		setLayout(groupLayout);
		settings.put(RendererFactory.KEY_xmin, new Double(0));
		settings.put(RendererFactory.KEY_ymin, new Double(0));
		settings.put(RendererFactory.KEY_xmax, new Double(100));
		settings.put(RendererFactory.KEY_ymax, new Double(100));
		settings.put(RendererFactory.KEY_xBins,new Integer(512));
		settings.put(RendererFactory.KEY_yBins,new Integer(512));
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
		final GaussRendererPanel tp = new GaussRendererPanel( );
		final JFrame frame = new JFrame();
		frame.getContentPane().add( tp );
		frame.setDefaultCloseOperation( WindowConstants.DISPOSE_ON_CLOSE );
		frame.pack();
		frame.setVisible( true );
	}
}
