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

	private static final long serialVersionUID = -3031663211936690561L;
	private final JTextField textXBins;
	private final JTextField textYBins;
	private final JLabel lblX;
	private final JLabel lblY;
	private final JLabel labelX2;
	private final JLabel labelY2;
	private double zmin,zmax;

	public HistogramRendererPanel() {
		setBorder(null);
		
		lblX = new JLabel("0");
		
		lblY = new JLabel("0");
		
		JLabel lblXBins = new JLabel("X Bins");
		
		JLabel lblYBins = new JLabel("Y Bins");
		
		textXBins = new JTextField();
		textXBins.setHorizontalAlignment(SwingConstants.TRAILING);
		textXBins.setText("500");
		textXBins.addKeyListener(new WaitForKeyListener(500, new Runnable() {
			@Override
			public void run() {
				fireChanged();
			}
		}));
		
		textYBins = new JTextField();
		textYBins.setHorizontalAlignment(SwingConstants.TRAILING);
		textYBins.setText("500");
		textYBins.addKeyListener(new WaitForKeyListener(500, new Runnable() {
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
		zmin=0;
		zmax=255;
	}

	@Override
	public void setSettings(Map<String, Object> settings) {
		lblX.setText(String.format("%.4f",(double)settings.get(RendererFactory.KEY_xmin)));
		lblY.setText(String.format("%.4f",(double)settings.get(RendererFactory.KEY_ymin)));
		labelX2.setText(String.format("%.4f",(double)settings.get(RendererFactory.KEY_xmax)));
		labelY2.setText(String.format("%.4f",(double)settings.get(RendererFactory.KEY_ymax)));
		textXBins.setText(String.valueOf(settings.get(RendererFactory.KEY_xBins)));
		textYBins.setText(String.valueOf(settings.get(RendererFactory.KEY_yBins)));
		zmin = (Double) settings.get(RendererFactory.KEY_zmin);
		zmax = (Double) settings.get(RendererFactory.KEY_zmax);
	}

	@Override
	public Map<String, Object> getSettings() {
		final Map<String, Object> settings = new HashMap<>(8);
		settings.put(RendererFactory.KEY_xmin, Double.parseDouble(lblX.getText()));
		settings.put(RendererFactory.KEY_ymin, Double.parseDouble(lblY.getText()));
		settings.put(RendererFactory.KEY_xmax, Double.parseDouble(labelX2.getText()));
		settings.put(RendererFactory.KEY_ymax, Double.parseDouble(labelY2.getText()));
		settings.put(RendererFactory.KEY_xBins, Integer.parseInt(textXBins.getText()));
		settings.put(RendererFactory.KEY_yBins, Integer.parseInt(textYBins.getText()));
		settings.put(RendererFactory.KEY_zmin, zmin);
		settings.put(RendererFactory.KEY_zmax, zmax);
		return settings;
	}
	
	public static Map<String, Object> getInitialSettings(){
		final Map<String, Object> settings = new HashMap<>(8);
		settings.put(RendererFactory.KEY_xmin, 0d);
		settings.put(RendererFactory.KEY_ymin, 0d);
		settings.put(RendererFactory.KEY_xmax, 100d);
		settings.put(RendererFactory.KEY_ymax, 100d);
		settings.put(RendererFactory.KEY_xBins, 500);
		settings.put(RendererFactory.KEY_yBins, 500);
		settings.put(RendererFactory.KEY_zmin, 0d);
		settings.put(RendererFactory.KEY_zmax, 255d);
		return settings;
	}

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
