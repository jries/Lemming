package org.lemming.gui;

import java.util.HashMap;
import java.util.Map;

import javax.swing.GroupLayout;
import javax.swing.GroupLayout.Alignment;
import javax.swing.JLabel;
import javax.swing.JSpinner;
import javax.swing.JTextField;
import javax.swing.LayoutStyle.ComponentPlacement;
import javax.swing.SpinnerNumberModel;
import javax.swing.SwingConstants;
import org.lemming.tools.WaitForChangeListener;
import org.lemming.tools.WaitForKeyListener;

public class DoGFinderPanel extends ConfigurationPanel {

	/**
	 * 
	 */
	private static final long serialVersionUID = -273740217694586888L;
	public static final String KEY_THRESHOLD = "DOG_THRESHOLD";
	public static final String KEY_RADIUS = "RADIUS";
	private final JTextField jTextFieldThreshold;
	private final JSpinner spinnerRadius;

	public DoGFinderPanel() {
		setBorder(null);
		
		JLabel lblWindowSize = new JLabel("Threshold");
		
		jTextFieldThreshold = new JTextField();
		jTextFieldThreshold.setHorizontalAlignment(SwingConstants.RIGHT);
		jTextFieldThreshold.setText("100");
		jTextFieldThreshold.addKeyListener(new WaitForKeyListener(500, new Runnable() {
			@Override
			public void run() {
				fireChanged();
			}
		}));
		
		JLabel lblRadius = new JLabel("Radius");
		
		spinnerRadius = new JSpinner();
		spinnerRadius.addChangeListener(new WaitForChangeListener(500, new Runnable() {
			@Override
			public void run() {
				fireChanged();
			}
		}));
		spinnerRadius.setModel(new SpinnerNumberModel(7, 1, null, 1));
		GroupLayout groupLayout = new GroupLayout(this);
		groupLayout.setHorizontalGroup(
			groupLayout.createParallelGroup(Alignment.LEADING)
				.addGroup(groupLayout.createSequentialGroup()
					.addGap(26)
					.addGroup(groupLayout.createParallelGroup(Alignment.LEADING)
						.addComponent(lblWindowSize, GroupLayout.PREFERRED_SIZE, 97, GroupLayout.PREFERRED_SIZE)
						.addComponent(lblRadius))
					.addPreferredGap(ComponentPlacement.RELATED)
					.addGroup(groupLayout.createParallelGroup(Alignment.LEADING, false)
						.addComponent(spinnerRadius)
						.addComponent(jTextFieldThreshold, GroupLayout.DEFAULT_SIZE, 67, Short.MAX_VALUE))
					.addGap(63))
		);
		groupLayout.setVerticalGroup(
			groupLayout.createParallelGroup(Alignment.TRAILING)
				.addGroup(groupLayout.createSequentialGroup()
					.addGap(11)
					.addGroup(groupLayout.createParallelGroup(Alignment.BASELINE)
						.addComponent(jTextFieldThreshold, GroupLayout.PREFERRED_SIZE, 36, GroupLayout.PREFERRED_SIZE)
						.addComponent(lblWindowSize, GroupLayout.PREFERRED_SIZE, 28, GroupLayout.PREFERRED_SIZE))
					.addPreferredGap(ComponentPlacement.RELATED)
					.addGroup(groupLayout.createParallelGroup(Alignment.BASELINE)
						.addComponent(spinnerRadius, GroupLayout.PREFERRED_SIZE, 35, GroupLayout.PREFERRED_SIZE)
						.addComponent(lblRadius))
					.addContainerGap(72, Short.MAX_VALUE))
		);
		setLayout(groupLayout);
	}

	@Override
	public void setSettings(Map<String, Object> settings) {
		spinnerRadius.setValue(settings.get(KEY_RADIUS));
		jTextFieldThreshold.setText(""+settings.get(KEY_THRESHOLD)); 
	}

	@Override
	public Map<String, Object> getSettings() {
		final Map< String, Object > settings = new HashMap<>( 2 );
		final int radius = (int) spinnerRadius.getValue();
		final double threshold = Double.parseDouble( jTextFieldThreshold.getText() );
		settings.put( KEY_RADIUS, radius);
		settings.put( KEY_THRESHOLD, threshold);
		
		return settings;
	}

}
