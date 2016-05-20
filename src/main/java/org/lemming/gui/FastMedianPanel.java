package org.lemming.gui;

import java.util.HashMap;
import java.util.Map;
import javax.swing.GroupLayout;
import javax.swing.GroupLayout.Alignment;
import javax.swing.JLabel;
import javax.swing.JSpinner;
import javax.swing.LayoutStyle.ComponentPlacement;
import javax.swing.SpinnerNumberModel;
import javax.swing.JCheckBox;
import org.lemming.tools.WaitForChangeListener;
import org.lemming.tools.WaitForKeyListener;

import javax.swing.JTextField;

public class FastMedianPanel extends ConfigurationPanel {

	/**
	 * 
	 */
	private static final long serialVersionUID = 3273186486718647271L;
	public static final String KEY_INTERPOLATING = "INTERPOLATING";
	public static final String KEY_FRAMES = "FRAMES";
	public static final String KEY_THRESHOLD = "THRESHOLD";
	public static final String KEY_WINDOWSIZE = "WINDOWSIZE";
	private final JCheckBox chckbxInterpolating;
	private final JSpinner spinnerFrames;
	private final JTextField textFieldThreshold;
	private final JSpinner spinnerWindowSize;

	public FastMedianPanel() {
		setBorder(null);
		
		JLabel lblFrames = new JLabel("Frames");
		
		spinnerFrames = new JSpinner();
		spinnerFrames.addChangeListener(new WaitForChangeListener(500, new Runnable() {
			@Override
			public void run() {
				fireChanged();
			}
		}));
		spinnerFrames.setModel(new SpinnerNumberModel(50, null, null, 1));
		
		chckbxInterpolating = new JCheckBox("Interpolation");
		
		JLabel lblThreshold = new JLabel("Threshold");
		
		textFieldThreshold = new JTextField();
		textFieldThreshold.setText("10");
		textFieldThreshold.setColumns(10);
		textFieldThreshold.addKeyListener(new WaitForKeyListener(500, new Runnable() {
			@Override
			public void run() {
				fireChanged();
			}
		}));
		
		JLabel lblWindowsize = new JLabel("WindowSize");
		
		spinnerWindowSize = new JSpinner();
		spinnerWindowSize.setModel(new SpinnerNumberModel(15, 1, null, 1));
		spinnerWindowSize.addChangeListener(new WaitForChangeListener(500, new Runnable() {
			@Override
			public void run() {
				fireChanged();
			}
		}));
		GroupLayout groupLayout = new GroupLayout(this);
		groupLayout.setHorizontalGroup(
			groupLayout.createParallelGroup(Alignment.LEADING)
				.addGroup(groupLayout.createSequentialGroup()
					.addContainerGap()
					.addGroup(groupLayout.createParallelGroup(Alignment.LEADING)
						.addComponent(chckbxInterpolating)
						.addGroup(groupLayout.createSequentialGroup()
							.addGroup(groupLayout.createParallelGroup(Alignment.LEADING)
								.addComponent(lblWindowsize)
								.addComponent(lblFrames)
								.addComponent(lblThreshold))
							.addPreferredGap(ComponentPlacement.RELATED)
							.addGroup(groupLayout.createParallelGroup(Alignment.TRAILING, false)
								.addComponent(spinnerFrames)
								.addComponent(textFieldThreshold, 0, 0, Short.MAX_VALUE)
								.addComponent(spinnerWindowSize, GroupLayout.DEFAULT_SIZE, 49, Short.MAX_VALUE))
							.addGap(12)))
					.addContainerGap(210, GroupLayout.PREFERRED_SIZE))
		);
		groupLayout.setVerticalGroup(
			groupLayout.createParallelGroup(Alignment.LEADING)
				.addGroup(groupLayout.createSequentialGroup()
					.addContainerGap()
					.addGroup(groupLayout.createParallelGroup(Alignment.BASELINE)
						.addComponent(lblFrames)
						.addComponent(spinnerFrames, GroupLayout.PREFERRED_SIZE, GroupLayout.DEFAULT_SIZE, GroupLayout.PREFERRED_SIZE))
					.addPreferredGap(ComponentPlacement.RELATED)
					.addGroup(groupLayout.createParallelGroup(Alignment.BASELINE)
						.addComponent(lblThreshold)
						.addComponent(textFieldThreshold, GroupLayout.PREFERRED_SIZE, GroupLayout.DEFAULT_SIZE, GroupLayout.PREFERRED_SIZE))
					.addPreferredGap(ComponentPlacement.RELATED)
					.addGroup(groupLayout.createParallelGroup(Alignment.BASELINE)
						.addComponent(lblWindowsize)
						.addComponent(spinnerWindowSize, GroupLayout.PREFERRED_SIZE, GroupLayout.DEFAULT_SIZE, GroupLayout.PREFERRED_SIZE))
					.addPreferredGap(ComponentPlacement.RELATED)
					.addComponent(chckbxInterpolating)
					.addContainerGap(166, Short.MAX_VALUE))
		);
		setLayout(groupLayout);
	}

	@Override
	public void setSettings(Map<String, Object> settings) {
		chckbxInterpolating.setSelected((boolean) settings.get(KEY_INTERPOLATING));
		spinnerFrames.setValue(settings.get(KEY_FRAMES));
		spinnerWindowSize.setValue(settings.get(KEY_WINDOWSIZE));
		textFieldThreshold.setText(String.valueOf(settings.get(KEY_THRESHOLD)));
	}

	@Override
	public Map<String, Object> getSettings() {
		final Map< String, Object > settings = new HashMap<>( 4 );
		final boolean interpolating = chckbxInterpolating.isSelected();
		final int frames = (int) spinnerFrames.getValue();
		final int stepsize = (Integer) spinnerWindowSize.getValue();
		final double threshold = Double.parseDouble(textFieldThreshold.getText());
		settings.put(KEY_INTERPOLATING, interpolating);
		settings.put(KEY_FRAMES, frames);
		settings.put(KEY_WINDOWSIZE, stepsize);
		settings.put(KEY_THRESHOLD, threshold);
		return settings;
	}
}
