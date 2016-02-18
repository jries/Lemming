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

public class FastMedianPanel extends ConfigurationPanel {

	/**
	 * 
	 */
	private static final long serialVersionUID = 3273186486718647271L;
	public static final String KEY_INTERPOLATING = "INTERPOLATING";
	public static final String KEY_FRAMES = "FRAMES";
	private JCheckBox chckbxInterpolating;
	private JSpinner spinnerFrames;

	public FastMedianPanel() {
		setBorder(null);
		
		JLabel lblFrames = new JLabel("Frames");
		
		spinnerFrames = new JSpinner();
		spinnerFrames.addChangeListener(new WaitForChangeListener(500, new Runnable(){
			@Override
			public void run() {
				fireChanged();
			}
		}));
		spinnerFrames.setModel(new SpinnerNumberModel(new Integer(50), null, null, new Integer(1)));
		
		chckbxInterpolating = new JCheckBox("Interpolation");
		GroupLayout groupLayout = new GroupLayout(this);
		groupLayout.setHorizontalGroup(
			groupLayout.createParallelGroup(Alignment.LEADING)
				.addGroup(groupLayout.createSequentialGroup()
					.addContainerGap()
					.addGroup(groupLayout.createParallelGroup(Alignment.LEADING)
						.addGroup(groupLayout.createSequentialGroup()
							.addComponent(lblFrames)
							.addPreferredGap(ComponentPlacement.RELATED)
							.addComponent(spinnerFrames, GroupLayout.PREFERRED_SIZE, 67, GroupLayout.PREFERRED_SIZE))
						.addComponent(chckbxInterpolating))
					.addContainerGap(307, Short.MAX_VALUE))
		);
		groupLayout.setVerticalGroup(
			groupLayout.createParallelGroup(Alignment.LEADING)
				.addGroup(groupLayout.createSequentialGroup()
					.addContainerGap()
					.addGroup(groupLayout.createParallelGroup(Alignment.BASELINE)
						.addComponent(lblFrames)
						.addComponent(spinnerFrames, GroupLayout.PREFERRED_SIZE, GroupLayout.DEFAULT_SIZE, GroupLayout.PREFERRED_SIZE))
					.addPreferredGap(ComponentPlacement.RELATED)
					.addComponent(chckbxInterpolating)
					.addContainerGap(243, Short.MAX_VALUE))
		);
		setLayout(groupLayout);
	}

	@Override
	public void setSettings(Map<String, Object> settings) {
		chckbxInterpolating.setSelected((boolean) settings.get(KEY_INTERPOLATING));
		spinnerFrames.setValue(settings.get(KEY_FRAMES));

	}

	@Override
	public Map<String, Object> getSettings() {
		final Map< String, Object > settings = new HashMap<>( 2 );
		final boolean interpolating = chckbxInterpolating.isSelected();
		final int frames = (int) spinnerFrames.getValue();
		settings.put(KEY_INTERPOLATING, interpolating);
		settings.put(KEY_FRAMES, frames);
		return settings;
	}
}
