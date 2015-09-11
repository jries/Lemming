package org.lemming.gui;

import java.util.HashMap;
import java.util.Map;

import javax.swing.GroupLayout;
import javax.swing.JLabel;
import javax.swing.JSpinner;
import javax.swing.JTextField;
import javax.swing.SpinnerNumberModel;
import javax.swing.SwingConstants;
import javax.swing.GroupLayout.Alignment;
import javax.swing.LayoutStyle.ComponentPlacement;

public class PeakFinderPanel extends ConfigurationPanel {

	/**
	 * 
	 */
	private static final long serialVersionUID = 7807972034055388559L;
	private JTextField jTextFieldThreshold;
	private JSpinner spinnerKernelSize;
	public static final String KEY_KERNEL_SIZE = "KERNEL_SIZE";
	public static final String KEY_THRESHOLD = "THRESHOLD";

	public PeakFinderPanel() {
		setBorder(null);
		
		JLabel lblWindowSize = new JLabel("Threshold");
		
		jTextFieldThreshold = new JTextField();
		jTextFieldThreshold.setHorizontalAlignment(SwingConstants.RIGHT);
		jTextFieldThreshold.setText("0");
		
		JLabel lblKernelSize = new JLabel("KernelSize");
		
		spinnerKernelSize = new JSpinner();
		spinnerKernelSize.setModel(new SpinnerNumberModel(new Integer(1), new Integer(1), null, new Integer(1)));
		GroupLayout groupLayout = new GroupLayout(this);
		groupLayout.setHorizontalGroup(
			groupLayout.createParallelGroup(Alignment.LEADING)
				.addGroup(groupLayout.createSequentialGroup()
					.addGap(26)
					.addGroup(groupLayout.createParallelGroup(Alignment.LEADING)
						.addComponent(lblWindowSize, GroupLayout.PREFERRED_SIZE, 97, GroupLayout.PREFERRED_SIZE)
						.addComponent(lblKernelSize))
					.addPreferredGap(ComponentPlacement.RELATED)
					.addGroup(groupLayout.createParallelGroup(Alignment.LEADING, false)
						.addComponent(spinnerKernelSize)
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
						.addComponent(spinnerKernelSize, GroupLayout.PREFERRED_SIZE, 35, GroupLayout.PREFERRED_SIZE)
						.addComponent(lblKernelSize))
					.addContainerGap(72, Short.MAX_VALUE))
		);
		setLayout(groupLayout);
	}

	@Override
	public void setSettings(Map<String, Object> settings) {
		spinnerKernelSize.setValue(settings.get(KEY_KERNEL_SIZE));
		jTextFieldThreshold.setText(""+settings.get(KEY_THRESHOLD)); 

	}

	@Override
	public Map<String, Object> getSettings() {
		final Map< String, Object > settings = new HashMap<>( 2 );
		final int stepsize = (int) spinnerKernelSize.getValue();
		final double threshold = Double.parseDouble( jTextFieldThreshold.getText() );
		settings.put( KEY_KERNEL_SIZE, stepsize );
		settings.put( KEY_THRESHOLD, threshold );
		
		return settings;
	}

}
