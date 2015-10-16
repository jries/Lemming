package org.lemming.gui;

import java.util.HashMap;
import java.util.Map;

import javax.swing.JLabel;
import javax.swing.JSpinner;
import javax.swing.SpinnerNumberModel;
import javax.swing.GroupLayout;
import javax.swing.GroupLayout.Alignment;
import javax.swing.LayoutStyle.ComponentPlacement;
import javax.swing.JTextField;
import javax.swing.SwingConstants;
import javax.swing.event.ChangeListener;
import javax.swing.event.ChangeEvent;

import org.lemming.tools.WaitForKeyListener;


public class NMSDetectorPanel extends ConfigurationPanel {
	private JTextField jTextFieldThreshold;
	private JSpinner spinnerStepSize;

	public NMSDetectorPanel() {
		setBorder(null);
		
		JLabel lblWindowSize = new JLabel("Threshold");
		
		jTextFieldThreshold = new JTextField();
		jTextFieldThreshold.addKeyListener(new WaitForKeyListener(1000, new Runnable(){
			@Override
			public void run() {
				fireChanged();
			}
		}));
		jTextFieldThreshold.setHorizontalAlignment(SwingConstants.RIGHT);
		jTextFieldThreshold.setText("100");
		
		JLabel lblStepsize = new JLabel("StepSize");
		
		spinnerStepSize = new JSpinner();
		spinnerStepSize.addChangeListener(new ChangeListener() {
			public void stateChanged(ChangeEvent e) {
				fireChanged();			}
		});
		spinnerStepSize.setModel(new SpinnerNumberModel(new Integer(10), new Integer(1), null, new Integer(1)));
		GroupLayout groupLayout = new GroupLayout(this);
		groupLayout.setHorizontalGroup(
			groupLayout.createParallelGroup(Alignment.LEADING)
				.addGroup(groupLayout.createSequentialGroup()
					.addGap(26)
					.addGroup(groupLayout.createParallelGroup(Alignment.LEADING)
						.addComponent(lblWindowSize, GroupLayout.PREFERRED_SIZE, 97, GroupLayout.PREFERRED_SIZE)
						.addComponent(lblStepsize))
					.addPreferredGap(ComponentPlacement.RELATED)
					.addGroup(groupLayout.createParallelGroup(Alignment.LEADING, false)
						.addComponent(spinnerStepSize)
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
						.addComponent(spinnerStepSize, GroupLayout.PREFERRED_SIZE, 35, GroupLayout.PREFERRED_SIZE)
						.addComponent(lblStepsize))
					.addContainerGap(72, Short.MAX_VALUE))
		);
		setLayout(groupLayout);
	}

	/**
	 * 
	 */
	private static final long serialVersionUID = -4601480448696314069L;
	public static final String KEY_NMS_STEPSIZE = "NMS_STEPSIZE";
	public static final String KEY_NMS_THRESHOLD = "NMS_THRESHOLD";

	@Override
	public void setSettings(Map<String, Object> settings) {
		spinnerStepSize.setValue(settings.get(KEY_NMS_STEPSIZE));
		jTextFieldThreshold.setText(""+settings.get(KEY_NMS_THRESHOLD));
	}

	@Override
	public Map<String, Object> getSettings() {
		final Map< String, Object > settings = new HashMap<>( 2 );
		final int stepsize = (int) spinnerStepSize.getValue();
		final double threshold = Double.parseDouble( jTextFieldThreshold.getText() );
		settings.put( KEY_NMS_STEPSIZE, stepsize );
		settings.put( KEY_NMS_THRESHOLD, threshold );
		
		return settings;
	}
}
