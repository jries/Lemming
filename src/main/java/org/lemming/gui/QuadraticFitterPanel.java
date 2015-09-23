package org.lemming.gui;

import java.util.HashMap;
import java.util.Map;

import javax.swing.GroupLayout;
import javax.swing.GroupLayout.Alignment;
import javax.swing.JLabel;
import javax.swing.JSpinner;
import javax.swing.LayoutStyle.ComponentPlacement;
import javax.swing.SpinnerNumberModel;
import java.io.File;
import javax.swing.event.ChangeListener;
import javax.swing.event.ChangeEvent;

public class QuadraticFitterPanel extends ConfigurationPanel {

	/**
	 * 
	 */
	private static final long serialVersionUID = 3081886846323191618L;
	public static final String KEY_QUAD_WINDOW_SIZE = "QUAD_WINDOW_SIZE";
	public static final String KEY_QUAD_QUEUE_SIZE = "QUAD_QUEUE_SIZE";
	private JSpinner spinnerWindowSize;
	private JSpinner spinnerQueueSize;
	protected File calibFile;
	protected File camFile;

	public QuadraticFitterPanel() {
		setBorder(null);
		
		JLabel lblWindowSize = new JLabel("Window Size");
		
		spinnerWindowSize = new JSpinner();
		spinnerWindowSize.addChangeListener(new ChangeListener() {
			public void stateChanged(ChangeEvent e) {
				fireChanged();
			}
		});
		spinnerWindowSize.setModel(new SpinnerNumberModel(new Integer(10), null, null, new Integer(1)));
		
		JLabel lblQueueSize = new JLabel("Queue Size");
		
		spinnerQueueSize = new JSpinner();
		spinnerQueueSize.setModel(new SpinnerNumberModel(new Integer(60), null, null, new Integer(1)));
		
		GroupLayout groupLayout = new GroupLayout(this);
		groupLayout.setHorizontalGroup(
			groupLayout.createParallelGroup(Alignment.LEADING)
				.addGroup(groupLayout.createSequentialGroup()
					.addContainerGap()
					.addGroup(groupLayout.createParallelGroup(Alignment.LEADING)
						.addComponent(lblWindowSize)
						.addComponent(lblQueueSize))
					.addGap(18)
					.addGroup(groupLayout.createParallelGroup(Alignment.LEADING, false)
						.addComponent(spinnerQueueSize)
						.addComponent(spinnerWindowSize, GroupLayout.DEFAULT_SIZE, 62, Short.MAX_VALUE))
					.addContainerGap(286, Short.MAX_VALUE))
		);
		groupLayout.setVerticalGroup(
			groupLayout.createParallelGroup(Alignment.LEADING)
				.addGroup(groupLayout.createSequentialGroup()
					.addContainerGap()
					.addGroup(groupLayout.createParallelGroup(Alignment.BASELINE)
						.addComponent(lblWindowSize)
						.addComponent(spinnerWindowSize, GroupLayout.PREFERRED_SIZE, GroupLayout.DEFAULT_SIZE, GroupLayout.PREFERRED_SIZE))
					.addPreferredGap(ComponentPlacement.UNRELATED)
					.addGroup(groupLayout.createParallelGroup(Alignment.BASELINE)
						.addComponent(lblQueueSize)
						.addComponent(spinnerQueueSize, GroupLayout.PREFERRED_SIZE, GroupLayout.DEFAULT_SIZE, GroupLayout.PREFERRED_SIZE))
					.addContainerGap(221, Short.MAX_VALUE))
		);
		setLayout(groupLayout);
	}

	@Override
	public void setSettings(Map<String, Object> settings) {
		spinnerWindowSize.setValue(settings.get(KEY_QUAD_WINDOW_SIZE));
		spinnerQueueSize.setValue(settings.get(KEY_QUAD_QUEUE_SIZE));
	}

	@Override
	public Map<String, Object> getSettings() {
		final Map< String, Object > settings = new HashMap<>( 2 );
		settings.put(KEY_QUAD_WINDOW_SIZE, spinnerWindowSize.getValue());
		settings.put(KEY_QUAD_QUEUE_SIZE, spinnerQueueSize.getValue());
		return settings;
	}
}
