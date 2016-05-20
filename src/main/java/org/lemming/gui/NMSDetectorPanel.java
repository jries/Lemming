package org.lemming.gui;

import java.util.HashMap;
import java.util.Map;

import javax.swing.JLabel;
import javax.swing.JSpinner;
import javax.swing.SpinnerNumberModel;
import javax.swing.GroupLayout;
import javax.swing.GroupLayout.Alignment;
import javax.swing.LayoutStyle.ComponentPlacement;
import org.lemming.tools.WaitForChangeListener;
import java.awt.Font;


public class NMSDetectorPanel extends ConfigurationPanel {

	public NMSDetectorPanel() {
		setBorder(null);
		
		JLabel lblWindowSize = new JLabel("Threshold");
		
		spinnerThreshold = new JSpinner();
		spinnerThreshold.setModel(new SpinnerNumberModel(10, 1, null, 1));
		spinnerThreshold.addChangeListener(new WaitForChangeListener(500, new Runnable() {
			@Override
			public void run() {
				fireChanged();
			}
		}));
		
		JLabel lblStepsize = new JLabel("StepSize");
		
		spinnerStepSize = new JSpinner();
		spinnerStepSize.setFont(new Font("Dialog", Font.PLAIN, 12));
		spinnerStepSize.addChangeListener(new WaitForChangeListener(500, new Runnable() {
			@Override
			public void run() {
				fireChanged();
			}
		}));
		spinnerStepSize.setModel(new SpinnerNumberModel(10, 1, null, 1));
		
		JLabel labelGaussian = new JLabel("Gaussian");
		
		spinnerGaussian = new JSpinner();
		spinnerGaussian.addChangeListener(new WaitForChangeListener(500, new Runnable() {
			@Override
			public void run() {
				fireChanged();
			}
		}));
		spinnerGaussian.setToolTipText("Prefilter with Gaussian (0=no filtering)");
		GroupLayout groupLayout = new GroupLayout(this);
		groupLayout.setHorizontalGroup(
			groupLayout.createParallelGroup(Alignment.LEADING)
				.addGroup(groupLayout.createSequentialGroup()
					.addContainerGap()
					.addGroup(groupLayout.createParallelGroup(Alignment.LEADING)
						.addComponent(lblWindowSize, GroupLayout.PREFERRED_SIZE, 97, GroupLayout.PREFERRED_SIZE)
						.addGroup(groupLayout.createParallelGroup(Alignment.TRAILING, false)
							.addComponent(lblStepsize, Alignment.LEADING, GroupLayout.DEFAULT_SIZE, GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
							.addComponent(labelGaussian, Alignment.LEADING, GroupLayout.DEFAULT_SIZE, 82, Short.MAX_VALUE)))
					.addPreferredGap(ComponentPlacement.RELATED)
					.addGroup(groupLayout.createParallelGroup(Alignment.LEADING, false)
						.addComponent(spinnerGaussian)
						.addComponent(spinnerThreshold, GroupLayout.DEFAULT_SIZE, 67, Short.MAX_VALUE)
						.addComponent(spinnerStepSize))
					.addContainerGap(262, Short.MAX_VALUE))
		);
		groupLayout.setVerticalGroup(
			groupLayout.createParallelGroup(Alignment.LEADING)
				.addGroup(groupLayout.createSequentialGroup()
					.addContainerGap()
					.addGroup(groupLayout.createParallelGroup(Alignment.BASELINE)
						.addComponent(spinnerThreshold, GroupLayout.PREFERRED_SIZE, 30, GroupLayout.PREFERRED_SIZE)
						.addComponent(lblWindowSize, GroupLayout.PREFERRED_SIZE, 28, GroupLayout.PREFERRED_SIZE))
					.addPreferredGap(ComponentPlacement.RELATED)
					.addGroup(groupLayout.createParallelGroup(Alignment.BASELINE)
						.addComponent(spinnerStepSize, GroupLayout.PREFERRED_SIZE, 32, GroupLayout.PREFERRED_SIZE)
						.addComponent(lblStepsize))
					.addPreferredGap(ComponentPlacement.UNRELATED)
					.addGroup(groupLayout.createParallelGroup(Alignment.BASELINE)
						.addComponent(labelGaussian)
						.addComponent(spinnerGaussian, GroupLayout.PREFERRED_SIZE, 28, GroupLayout.PREFERRED_SIZE))
					.addContainerGap(174, Short.MAX_VALUE))
		);
		setLayout(groupLayout);
	}

	/**
	 * 
	 */
	private static final long serialVersionUID = -4601480448696314069L;
    private final JSpinner spinnerThreshold;
	private final JSpinner spinnerStepSize;
	public static final String KEY_NMS_STEPSIZE = "NMS_STEPSIZE";
	public static final String KEY_NMS_THRESHOLD = "NMS_THRESHOLD";
    public static final String KEY_NMS_GAUSSIAN_SIZE = "NMS_GAUSSIAN_SIZE";
	private final JSpinner spinnerGaussian;

	@Override
	public void setSettings(Map<String, Object> settings) {
		spinnerStepSize.setValue(settings.get(KEY_NMS_STEPSIZE));
		spinnerThreshold.setValue(settings.get(KEY_NMS_THRESHOLD));
        spinnerGaussian.setValue(settings.get(KEY_NMS_GAUSSIAN_SIZE));
	}

	@Override
	public Map<String, Object> getSettings() {
		final Map< String, Object > settings = new HashMap<>( 3 );
		final int stepsize = (int) spinnerStepSize.getValue();
		final int gaussianSize = (int) spinnerGaussian.getValue();
		final int threshold = (int) spinnerThreshold.getValue();
		settings.put( KEY_NMS_STEPSIZE, stepsize );
		settings.put( KEY_NMS_THRESHOLD, threshold );
		settings.put( KEY_NMS_GAUSSIAN_SIZE, gaussianSize);
		return settings;
	}
}
