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
import org.lemming.tools.WaitForChangeListener;
import org.lemming.tools.WaitForKeyListener;


public class NMSDetectorPanel extends ConfigurationPanel {

	public NMSDetectorPanel() {
		setBorder(null);
		
		JLabel lblWindowSize = new JLabel("Threshold");
		
		jTextFieldThreshold = new JTextField();
		jTextFieldThreshold.addKeyListener(new WaitForKeyListener(500, new Runnable(){
			@Override
			public void run() {
				fireChanged();
			}
		}));
		jTextFieldThreshold.setHorizontalAlignment(SwingConstants.RIGHT);
		jTextFieldThreshold.setText("100");
		
		JLabel lblStepsize = new JLabel("StepSize");
		
		spinnerStepSize = new JSpinner();
		spinnerStepSize.addChangeListener(new WaitForChangeListener(500, new Runnable(){
			@Override
			public void run() {
				fireChanged();
			}
		}));
		spinnerStepSize.setModel(new SpinnerNumberModel(new Integer(10), new Integer(1), null, new Integer(1)));
		
		JLabel labelGaussian = new JLabel("Gaussian");
		
		spinnerGaussian = new JSpinner();
		spinnerGaussian.addChangeListener(new WaitForChangeListener(500, new Runnable(){
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
						.addComponent(lblStepsize)
						.addComponent(labelGaussian, GroupLayout.PREFERRED_SIZE, 43, GroupLayout.PREFERRED_SIZE))
					.addPreferredGap(ComponentPlacement.RELATED)
					.addGroup(groupLayout.createParallelGroup(Alignment.LEADING)
						.addGroup(groupLayout.createSequentialGroup()
							.addComponent(spinnerGaussian, GroupLayout.PREFERRED_SIZE, 67, GroupLayout.PREFERRED_SIZE)
							.addContainerGap(272, Short.MAX_VALUE))
						.addGroup(groupLayout.createSequentialGroup()
							.addGroup(groupLayout.createParallelGroup(Alignment.LEADING)
								.addComponent(spinnerStepSize)
								.addComponent(jTextFieldThreshold, GroupLayout.PREFERRED_SIZE, 67, GroupLayout.PREFERRED_SIZE))
							.addContainerGap(282, Short.MAX_VALUE))))
		);
		groupLayout.setVerticalGroup(
			groupLayout.createParallelGroup(Alignment.TRAILING)
				.addGroup(Alignment.LEADING, groupLayout.createSequentialGroup()
					.addContainerGap()
					.addGroup(groupLayout.createParallelGroup(Alignment.BASELINE)
						.addComponent(jTextFieldThreshold, GroupLayout.PREFERRED_SIZE, 36, GroupLayout.PREFERRED_SIZE)
						.addComponent(lblWindowSize, GroupLayout.PREFERRED_SIZE, 28, GroupLayout.PREFERRED_SIZE))
					.addPreferredGap(ComponentPlacement.RELATED)
					.addGroup(groupLayout.createParallelGroup(Alignment.BASELINE)
						.addComponent(spinnerStepSize, GroupLayout.PREFERRED_SIZE, 32, GroupLayout.PREFERRED_SIZE)
						.addComponent(lblStepsize))
					.addPreferredGap(ComponentPlacement.UNRELATED)
					.addGroup(groupLayout.createParallelGroup(Alignment.BASELINE)
						.addComponent(labelGaussian)
						.addComponent(spinnerGaussian, GroupLayout.PREFERRED_SIZE, 28, GroupLayout.PREFERRED_SIZE))
					.addContainerGap(190, Short.MAX_VALUE))
		);
		setLayout(groupLayout);
	}

	/**
	 * 
	 */
	private static final long serialVersionUID = -4601480448696314069L;
    private JTextField jTextFieldThreshold;
	private JSpinner spinnerStepSize;
	public static final String KEY_NMS_STEPSIZE = "NMS_STEPSIZE";
	public static final String KEY_NMS_THRESHOLD = "NMS_THRESHOLD";
    public static final String KEY_NMS_GAUSSIAN_SIZE = "NMS_GAUSSIAN_SIZE";
	private JSpinner spinnerGaussian;

	@Override
	public void setSettings(Map<String, Object> settings) {
		spinnerStepSize.setValue(settings.get(KEY_NMS_STEPSIZE));
		jTextFieldThreshold.setText(""+settings.get(KEY_NMS_THRESHOLD));
        spinnerGaussian.setValue(settings.get(KEY_NMS_GAUSSIAN_SIZE));
	}

	@Override
	public Map<String, Object> getSettings() {
		final Map< String, Object > settings = new HashMap<>( 3 );
		final int stepsize = (int) spinnerStepSize.getValue();
		final double threshold = Double.parseDouble( jTextFieldThreshold.getText() );
		final int gaussianSize = (int) spinnerGaussian.getValue();
		settings.put( KEY_NMS_STEPSIZE, stepsize );
		settings.put( KEY_NMS_THRESHOLD, threshold );
		settings.put( KEY_NMS_GAUSSIAN_SIZE, gaussianSize);
		return settings;
	}
}
