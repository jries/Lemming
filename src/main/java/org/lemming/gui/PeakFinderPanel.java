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

import org.lemming.tools.WaitForChangeListener;
import org.lemming.tools.WaitForKeyListener;

public class PeakFinderPanel extends ConfigurationPanel {

	/**
	 * 
	 */
	private static final long serialVersionUID = 7807972034055388559L;
	private final JTextField jTextFieldThreshold;
	private final JSpinner spinnerKernelSize;
	private final JSpinner spinnerGaussian;
	public static final String KEY_KERNEL_SIZE = "KERNEL_SIZE";
	public static final String KEY_THRESHOLD = "PEAK_THRESHOLD";
	public static final String KEY_GAUSSIAN_SIZE = "GAUSSIAN_SIZE";

	public PeakFinderPanel() {
		setBorder(null);
		
		JLabel lblWindowSize = new JLabel("Threshold");
		
		jTextFieldThreshold = new JTextField();
		jTextFieldThreshold.setToolTipText("Threshold");
		jTextFieldThreshold.setHorizontalAlignment(SwingConstants.RIGHT);
		jTextFieldThreshold.setText("100");
		jTextFieldThreshold.addKeyListener(new WaitForKeyListener(500, new Runnable() {
			@Override
			public void run() {
				fireChanged();
			}
		}));
		
		JLabel lblKernelSize = new JLabel("KernelSize");
		
		spinnerKernelSize = new JSpinner();
		spinnerKernelSize.setToolTipText("Kernel Size");
		spinnerKernelSize.addChangeListener(new WaitForChangeListener(500, new Runnable() {
			@Override
			public void run() {
				fireChanged();
			}
		}));
		spinnerKernelSize.setModel(new SpinnerNumberModel(10, 1, null, 1));
		
		JLabel lblGaussian = new JLabel("Gaussian");
		
		spinnerGaussian = new JSpinner();
		spinnerGaussian.addChangeListener(new WaitForChangeListener(500, new Runnable() {
			@Override
			public void run() {
				fireChanged();
			}
		}));
		spinnerGaussian.setToolTipText("Prefilter with Gaussian (0=no filtering)");
		spinnerGaussian.setModel(new SpinnerNumberModel(0, 0, 20, 1));
		GroupLayout groupLayout = new GroupLayout(this);
		groupLayout.setHorizontalGroup(
			groupLayout.createParallelGroup(Alignment.LEADING)
				.addGroup(groupLayout.createSequentialGroup()
					.addGap(26)
					.addGroup(groupLayout.createParallelGroup(Alignment.LEADING)
						.addComponent(lblWindowSize, GroupLayout.PREFERRED_SIZE, 97, GroupLayout.PREFERRED_SIZE)
						.addComponent(lblKernelSize)
						.addComponent(lblGaussian))
					.addPreferredGap(ComponentPlacement.RELATED)
					.addGroup(groupLayout.createParallelGroup(Alignment.LEADING, false)
						.addComponent(spinnerGaussian)
						.addComponent(spinnerKernelSize)
						.addComponent(jTextFieldThreshold, GroupLayout.DEFAULT_SIZE, 67, Short.MAX_VALUE))
					.addGap(256))
		);
		groupLayout.setVerticalGroup(
			groupLayout.createParallelGroup(Alignment.TRAILING)
				.addGroup(groupLayout.createSequentialGroup()
					.addGap(11)
					.addGroup(groupLayout.createParallelGroup(Alignment.BASELINE)
						.addComponent(jTextFieldThreshold, GroupLayout.PREFERRED_SIZE, 31, GroupLayout.PREFERRED_SIZE)
						.addComponent(lblWindowSize, GroupLayout.PREFERRED_SIZE, 28, GroupLayout.PREFERRED_SIZE))
					.addPreferredGap(ComponentPlacement.RELATED)
					.addGroup(groupLayout.createParallelGroup(Alignment.BASELINE)
						.addComponent(lblKernelSize)
						.addComponent(spinnerKernelSize, GroupLayout.PREFERRED_SIZE, 28, GroupLayout.PREFERRED_SIZE))
					.addPreferredGap(ComponentPlacement.UNRELATED)
					.addGroup(groupLayout.createParallelGroup(Alignment.BASELINE)
						.addComponent(spinnerGaussian, GroupLayout.PREFERRED_SIZE, 28, GroupLayout.PREFERRED_SIZE)
						.addComponent(lblGaussian))
					.addContainerGap(185, Short.MAX_VALUE))
		);
		setLayout(groupLayout);
	}


	@Override
	public void setSettings(Map<String, Object> map) {
		spinnerKernelSize.setValue(map.get(KEY_KERNEL_SIZE));
		jTextFieldThreshold.setText(""+map.get(KEY_THRESHOLD)); 
		spinnerGaussian.setValue(map.get(KEY_GAUSSIAN_SIZE));
	}

	@Override
	public Map<String, Object> getSettings() {
		final Map< String, Object > map = new HashMap<>( 2 );
		final int stepsize = (int) spinnerKernelSize.getValue();
		final double threshold = Double.parseDouble( jTextFieldThreshold.getText() );
		final int gaussianSize = (int) spinnerGaussian.getValue();
		map.put( KEY_KERNEL_SIZE, stepsize );
		map.put( KEY_THRESHOLD, threshold );
		map.put( KEY_GAUSSIAN_SIZE, gaussianSize);
		return map;
	}

}
