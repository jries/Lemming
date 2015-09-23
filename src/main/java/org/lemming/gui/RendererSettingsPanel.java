package org.lemming.gui;

import java.awt.BorderLayout;
import java.awt.Dialog;
import java.awt.FlowLayout;
import java.awt.Window;

import javax.swing.JButton;
import javax.swing.JDialog;
import javax.swing.JPanel;
import javax.swing.JLabel;
import javax.swing.UIManager;
import javax.swing.JSpinner;
import javax.swing.SpinnerNumberModel;
import javax.swing.GroupLayout;
import javax.swing.GroupLayout.Alignment;
import javax.swing.LayoutStyle.ComponentPlacement;

import java.awt.event.ActionListener;
import java.util.HashMap;
import java.util.Map;
import java.awt.event.ActionEvent;

public class RendererSettingsPanel extends JDialog {

	/**
	 * 
	 */
	private static final long serialVersionUID = -47265215899943480L;
	public static final String KEY_RENDERER_WIDTH = "RENDERER_WIDTH";
	public static final String KEY_RENDERER_HEIGHT = "RENDERER_HEIGHT";
	private final JPanel contentPanel = new JPanel();
	private JSpinner spinnerWidth;
	private JSpinner spinnerHeight;


	/**
	 * Create the dialog.
	 */
	public RendererSettingsPanel() {
		setType(Window.Type.POPUP);
		setModalityType(Dialog.ModalityType.APPLICATION_MODAL);
		setDefaultCloseOperation(JDialog.DISPOSE_ON_CLOSE);
		setTitle("Settings");
		setBounds(100, 100, 183, 151);
		getContentPane().setLayout(new BorderLayout());
		contentPanel.setBorder(UIManager.getBorder("List.focusCellHighlightBorder"));
		getContentPane().add(contentPanel, BorderLayout.CENTER);
		
		JLabel lblMaxWidth = new JLabel("Max. width");
		
		JLabel lblMaxHeight = new JLabel("Max. height");
		
		spinnerWidth = new JSpinner();
		spinnerWidth.setModel(new SpinnerNumberModel(new Integer(500), null, null, new Integer(1)));
		
		spinnerHeight = new JSpinner();
		spinnerHeight.setModel(new SpinnerNumberModel(new Integer(500), null, null, new Integer(1)));
		GroupLayout gl_contentPanel = new GroupLayout(contentPanel);
		gl_contentPanel.setHorizontalGroup(
			gl_contentPanel.createParallelGroup(Alignment.LEADING)
				.addGroup(gl_contentPanel.createSequentialGroup()
					.addContainerGap()
					.addGroup(gl_contentPanel.createParallelGroup(Alignment.LEADING)
						.addComponent(lblMaxWidth)
						.addComponent(lblMaxHeight))
					.addPreferredGap(ComponentPlacement.RELATED)
					.addGroup(gl_contentPanel.createParallelGroup(Alignment.LEADING, false)
						.addComponent(spinnerHeight)
						.addComponent(spinnerWidth, GroupLayout.DEFAULT_SIZE, 71, Short.MAX_VALUE))
					.addContainerGap(85, Short.MAX_VALUE))
		);
		gl_contentPanel.setVerticalGroup(
			gl_contentPanel.createParallelGroup(Alignment.LEADING)
				.addGroup(gl_contentPanel.createSequentialGroup()
					.addContainerGap()
					.addGroup(gl_contentPanel.createParallelGroup(Alignment.BASELINE)
						.addComponent(lblMaxWidth)
						.addComponent(spinnerWidth, GroupLayout.PREFERRED_SIZE, GroupLayout.DEFAULT_SIZE, GroupLayout.PREFERRED_SIZE))
					.addPreferredGap(ComponentPlacement.RELATED)
					.addGroup(gl_contentPanel.createParallelGroup(Alignment.BASELINE)
						.addComponent(lblMaxHeight)
						.addComponent(spinnerHeight, GroupLayout.PREFERRED_SIZE, GroupLayout.DEFAULT_SIZE, GroupLayout.PREFERRED_SIZE))
					.addContainerGap(109, Short.MAX_VALUE))
		);
		contentPanel.setLayout(gl_contentPanel);
		{
			JPanel buttonPane = new JPanel();
			buttonPane.setLayout(new FlowLayout(FlowLayout.RIGHT));
			getContentPane().add(buttonPane, BorderLayout.SOUTH);
			{
				JButton okButton = new JButton("OK");
				okButton.addActionListener(new ActionListener() {
					public void actionPerformed(ActionEvent e) {
						setVisible(false);
						dispose();
					}
				});
				okButton.setActionCommand("OK");
				buttonPane.add(okButton);
				getRootPane().setDefaultButton(okButton);
			}
			{
				JButton cancelButton = new JButton("Cancel");
				cancelButton.setActionCommand("Cancel");
				buttonPane.add(cancelButton);
			}
		}
		pack();
	}
	
	public Map<String, Object> getSettings() {
		Map<String, Object> settings = new HashMap<>( 2 );
		settings.put(KEY_RENDERER_HEIGHT, spinnerHeight.getValue());
		settings.put(KEY_RENDERER_WIDTH, spinnerWidth.getValue());
		return settings;
	}

}
