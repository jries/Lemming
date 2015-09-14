package org.lemming.gui;

import java.util.ArrayDeque;
import java.util.Deque;
import java.util.Map;

import javax.swing.GroupLayout;
import javax.swing.GroupLayout.Alignment;
import javax.swing.JScrollPane;
import javax.swing.ScrollPaneConstants;

import java.awt.Dimension;

import javax.swing.JButton;
import javax.swing.LayoutStyle.ComponentPlacement;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

import org.lemming.pipeline.ExtendableTable;

import java.awt.Component;
import java.awt.event.ActionListener;
import java.awt.event.ActionEvent;

public class FilterPanel extends ConfigurationPanel implements ActionListener, ChangeListener {

	/**
	 * 
	 */
	private static final long serialVersionUID = -2042228716255813527L;
	private final ChangeEvent CHANGE_EVENT = new ChangeEvent( this );
	private JButton btnAdd;
	private JButton btnRemove;
	private JScrollPane scrollPane;
	private ExtendableTable table;
	private Deque<HistogramPanel> panelStack = new ArrayDeque<>();

	public FilterPanel(ExtendableTable table) {
		this.table = table;
		setBorder(null);
		
		scrollPane = new JScrollPane();
		scrollPane.setMaximumSize(new Dimension(1000, 125));
		scrollPane.setPreferredSize(new Dimension(260, 260));
		scrollPane.setOpaque(true);
		scrollPane.setAutoscrolls(true);
		scrollPane.setVerticalScrollBarPolicy(ScrollPaneConstants.VERTICAL_SCROLLBAR_AS_NEEDED);
		scrollPane.setHorizontalScrollBarPolicy(ScrollPaneConstants.HORIZONTAL_SCROLLBAR_NEVER);
		scrollPane.setEnabled(true);
		
		btnAdd = new JButton("Add");
		btnAdd.addActionListener(this);
		btnAdd.setAlignmentY(Component.TOP_ALIGNMENT);
		
		btnRemove = new JButton("Remove");
		btnRemove.addActionListener(this);
		GroupLayout groupLayout = new GroupLayout(this);
		groupLayout.setHorizontalGroup(
			groupLayout.createParallelGroup(Alignment.LEADING)
				.addGroup(groupLayout.createSequentialGroup()
					.addContainerGap()
					.addGroup(groupLayout.createParallelGroup(Alignment.LEADING)
						.addComponent(scrollPane, GroupLayout.PREFERRED_SIZE, GroupLayout.DEFAULT_SIZE, GroupLayout.PREFERRED_SIZE)
						.addGroup(groupLayout.createSequentialGroup()
							.addComponent(btnAdd)
							.addPreferredGap(ComponentPlacement.RELATED)
							.addComponent(btnRemove)))
					.addContainerGap(184, Short.MAX_VALUE))
		);
		groupLayout.setVerticalGroup(
			groupLayout.createParallelGroup(Alignment.LEADING)
				.addGroup(groupLayout.createSequentialGroup()
					.addContainerGap()
					.addComponent(scrollPane, GroupLayout.PREFERRED_SIZE, GroupLayout.DEFAULT_SIZE, GroupLayout.PREFERRED_SIZE)
					.addPreferredGap(ComponentPlacement.RELATED)
					.addGroup(groupLayout.createParallelGroup(Alignment.BASELINE)
						.addComponent(btnAdd)
						.addComponent(btnRemove))
					.addContainerGap(GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
		);
		setLayout(groupLayout);
	}

	@Override
	public void setSettings(Map<String, Object> settings) {

	}

	@Override
	public Map<String, Object> getSettings() {
		return null;
	}

	@Override
	public void actionPerformed(ActionEvent e) {
		Object s = e.getSource();
		
		if (s == this.btnAdd){
			HistogramPanel hPanel = new HistogramPanel(table,0);
			scrollPane.add(hPanel);
			panelStack.add(hPanel);
			stateChanged( CHANGE_EVENT );

		}
		
		if (s == this.btnRemove){
			if (!panelStack.isEmpty())
				scrollPane.remove(panelStack.removeLast());
			stateChanged( CHANGE_EVENT );
		}
	}

	@Override
	public void stateChanged(ChangeEvent e) {
		
	}
}
