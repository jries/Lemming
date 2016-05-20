package org.lemming.gui;

import java.io.File;
import java.util.ArrayDeque;
import java.util.Deque;
import java.util.HashMap;
import java.util.Map;

import javax.swing.JFrame;
import javax.swing.JScrollPane;
import javax.swing.ScrollPaneConstants;
import javax.swing.WindowConstants;

import java.awt.Dimension;

import javax.swing.JButton;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

import org.lemming.modules.TableLoader;
import org.lemming.pipeline.ExtendableTable;

import java.awt.Component;
import java.awt.event.ActionListener;
import java.awt.event.ActionEvent;

import javax.swing.JPanel;

import java.awt.FlowLayout;

import javax.swing.BoxLayout;

public class FilterPanel extends ConfigurationPanel implements ChangeListener {

	private static final long serialVersionUID = -2042228716255813527L;
	public static final String KEY = "FILTER";
	private ExtendableTable table;
	private final Deque<HistogramPanel> panelStack = new ArrayDeque<>();
	private final ChangeEvent CHANGE_EVENT = new ChangeEvent( this );
	private final JPanel panelHolder;

	public FilterPanel() {
		setBorder(null);
		setMinimumSize(new Dimension(295, 315));
		setPreferredSize(new Dimension(300, 340));
		setName("FILTER");

		JScrollPane scrollPane = new JScrollPane();
		scrollPane.setPreferredSize(new Dimension(290, 300));
		scrollPane.setOpaque(true);
		scrollPane.setAutoscrolls(true);
		scrollPane.setVerticalScrollBarPolicy(ScrollPaneConstants.VERTICAL_SCROLLBAR_ALWAYS);
		scrollPane.setHorizontalScrollBarPolicy(ScrollPaneConstants.HORIZONTAL_SCROLLBAR_NEVER);
		scrollPane.setEnabled(true);
		
		panelHolder = new JPanel();
		scrollPane.setViewportView(panelHolder);
		panelHolder.setLayout(new BoxLayout(panelHolder, BoxLayout.Y_AXIS));

		JPanel panelButtons = new JPanel();
		FlowLayout flowLayout = (FlowLayout) panelButtons.getLayout();
		flowLayout.setHgap(0);

		JButton btnAdd = new JButton("Add");
		panelButtons.add(btnAdd);
		btnAdd.addActionListener(new ActionListener(){
			@Override
			public void actionPerformed(ActionEvent e){addPanel();}});
		btnAdd.setAlignmentY(Component.TOP_ALIGNMENT);

		JButton btnRemove = new JButton("Remove");
		panelButtons.add(btnRemove);
		setLayout(new BoxLayout(this, BoxLayout.Y_AXIS));
		add(scrollPane);
		add(panelButtons);
		btnRemove.addActionListener(new ActionListener(){
			@Override
			public void actionPerformed(ActionEvent e){removePanel();}});
		table = new ExtendableTable();
	}
	
	public void setTable(ExtendableTable table){
		this.table = table;
	}

	private void removePanel() {
		if (!panelStack.isEmpty()){
			HistogramPanel hPanel = panelStack.removeLast();
			hPanel.removeChangeListener(this);
			panelHolder.remove(hPanel);
			repaint();
			stateChanged( CHANGE_EVENT );	
		}			
	}

	private void addPanel() {
		if (table.getNames().isEmpty()) return;
		HistogramPanel hPanel = new HistogramPanel(table);
		hPanel.addChangeListener(this);
		panelStack.add(hPanel);
		panelHolder.add(hPanel);
		panelHolder.validate();
		stateChanged( CHANGE_EVENT );		
	}

	@Override
	public void stateChanged(ChangeEvent e) {
		table.filtersCollection.clear();
		for ( final HistogramPanel hp : panelStack ){
			table.addFilterMinMax(hp.getKey(), hp.getThreshold(), hp.getUpperThreshold());
		}		
		fireChanged();
	}
	
	public static void main( final String[] args )
	{
		TableLoader loader = new TableLoader(new File("/home/ronny/ownCloud/storm/testTable.csv"));
		//loader.readObjects();
		loader.readCSV(',');
		
		// Create GUI
		final FilterPanel tp = new FilterPanel();
		tp.setTable(loader.getTable());
		final JFrame frame = new JFrame();
		frame.getContentPane().add( tp );
		frame.setDefaultCloseOperation( WindowConstants.DISPOSE_ON_CLOSE );
		frame.pack();
		frame.setVisible( true );
	}

	@Override
	public void setSettings(Map<String, Object> settings) {
		// nothing
	}

	@Override
	public Map<String, Object> getSettings() {
		return new HashMap<>();
	}

}
