package org.lemming.gui;

import java.io.File;
import java.util.ArrayDeque;
import java.util.Deque;
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
import java.awt.BorderLayout;

public class FilterPanel extends ConfigurationPanel implements ChangeListener {

	/**
	 * 
	 */
	private static final long serialVersionUID = -2042228716255813527L;
	
	private JButton btnAdd;
	private JButton btnRemove;
	private JScrollPane scrollPane;
	private ExtendableTable table;
	private Deque<HistogramPanel> panelStack = new ArrayDeque<>();
	private final ChangeEvent CHANGE_EVENT = new ChangeEvent( this );
	private JPanel panelHolder;
	private JPanel panelButtons;
	
	public FilterPanel(ExtendableTable table) {
		setMinimumSize(new Dimension(280, 300));
		setPreferredSize(new Dimension(320, 340));
		this.table = table;
		
		scrollPane = new JScrollPane();
		scrollPane.setPreferredSize(new Dimension(290, 300));
		scrollPane.setOpaque(true);
		scrollPane.setAutoscrolls(true);
		scrollPane.setVerticalScrollBarPolicy(ScrollPaneConstants.VERTICAL_SCROLLBAR_ALWAYS);
		scrollPane.setHorizontalScrollBarPolicy(ScrollPaneConstants.HORIZONTAL_SCROLLBAR_NEVER);
		scrollPane.setEnabled(true);
		setLayout(new BorderLayout(0, 0));
		
		panelHolder = new JPanel();
		scrollPane.setViewportView(panelHolder);
		panelHolder.setLayout(new BoxLayout(panelHolder, BoxLayout.Y_AXIS));
		add(scrollPane, BorderLayout.CENTER);
		
		panelButtons = new JPanel();
		FlowLayout flowLayout = (FlowLayout) panelButtons.getLayout();
		flowLayout.setHgap(0);
		add(panelButtons, BorderLayout.SOUTH);
		
		btnAdd = new JButton("Add");
		panelButtons.add(btnAdd);
		btnAdd.addActionListener(new ActionListener(){
			@Override
			public void actionPerformed( final ActionEvent e ){
				addPanel();
			}
		});
		btnAdd.setAlignmentY(Component.TOP_ALIGNMENT);
		
		btnRemove = new JButton("Remove");
		panelButtons.add(btnRemove);
		btnRemove.addActionListener(new ActionListener(){
			@Override
			public void actionPerformed( final ActionEvent e ){
				removePanel();
			}
		});
	}

	protected void removePanel() {
		if (!panelStack.isEmpty()){
			HistogramPanel hPanel = panelStack.removeLast();
			hPanel.removeChangeListener(this);
			panelHolder.remove(hPanel);
			repaint();
			stateChanged( CHANGE_EVENT );	
		}			
	}

	protected void addPanel() {
		HistogramPanel hPanel = new HistogramPanel(table);
		hPanel.addChangeListener(this);
		panelStack.add(hPanel);
		panelHolder.add(hPanel);
		panelHolder.revalidate();
		stateChanged( CHANGE_EVENT );		
	}

	@Override
	public void setSettings(Map<String, Object> settings) {

	}

	@Override
	public Map<String, Object> getSettings() {
		return null;
	}


	@Override
	public void stateChanged(ChangeEvent e) {
		fireChanged();
	}
	
	/**
 * Display this JPanel inside a new JFrame.
 */
	public static void main( final String[] args )
	{
		TableLoader loader = new TableLoader(new File("/Users/ronny/Documents/testTable.csv"));
		//loader.readObjects();
		loader.readCSV(',');
		
		// Create GUI
		final FilterPanel tp = new FilterPanel( loader.getTable());
		final JFrame frame = new JFrame();
		frame.getContentPane().add( tp );
		frame.setDefaultCloseOperation( WindowConstants.DISPOSE_ON_CLOSE );
		frame.pack();
		frame.setVisible( true );
	}
}
