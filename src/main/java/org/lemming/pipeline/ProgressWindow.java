package org.lemming.pipeline;

import java.awt.HeadlessException;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;

import javax.swing.JFrame;

import org.lemming.interfaces.Store;
import javax.swing.BoxLayout;
import javax.swing.JProgressBar;
import java.awt.Dimension;
import javax.swing.Box;

public class ProgressWindow extends JFrame {

	/**
	 * 
	 */
	private static final long serialVersionUID = 243759897409202088L;
	private Map<Integer, Store> stores;
	private ScheduledExecutorService executor = Executors.newSingleThreadScheduledExecutor();
	private ScheduledFuture< ? > future;
	private Map<Integer, JProgressBar> bars = new LinkedHashMap<>();

	public ProgressWindow(Map<Integer,Store> storeMap) throws HeadlessException {
		super("Progress");
		this.stores = storeMap;
		setPreferredSize(new Dimension(300, 200));
		setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
		
		getContentPane().setLayout(new BoxLayout(getContentPane(), BoxLayout.Y_AXIS));
		
		Set<Integer> keys = stores.keySet();
		for(Integer key : keys){
			JProgressBar progressBar = new JProgressBar(0,1000);
			progressBar.setStringPainted(true);
			getContentPane().add(progressBar);
			getContentPane().add(Box.createVerticalStrut(5));
			bars.put(key, progressBar);
		}
		
		future = executor.scheduleAtFixedRate(new Runnable(){

			@Override
			public void run() {
				Set<Integer> skeys = stores.keySet();
				for(Integer key : skeys){
					JProgressBar progressBar = bars.get(key);
					int n = stores.get(key).getLength();
					if (progressBar.getMaximum()<n)
						progressBar.setMaximum(n);
					progressBar.setValue(n);
				}
				validate();
			}
			
		}, 100, 100 ,TimeUnit.MILLISECONDS);
		pack();
		setVisible(true);
	}

	
	public void cancel(){
		if (future != null && !future.isDone()) {
			future.cancel(false);
		}
		setVisible(false);
	}
}
