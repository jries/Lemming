package org.lemming.pipeline;

import java.beans.PropertyChangeEvent;
import java.beans.PropertyChangeListener;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import javax.swing.SwingWorker;

import org.lemming.interfaces.Store;

import ij.IJ;

public class Manager extends SwingWorker<Void,Void> {
	
	private Map<Integer,Store> storeMap = new LinkedHashMap<>();
	private Map<Integer,AbstractModule> modules = new LinkedHashMap<>();
	private int counter;
	private boolean done = false;
	private int maximum = 1;

	public Manager() {
		counter = 0;
	}
	
	public void add(AbstractModule module){		
		modules.put(module.hashCode(),module);		
	}
	
	public void linkModules(AbstractModule current, AbstractModule fromOrTo , boolean noInputs ){
		if (noInputs){
			Store s = new FastStore();
			AbstractModule cur = modules.get(current.hashCode());
			if (cur==null) throw new NullPointerException("Wrong linkage!");
			cur.setOutput(s);
			AbstractModule well = modules.get(fromOrTo.hashCode());
			if (well==null) throw new NullPointerException("Wrong linkage!");
			well.setInput(s);
			storeMap.put(s.hashCode(), s);
			return;
		}
		
		AbstractModule well = modules.get(fromOrTo.hashCode());
		if (well==null) throw new NullPointerException("Wrong linkage!");
		Store s = new FastStore();
		well.setInput(s);
		AbstractModule cur = modules.get(current.hashCode());
		if (cur==null) throw new NullPointerException("Wrong linkage!");
		cur.setOutput(s);
		storeMap.put(s.hashCode(), s);
	}
	
	public void linkModules(AbstractModule from , AbstractModule to ){
		AbstractModule source = modules.get(from.hashCode());
		if (source==null) throw new NullPointerException("Wrong linkage!");
		Store s = new FastStore();
		source.setOutput(s);
		AbstractModule current = modules.get(to.hashCode());
		if (current==null) throw new NullPointerException("Wrong linkage!");
		current.setInput(s);
		storeMap.put(s.hashCode(), s);
	}
	
	public void step(){
		if (counter >= modules.values().toArray().length) return;
		AbstractModule m = (AbstractModule) modules.values().toArray()[counter];
		if (!m.check()) return;
		Thread t = new Thread(m, m.getClass().getSimpleName());
		t.start();
		try {
			t.join();
		} catch (InterruptedException e) {
			System.err.println(e.getMessage());
		}
		counter++;
	}
	
	public Map<Integer, Store> getMap(){
		return storeMap;
	}
	
	@Override
	protected Void doInBackground() throws Exception {
		if (modules.isEmpty()) return null;
		StoreMonitor sm = new StoreMonitor();
		sm.addPropertyChangeListener(new PropertyChangeListener(){
			@Override
			public void propertyChange(PropertyChangeEvent evt) {
				if (evt.getPropertyName().equals("progress")) 
					setProgress((int) evt.getNewValue());
			}});
		sm.execute();
		final List<Thread> threads= new ArrayList<>();
		
		for(AbstractModule starter:modules.values()){
			if (!starter.check()) {
				IJ.error("Module not linked properly " + starter.getClass().getSimpleName());
				break;
			}	
//			starter.setService(service);

			Thread t = new Thread(starter, starter.getClass().getSimpleName());
			t.start();
			threads.add(t);	
			
			try {
				Thread.sleep(100); 						// HACK : give the module some time to start working
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
		try {
			for(Thread joiner:threads)
				joiner.join();

		} catch (InterruptedException e) {
			System.err.println(e.getMessage());
		}
		done = true;
		return null;
	}

	public void reset() {
		modules.clear();
		storeMap.clear();
		counter=0;
	}
	
	
	class StoreMonitor extends SwingWorker<Void,Integer> {

		public StoreMonitor() {
		}

		@Override
		protected Void doInBackground() throws Exception {
			while(!done){
				try {
	                Thread.sleep(200);
	            } catch (InterruptedException ignore) {}
				int max = 0;
				int n = 0;
				for(Integer key : storeMap.keySet()){
					n = storeMap.get(key).getLength();
					max= Math.max(n, max);
				}
				if (max > maximum)
					maximum = max;
				
				setProgress(Math.round(100-(float)max/maximum*100));
				}
			return null;
		}

	}


}
