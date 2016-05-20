package org.lemming.pipeline;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.beans.PropertyChangeEvent;
import java.beans.PropertyChangeListener;

import org.lemming.interfaces.ModuleInterface;
import org.lemming.interfaces.Store;

import ij.IJ;

/**
 * a swing worker class for background execution of the added modules.
 * The needed queues for data transport are created automatically.
 * 
 * @author Ronny Sczech
 *
 */
public class Manager{

	public static final int STATE_DONE = 1;
	final private Map<Integer,Store> storeMap = new LinkedHashMap<>();
	final private Map<Integer,ModuleInterface> modules = new LinkedHashMap<>();
	private boolean done = false;
	private int maximum = 1;
	private final ExecutorService service;
	private final List< PropertyChangeListener > changeListeners = new ArrayList<>();
	private int progress;

	public Manager(ExecutorService service) {
		this.service = service;
	}

	public void addPropertyChangeListener( final PropertyChangeListener listener ){
		changeListeners.add( listener );
	}

	public void firePropertyChanged(PropertyChangeEvent e) {
		for ( final PropertyChangeListener cl : changeListeners )
			cl.propertyChange( e );
	}

	public void add(ModuleInterface module){
		modules.put(module.hashCode(),module);
	}

	public void linkModules(ModuleInterface from, ModuleInterface to, boolean noInputs, int maxElements ){
		Store s = null;
		if (noInputs){
			int n = (int) Math.min(Math.pow(2,6)*Runtime.getRuntime().availableProcessors(), maxElements*0.5); // performance tweak
			System.out.println("Manager starts with maximal "+n+" elements" );
			s = new LinkedStore(n);
		} else {
			s = new LinkedStore(maxElements);
		}
		ModuleInterface source = modules.get(from.hashCode());
		if (source==null) throw new NullPointerException("Wrong linkage!");
		source.setOutput(s);
		ModuleInterface well = modules.get(to.hashCode());
		if (well==null) throw new NullPointerException("Wrong linkage!");
		well.setInput(s);
		storeMap.put(s.hashCode(), s);
	}

	public void linkModules(ModuleInterface from , ModuleInterface to ){
		ModuleInterface source = modules.get(from.hashCode());
		if (source==null) throw new NullPointerException("Wrong linkage!");
		Store s = new LinkedStore(Integer.MAX_VALUE);
		source.setOutput(s);
		ModuleInterface well = modules.get(to.hashCode());
		if (well==null) throw new NullPointerException("Wrong linkage!");
		well.setInput(s);
		storeMap.put(s.hashCode(), s);
	}

	public Map<Integer, Store> getMap(){
		return storeMap;
	}

	private void setProgress(int i) {
		final PropertyChangeEvent EVENT_PROGRESS = new PropertyChangeEvent(this, "progress", progress, i);
		firePropertyChanged(EVENT_PROGRESS);
		progress=i;
	}

	public void reset(){
		for (ModuleInterface starter:modules.values())
			((AbstractModule)starter).reset();
		storeMap.clear();
		modules.clear();
		done = false;
		setProgress(0);
		maximum = 1;
	}

	public void run(){
		execute();
	}

	public void execute() {
		final Runner r = new Runner();
		r.start();
		try {
			r.join();
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
	}

	private class Runner extends Thread{

		@Override
		public void run() {
			if (modules.isEmpty()) return;
			StoreMonitor sm = new StoreMonitor();
			sm.start();
			final List<Object> threads= new ArrayList<>();

			for(ModuleInterface starter:modules.values()){
				if (!starter.check()) {
					IJ.error("Module not linked properly " + starter.getClass().getSimpleName());
					break;
				}
				starter.setService(service);
				threads.add(service.submit(starter));

				try {
					Thread.sleep(10); 						// HACK : give the module some time to start working
				} catch (InterruptedException e) {
					e.printStackTrace();
				}
			}
			try {
				for(Object joiner:threads)
					((Future<?>) joiner).get();
			} catch (Exception e) {
				System.err.println(e.getMessage());
			}
			done = true;
			try {
				sm.join(200);
			} catch (InterruptedException ignore) {}

			final PropertyChangeEvent EVENT_DONE = new PropertyChangeEvent(this, "state", 0, STATE_DONE);
			firePropertyChanged(EVENT_DONE);
		}
	}

	private class StoreMonitor extends Thread {

		@Override
		public void run() {
			while(!done){
				try {
					Thread.sleep(200);
				} catch (InterruptedException ignore) {}
				int max = 0;
				int n = 0;
				for(Integer key : storeMap.keySet()){
					n = storeMap.get(key).size();
					max= Math.max(n, max);
				}
				if (max > maximum)
					maximum = max;

				setProgress(Math.round(100-(float)max/maximum*100));
			}
		}

	}

}
