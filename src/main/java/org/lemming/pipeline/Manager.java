package org.lemming.pipeline;

import java.beans.PropertyChangeEvent;
import java.beans.PropertyChangeListener;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import javax.swing.SwingWorker;

import org.lemming.interfaces.Store;

import ij.IJ;

/**
 * a swing worker class for background execution of the added modules.
 * The needed queues for data transport are created automatically.
 * 
 * @author Ronny Sczech
 *
 */
public class Manager extends SwingWorker<Void,Void> {
	
	final private Map<Integer,Store> storeMap = new LinkedHashMap<>();
	final private Map<Integer,AbstractModule> modules = new LinkedHashMap<>();
	private boolean done = false;
	private int maximum = 1;
	final private ExecutorService service = Executors.newCachedThreadPool();
 
	public Manager() {
	}
	
	public void add(AbstractModule module){		
		modules.put(module.hashCode(),module);		
	}
	
	public void linkModules(AbstractModule from, AbstractModule to, boolean noInputs, int maxElements ){
		Store s = null;
		if (noInputs){
			int n = (int) Math.min(Math.pow(2,6)*Runtime.getRuntime().availableProcessors(), maxElements*0.5); // performance tweak
			System.out.println("Manager starts with maximal "+n+" elements" );
			s = new LinkedStore(n);
		} else {
			s = new LinkedStore(maxElements);
		}
		AbstractModule source = modules.get(from.hashCode());
		if (source==null) throw new NullPointerException("Wrong linkage!");
		source.setOutput(s);
		AbstractModule well = modules.get(to.hashCode());
		if (well==null) throw new NullPointerException("Wrong linkage!");
		well.setInput(s);
		storeMap.put(s.hashCode(), s);
	}
	
	public void linkModules(AbstractModule from , AbstractModule to ){
		AbstractModule source = modules.get(from.hashCode());
		if (source==null) throw new NullPointerException("Wrong linkage!");
		Store s = new LinkedStore(Integer.MAX_VALUE);
		source.setOutput(s);
		AbstractModule well = modules.get(to.hashCode());
		if (well==null) throw new NullPointerException("Wrong linkage!");
		well.setInput(s);
		storeMap.put(s.hashCode(), s);
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
		final List<Object> threads= new ArrayList<>();
		
		for(AbstractModule starter:modules.values()){
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
		sm.cancel(true);
		return null;
	}
	
	@Override
	public void done(){
		setProgress(0);
		service.shutdown();
	}


	public void reset() {
		modules.clear();
		storeMap.clear();
	}
	
	class StoreMonitor extends SwingWorker<Void,Void> {

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
					n = storeMap.get(key).size();
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
