package org.lemming.pipeline;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.lemming.interfaces.Store;

import ij.IJ;

public class Manager implements Runnable {
	
	private Map<Integer,Store> storeMap = new LinkedHashMap<>();
	private Map<Integer,AbstractModule> modules = new LinkedHashMap<>();
//	private ScheduledExecutorService executor = Executors.newSingleThreadScheduledExecutor();
//	private ScheduledFuture< ? > future;
	private int counter;

	public Manager() {
		counter=0;
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
		
		AbstractModule source = modules.get(fromOrTo.hashCode());
		if (source==null) throw new NullPointerException("Wrong linkage!");
		Store s = new FastStore();
		source.setOutput(s);
		AbstractModule cur = modules.get(current.hashCode());
		if (cur==null) throw new NullPointerException("Wrong linkage!");
		cur.setInput(s);
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
	
	public Map<Integer, Store> get(){
		return storeMap;
	}
	

	@Override
	public void run() {
		if (modules.isEmpty()) return;
		
		List<Thread> threads= new ArrayList<>();
		
		ProgressWindow progressWindow = new ProgressWindow(storeMap);
		
		//future = executor.scheduleAtFixedRate(new MonitorThread(this.storeMap), 100, 1000 ,TimeUnit.MILLISECONDS);
		
		for(AbstractModule starter:modules.values()){
			if (!starter.check()) {
				IJ.error("Module not linked properly " + starter.getClass().getSimpleName());
				break;
			}
		
			Thread t = new Thread(starter, starter.getClass().getSimpleName());
			t.start();
			threads.add(t);	
			
			try {
				Thread.sleep(100); 						// HACK : give the module some time to start working
			} catch (InterruptedException e) {
				e.printStackTrace();
			}

		}
		
		for(Thread joiner:threads){
			try {
				joiner.join();
			} catch (InterruptedException e) {
				System.err.println(e.getMessage());
			}
		}
//		if (future != null && !future.isDone()) {
//			future.cancel(false);
//		}
		progressWindow.cancel();
	}

	public void reset() {
		modules.clear();
		storeMap.clear();
		counter=0;
	}

}
