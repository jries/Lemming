package org.lemming.pipeline;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import net.imglib2.algorithm.Benchmark;
import net.imglib2.algorithm.MultiThreaded;

public abstract class ModuleProcessor implements ModuleInterface,MultiThreaded,Benchmark{
	
	private volatile boolean running;
	private int numTasks;
	private final ExecutorService service;
	protected Map<String, Store<?>> inputs;
	protected Map<String, Store<?>> outputs;
	private long start;
	private long end;
	
	public ModuleProcessor(){
		running = true;
		setNumThreads();
		service = Executors.newFixedThreadPool(numTasks);
		inputs = new HashMap<String, Store<?>>();
		outputs = new HashMap<String, Store<?>>();
	}
	
	
	/**
	 * Method to be overwritten by children of this class.
	 * @param data 
	 */
	public abstract void process(Map<String, Object> data);

	@Override
	public int getNumThreads() {
		return numTasks;
	}

	@Override
	public void setNumThreads() {
		setNumThreads(Runtime.getRuntime().availableProcessors());
		
	}

	@Override
	public void setNumThreads(int numThreads) {
		numTasks=numThreads;
	}

	@Override
	public void cancel() {
		running = false;
		service.shutdownNow();
	}


	@Override
	public Object getInput(String key) {
		return inputs.get(key).get();
	}

	@Override
	public Map<String, Object> getInputs() {
		Map<String, Object> outMap = new HashMap<String, Object>();
		for (String key : inputs.keySet())
			outMap.put(key, inputs.get(key).get());
		return outMap;
	}

	@Override
	public Object getOutput(String key) {
		return outputs.get(key).get();
	}

	@Override
	public Map<String, Object> getOutputs() {
		Map<String, Object> outMap = new HashMap<String, Object>();
		for (String key : outputs.keySet())
			outMap.put(key, outputs.get(key).get());
		return outMap;
	}

	@Override
	public void run() {
		
		start = System.currentTimeMillis();
		
		if (!inputs.isEmpty()){ // first check for existing inputs
			
			beforeRun();
			
			final ArrayList< Future< Void > > futures = new ArrayList< Future< Void > >();
			
			for ( int taskNum = 0; taskNum < numTasks; ++taskNum ){
	
				final Callable< Void > r = new Callable< Void >(){
					
					@Override
					public Void call() {
						while (running) {
								if (Thread.currentThread().isInterrupted()) break;
								Map<String, Object> data = nextInput();
								process(data);
						}
						return null;
					}
					
				};
				futures.add( service.submit( r ) );
			}
			
			for ( final Future< Void > f : futures ){
				try{
					f.get();
				}
				catch ( final InterruptedException | ExecutionException  e ){
					System.err.println(e.getMessage());
				}
			}
			
			afterRun();
			end = System.currentTimeMillis();
			return;
		} else if (!outputs.isEmpty()){ // no inputs but maybe only outputs
			
			beforeRun();
						
			while (hasMoreOutputs()) {
					if (Thread.currentThread().isInterrupted()) break;
					Map<String, Object> data = new HashMap<String, Object>();
					process(data);
					
					for (String key : data.keySet())
						newOutput(key,data.get(key));
			}
			
			afterRun();
			end = System.currentTimeMillis();
			return;
		}
	}

	protected void afterRun() {		
	}


	protected void beforeRun() {		
	}

	private  Map<String, Object> nextInput() {
		return getInputs();
	}
	
	@SuppressWarnings({ "rawtypes", "unchecked" })
	private void newOutput(String key, Object data){
		if (outputs.containsKey(key)){
			Store store = outputs.get(key);
			store.put(data);
		}
	}

	@Override
	public void setInput(String key, Store<?> data) {
		inputs.put(key, (Store<?>) data);
	}
	

	@Override
	public void setInputs(Map<String, Store<?>> storeMap) {
		inputs=storeMap;
	}


	@Override
	public void setOutput(String key, Store<?> store) {
		outputs.put(key, store);
	}


	@Override
	public void setOutputs(Map<String, Store<?>> storeMap) {
		outputs=storeMap;
	}


	@Override
	public long getProcessingTime() {
		return end-start;
	}

}
