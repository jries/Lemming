package org.lemming.pipeline;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import net.imglib2.algorithm.MultiThreaded;

@SuppressWarnings("rawtypes")
public abstract class AbstractModule implements ModuleInterface, MultiThreaded {
	
	protected volatile boolean running;
	private int numTasks;
	protected final ExecutorService service;
	
	protected Map<String, Store> inputs = new HashMap<>();
	protected Map<String, Store> outputs = new HashMap<>();
	
	public AbstractModule(){
		running = true;
		setNumThreads();
		service = Executors.newFixedThreadPool(numTasks);
	}

	@SuppressWarnings({ "unchecked" })
	protected void newOutput(final String key, Element data) {
		Store store = outputs.get(key);
		if (store==null)
			throw new NullPointerException("wrong mapping!");
		store.put(data);
	}
	
	protected Map<String, Element> nextInput() {
		return getInputs();
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
	public int getNumThreads() {
		return numTasks;
	}

	@Override
	public void cancel() {
		running = false;
		service.shutdown();
	}

	@Override
	public Object getInput(String key) {
		return inputs.get(key).get();
	}

	@Override
	public Map<String, Element> getInputs() {
		Map<String, Element> outMap = new HashMap<>();
		for (String key : inputs.keySet())
			outMap.put(key, inputs.get(key).get());
		return outMap;
	}

	@Override
	public Object getOutput(String key) {
		return outputs.get(key).get();
	}

	@Override
	public Map<String, Element> getOutputs() {
		Map<String, Element> outMap = new HashMap<>();
		for (String key : outputs.keySet())
			outMap.put(key, outputs.get(key).get());
		return outMap;
	}

	@Override
	public void setInput(String key, Store data) {
		inputs.put(key, data);		
	}

	@Override
	public void setInputs(Map<String, Store> storeMap) {
		inputs = storeMap;		
	}

	@Override
	public void setOutput(String key, Store store) {
		outputs.put(key, store);		
	}

	@Override
	public void setOutputs(Map<String, Store> storeMap) {
		outputs = storeMap;
	}

	public boolean isRunning() {
		return running;
	}
}
