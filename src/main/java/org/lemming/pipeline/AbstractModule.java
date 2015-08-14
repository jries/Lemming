package org.lemming.pipeline;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import org.lemming.interfaces.Element;
import org.lemming.interfaces.ModuleInterface;
import org.lemming.interfaces.Store;

import net.imglib2.algorithm.MultiThreaded;

public abstract class AbstractModule implements ModuleInterface, MultiThreaded {
	
	private int numTasks;
	protected final ExecutorService service;
	protected Map<String, Store> inputs = new HashMap<>();
	protected Map<String, Store> outputs = new HashMap<>();
	
	protected volatile boolean running = true;
	protected String iterator="";
	
	public AbstractModule(){
		setNumThreads();
		service = Executors.newFixedThreadPool(numTasks);
	}
	
	protected void newOutput(final Element data) {
		Store store = outputs.get(iterator);
		if (store==null)
			throw new NullPointerException("wrong mapping!");
		if (data != null)
			store.put(data);
	}
	
	protected Element nextInput() {
		return getInput(iterator);
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
	public Element getInput(String key) {
		Element el = inputs.get(key).get();
		return el;
	}

	@Override
	public Map<String, Element> getInputs() {
		Map<String, Element> outMap = new HashMap<>();
		for (String key : inputs.keySet())
			outMap.put(key, inputs.get(key).get());
		return outMap;
	}

	@Override
	public Element getOutput(String key) {
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
	
	protected static void pause(long ms){
		try {
			Thread.sleep(ms);
		} catch (InterruptedException e) {
			System.err.println(e.getMessage());
		}
	}
}
