package org.lemming.pipeline;

import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import org.lemming.interfaces.Element;
import org.lemming.interfaces.Frame;
import org.lemming.interfaces.ModuleInterface;
import org.lemming.interfaces.Store;

/**
 * base class for all modules handling input and output queues
 * 
 * @author Ronny Sczech
 *
 */
public abstract class AbstractModule implements ModuleInterface,Runnable {
	
	protected int numTasks;
	protected int numThreads = Runtime.getRuntime().availableProcessors()-2;
	protected ExecutorService service;
	protected Map<Integer, Store> inputs = new LinkedHashMap<>();
	protected Map<Integer, Store> outputs = new LinkedHashMap<>();
	protected long start;
	protected volatile boolean running = true;
	protected Integer iterator;
	
	
	public AbstractModule(){
		if(service == null)
			service = Executors.newCachedThreadPool();
	}
	
	public void setService(ExecutorService service){
		this.service = service;
	}
	
	public void reset(){
		inputs.clear();
		outputs.clear();
		iterator = null;
	}
	
	protected void newOutput(final Element data) {
		if (outputs.isEmpty()) throw new NullPointerException("No Output Mappings!");
		if (data == null) return;
		Iterator<Integer> it = outputs.keySet().iterator();
		while(it.hasNext()){
			Integer key = it.next();
			try {
				outputs.get(key).put(data);
			} catch (InterruptedException e) {
				break;
			}
		}
	}
	
	protected Element nextInput() {
		return getInput(iterator);
	}

	@Override
	public void cancel() {
		running = false;
		service.shutdown();
	}

	@Override
	public Element getInput(Integer key) {
		Element el = inputs.get(key).poll();
		return el;
	}

	@Override
	public Map<Integer, Element> getInputs() {
		Map<Integer, Element> outMap = new HashMap<>();
		Iterator<Integer> it = inputs.keySet().iterator();
		while(it.hasNext()){
			Integer key = it.next();
			outMap.put(key, inputs.get(key).poll());
		}
		return outMap;
	}

	@Override
	public Element getOutput(Integer key) {
		return outputs.get(key).poll();
	}

	@Override
	public Map<Integer, Element> getOutputs() {
		Map<Integer, Element> outMap = new HashMap<>();
		Iterator<Integer> it = outputs.keySet().iterator();
		while(it.hasNext()){
			Integer key = it.next();
			outMap.put(key, outputs.get(key).poll());
		}
		return outMap;
	}

	@Override
	public void setInput(Integer key, Store store) {
		inputs.put(key, store);		
	}
	
	public void setInput(Store store) {
		inputs.put(store.hashCode(), store);		
	}

	@Override
	public void setInputs(Map<Integer, Store> storeMap) {
		inputs = storeMap;		
	}

	@Override
	public void setOutput(Integer key, Store store) {
		outputs.put(key, store);		
	}
	
	public void setOutput(Store store) {
		outputs.put(store.hashCode(), store);		
	}

	@Override
	public void setOutputs(Map<Integer, Store> storeMap) {
		outputs = storeMap;
	}

	public boolean isRunning() {
		return running;
	}
	
	@SuppressWarnings("static-method")
	protected void pause(long ms){
		try {
			Thread.sleep(ms);
		} catch (InterruptedException e) {
			System.err.println("Pause:"+e.getMessage());
		}
	}
	
	public Element preview(Element el){
		return processData(el);
	}
	
	public <T> Element preview(Frame<T> el){
		return processData(el);
	}	
	
	/**
	 * Method to be overwritten by children of this class.
	 * @param data - data to process
	 * @return Element
	 */
	public abstract Element processData(Element data);
}
