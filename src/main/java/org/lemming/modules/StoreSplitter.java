package org.lemming.modules;

import java.util.Map;

import org.lemming.pipeline.Element;
import org.lemming.pipeline.SingleRunModule;
import org.lemming.pipeline.Store;

public class StoreSplitter<E extends Element> extends SingleRunModule {

	private String inputKey;
	private Store<E> store;
	private Integer counter=0;
	private long start;

	public StoreSplitter(String in) {
		inputKey = in;
	}
	
	@SuppressWarnings("unchecked")
	@Override
	protected void beforeRun(){
		start = System.currentTimeMillis();
		store = inputs.get(inputKey);
		if (store==null)
			throw new NullPointerException("input is empty!");
	}

	@SuppressWarnings("unchecked")
	@Override
	public void process(Map<String, Element> data) {
		E el = (E) data.get(inputKey); 
		if (el==null) return;
		
		if(el.isLast()){ //process the rest;
			for (String key : outputs.keySet()) {
				Element cloned = el.deepClone();// make a deep copy
				cloned.setLast(true);
				outputs.get(key).put(cloned);
			}
			cancel();
			return;
		}
		
		for (String key : outputs.keySet()) {
			Element cloned = el.deepClone();// make a deep copy
			outputs.get(key).put(cloned);
		}
		
		counter++;
		//if (counter % 100 == 0)
		//	System.out.println("Elements finished:"+counter);
	}
	
	@Override
	protected void afterRun(){
		System.out.println("Splitting done " + counter +" in " + (System.currentTimeMillis()-start) + "ms.");
	}

}
