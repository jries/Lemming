package org.lemming.modules;

import org.lemming.interfaces.Element;
import org.lemming.pipeline.SingleRunModule;

public class StoreSplitter extends SingleRunModule {

	private Integer counter=0;
	private long start;


	public StoreSplitter() {
	}
	
	@Override
	protected void beforeRun(){
		start = System.currentTimeMillis();
	}

	@Override
	public Element process(Element data) {
		Element el = data; 
		if (el==null) return null;
		
		if(el.isLast()){ //process the rest;
			running = false;
			for (Integer key : outputs.keySet()) {
				Element cloned = el.deepClone();// make a deep copy
				cloned.setLast(true);
				outputs.get(key).put(cloned);
			}
			counter++;
			
			return null;
		}
		
		for (Integer key : outputs.keySet()) {
			Element cloned = el.deepClone();// make a deep copy
			outputs.get(key).put(cloned);
		}
		
		counter++;
		return null;
	}
	
	@Override
	protected void afterRun(){
		System.out.println("Splitting of " + counter +" elements done in " + (System.currentTimeMillis()-start) + "ms.");
	}

	@Override
	public boolean check() {
		// TODO Auto-generated method stub
		return false;
	}

}
